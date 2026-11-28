#pragma once
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <acl/acl.h>
#include "common/kernel_trace_layout.h"

namespace ascend_deepep::trace {
struct KernelOpCounters {
    static constexpr std::size_t kCount = 10;
    KernelOpCounters()
    {
        for (auto& count : next) count.store(0, std::memory_order_relaxed);
    }
    std::array<std::atomic<int64_t>, kCount> next;
};

struct TraceMetadata {
    std::shared_ptr<const TraceMetadata> parent;
    std::map<std::string, int64_t> values;
    std::map<std::string, std::string> labels;
    bool has_call_ids = false;
    int64_t api_call_id = 0, handle_id = 0;
    int64_t dispatch_count = -1;
    std::shared_ptr<KernelOpCounters> op_counters;
};
using TraceMetadataSnapshot = std::shared_ptr<const TraceMetadata>;
TraceMetadataSnapshot MakeCallMetadata(TraceMetadataSnapshot parent, int64_t api, int64_t handle,
                                      int64_t dispatch_count = -1);
TraceMetadataSnapshot CurrentTraceMetadata();
TraceMetadataSnapshot ExchangeTraceMetadata(TraceMetadataSnapshot metadata);
class ScopedTraceMetadata {
public:
    explicit ScopedTraceMetadata(TraceMetadataSnapshot metadata);
    ~ScopedTraceMetadata();
    ScopedTraceMetadata(const ScopedTraceMetadata&) = delete;
    ScopedTraceMetadata& operator=(const ScopedTraceMetadata&) = delete;
private:
    TraceMetadataSnapshot previous_;
};
struct TraceRecord {
    std::string event, op_name, kernel_name;
    std::map<std::string, int64_t> values;
    std::map<std::string, std::string> labels;
};
std::string EncodeTraceRecords(const std::vector<TraceRecord>& records);
void ConfigureTraceTestMode(bool metadata_only);
// Opt-in Host diagnostics. Fixed-size samples are reserved before the workload and read afterwards.
enum class DispatchProbePoint : unsigned {
    NativeEntry, NotifyBegin, NotifyEnd, WaitBegin, FirstPending, FirstCountReady, AllCountsReady,
    WaitEnd, OutputsBegin, OutputsEnd, DispatchBegin, DispatchEnd, NativeExit, Count
};
constexpr unsigned kDispatchProbePoints = static_cast<unsigned>(DispatchProbePoint::Count);
struct HostDispatchSample {
    uint64_t ordinal;
    int rank, cached, wait_path; // 1=mapped polling, 2=Tensor.cpu, 3=fixed capacity, 4=cached.
    int64_t api_call_id;
    bool success;
    std::array<int64_t, kDispatchProbePoints> ticks, gil;
    uint64_t notify_address = 0, notify_sequence = 0, notify_done = 0;
    std::array<uint64_t, kNotifyProbeWords> notify_ticks{};
};
struct HostDispatchReport {
    std::vector<HostDispatchSample> samples;
    uint64_t dropped;
};
void ConfigureHostDispatchProbe(std::size_t capacity, int (*gil_state)(), bool device_timing = false);
HostDispatchReport TakeHostDispatchProbe();
class HostDispatchProbe {
public:
    HostDispatchProbe(int rank, bool cached);
    ~HostDispatchProbe();
    HostDispatchProbe(const HostDispatchProbe&) = delete;
    HostDispatchProbe& operator=(const HostDispatchProbe&) = delete;
    void Mark(DispatchProbePoint point) { if (enabled_) MarkEnabled(point); }
    void SetWaitPath(int path) { if (enabled_) sample_.wait_path = path; }
    void Finish() { if (enabled_) sample_.success = true; }
    uint64_t NotifyAddress();
    uint64_t NotifySequence() const { return enabled_ ? sample_.notify_sequence : 0; }
    bool Has(DispatchProbePoint point) const { return enabled_ && sample_.ticks[static_cast<unsigned>(point)] >= 0; }
private:
    void MarkEnabled(DispatchProbePoint point);
    bool enabled_ = false;
    int exceptions_;
    HostDispatchProbe* previous_ = nullptr;
    HostDispatchSample sample_;
};
HostDispatchProbe* CurrentHostDispatchProbe();

class KernelTrace {
public:
    static std::shared_ptr<KernelTrace> Open();
    ~KernelTrace();
    struct Ticket {
        uint64_t address = 0;
        unsigned index = 0;
        uint64_t sequence = 0;
    };
    Ticket Reserve(int rank, uint32_t instance, uint32_t cores,
                   const char* op_name, const char* kernel_name, uint64_t stream = 0, uint32_t ep_size = 0,
                   int64_t input_rows = -1, int64_t output_rows = -1);
    void Submit(Ticket ticket, int64_t launch_ns, int status);
    std::vector<TraceRecord> Drain(bool closing = false, bool recorder = false);
    // Each communication group receives its ordered batch of production JSONL records.
    std::map<std::string, std::string> DrainProductionBatches(uint32_t bus_id, bool closing = false);
    int Device() const { return device_; }
    int PollIntervalMs() const { return poll_interval_ms_; }
    TraceRecord Calibration() const;
    void Synchronize();
private:
    enum class State { Free, Reserved, Submitted, Quarantined };
    struct Slot {
        std::atomic<State> state{State::Free};
        uint64_t sequence{};
        int64_t launch_ns{};
        int64_t submit_steady_ns{};
        int rank{}, status{};
        int op = -1;
        int64_t op_count = -1;
        uint32_t instance{}, cores{};
        std::string op_name, kernel_name;
        TraceMetadataSnapshot metadata;
        uint64_t stream{};
        uint32_t ep_size{};
        int64_t input_rows{-1}, output_rows{-1};
        bool logged_launch{};
        bool logged_timeout{};
    };
    TraceRecord DescribeSlot(const Slot& slot) const;
    KernelTrace() = default;
    void Init(int device);
    bool recorder_mode_ = true;
    int poll_interval_ms_ = 2;
    int timeout_ms_ = 10000;
    unsigned slot_count_ = 128;
    std::unique_ptr<Slot[]> slots_;
    std::mutex drain_mutex_;
    std::atomic<unsigned> cursor_{0};
    std::atomic<uint64_t> dropped_{0};
    void* host_{};
    void* mapped_{};
    aclrtContext context_{};
    bool device_accessed_{};
    int device_{};
    int64_t offset_ns_{}, window_ns_{};
    uint64_t calibration_tick_{}, completed_{}, incomplete_{};
};
}
