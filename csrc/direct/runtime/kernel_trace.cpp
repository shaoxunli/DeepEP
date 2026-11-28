#include "runtime/kernel_trace.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>

#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unistd.h>

#include "common/kernel_trace_layout.h"
#include "common/launch.h"
#include "common/platform.h"

namespace ascend_deepep::trace {
namespace {
constexpr size_t kSlotBytes = platform::kMaxNumAiv * kCoreBytes;
constexpr int64_t kNsPerMs = 1'000'000;
constexpr std::array<std::string_view, KernelOpCounters::kCount> kRecorderKernelNames{
    "notify_dispatch_kernel", "cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel",
    "cached_notify_combine_kernel", "v1_combine_udma_kernel", "v1_combine_udma_comm_kernel",
    "v1_combine_udma_reduce_kernel", "v1_combine_udma_early_kernel", "v1_combine_udma_remaining_kernel",
    "v1_combine_udma_pipeline_kernel",
};
enum class TraceMode { Recorder, Test, TestMetadataOnly };
std::atomic<TraceMode> trace_mode{TraceMode::Recorder};
std::atomic<uint64_t> next_sequence{1};
thread_local TraceMetadataSnapshot active_metadata;
struct NotifyProbeMemory {
    void* host = nullptr;
    void* mapped = nullptr;
    aclrtContext context = nullptr;
    ~NotifyProbeMemory()
    {
        if (!host) return;
        aclrtContext previous = nullptr;
        if (aclrtGetCurrentContext(&previous) != ACL_SUCCESS ||
            aclrtSetCurrentContext(context) != ACL_SUCCESS || aclrtSynchronizeDevice() != ACL_SUCCESS) {
            std::fprintf(stderr, "DeepEP notify probe cleanup failed; retaining device-accessible memory\n");
            if (previous) (void)aclrtSetCurrentContext(previous);
            return;
        }
        if (!mapped || aclrtHostUnregister(host) == ACL_SUCCESS) (void)aclrtFreeHost(host);
        if (previous) (void)aclrtSetCurrentContext(previous);
    }
};
struct HostProbeState {
    std::size_t capacity = 0;
    uint64_t ordinal = 0, dropped = 0;
    int (*gil_state)() = nullptr;
    HostDispatchProbe* active = nullptr;
    std::vector<HostDispatchSample> samples;
    std::unique_ptr<NotifyProbeMemory> device;
};
thread_local HostProbeState host_probe;

template <typename Clock>
int64_t NowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

void Check(aclError status, const char* op)
{
    if (status != ACL_SUCCESS) throw std::runtime_error(std::string(op) + ": " + std::to_string(status));
}

int ReadRecorderMilliseconds(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    const char* end = value + std::strlen(value);
    int milliseconds = 0;
    const auto parsed = std::from_chars(value, end, milliseconds);
    if (parsed.ec != std::errc{} || parsed.ptr != end || milliseconds <= 0)
        throw std::invalid_argument(std::string(name) + " must be a positive integer in milliseconds");
    return milliseconds;
}
}  // namespace

namespace {
void JsonString(std::string& out, const std::string& value)
{
    constexpr char hex[] = "0123456789abcdef";
    out += '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') { out += '\\'; out += ch; }
        else if (ch < 0x20) { out += "\\u00"; out += hex[ch >> 4]; out += hex[ch & 15]; }
        else out += ch; // Preserve UTF-8; the Python file is opened with encoding="utf-8".
    }
    out += '"';
}

std::optional<uint64_t> ParseUnsigned(std::string_view text)
{
    if (text.empty()) return std::nullopt;
    uint64_t value{};
    const auto end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) return std::nullopt;
    return value;
}

bool IsCanonicalCommId(std::string_view comm_id)
{
    return !comm_id.empty() && (comm_id.size() == 1 || comm_id.front() != '0') &&
           ParseUnsigned(comm_id).has_value();
}

std::string ProductionWallTime(int64_t ns)
{
    constexpr int64_t kNsPerSecond = 1'000'000'000;
    constexpr int64_t kUtcOffsetSeconds = 8 * 3600;
    auto seconds = ns / kNsPerSecond;
    auto fraction = ns % kNsPerSecond;
    if (fraction < 0) {
        --seconds;
        fraction += kNsPerSecond;
    }
    const auto timestamp = static_cast<std::time_t>(seconds + kUtcOffsetSeconds);
    std::tm calendar{};
    char result[32]{};
    if (!gmtime_r(&timestamp, &calendar)) throw std::runtime_error("cannot convert trace wall time");
    const auto size = std::strftime(result, sizeof(result), "%Y-%m-%d %H:%M:%S", &calendar);
    if (!size) throw std::runtime_error("cannot format trace wall time");
    std::snprintf(result + size, sizeof(result) - size, ".%06d", static_cast<int>(fraction / 1000));
    return std::string("\"") + result + '"';
}

void AppendProductionRecord(std::string& jsonl, const TraceRecord& record, std::string_view comm_id, uint32_t bus_id)
{
    const auto value = [&](const char* name) -> std::optional<int64_t> {
        const auto it = record.values.find(name);
        if (it == record.values.end()) return std::nullopt;
        return it->second;
    };
    const bool completed = record.event == "end";
    jsonl += "{\"type\":";
    jsonl += completed ? '2' : '1';
    const auto field = [&](const char* name, std::string_view json) {
        jsonl += ",\"";
        jsonl += name;
        jsonl += "\":";
        jsonl += json;
    };
    const auto number = [&](const char* name, std::optional<int64_t> n) {
        if (n) field(name, std::to_string(*n));
        else field(name, "null");
    };
    const auto time = [&](const char* name, std::optional<int64_t> ns) {
        if (ns) field(name, ProductionWallTime(*ns));
        else field(name, "null");
    };

    const auto op = value("op");
    const bool short_format = op == 0 || op == 1 || op == 3 || op == 6;
    number("gid", value("gid"));
    number("op", op);
    number("op_count", value("op_count"));
    number("dispatch_count", value("dispatch_count"));
    if (!short_format) number("ep_size", value("ep_size"));
    number("bus_id", bus_id);
    const auto rank = value("rank");
    number("rank", rank);
    number("nvl_rank", rank ? std::optional<int64_t>{*rank % 16} : std::nullopt);
    if (!short_format) {
        number("hidden_size", value("hidden_size"));
        number("num_experts", value("num_experts"));
    }
    number("num_sms", value("aiv"));
    if (!short_format) {
        number("num_tokens", value("num_tokens"));
        number("topk", value("topk"));
    }
    number("pid", value("pid"));
    const auto stream = record.labels.find("stream");
    const auto handle = stream == record.labels.end() ? std::nullopt : ParseUnsigned(stream->second);
    field("stream", handle ? std::to_string(*handle) : "null");
    field("comm_id", comm_id);
    time("launch_time", value("launch_ns"));
    time("start_time", completed ? value("start_ns") : std::nullopt);
    time("end_time", completed ? value("end_ns") : std::nullopt);
    number("elapsed_ns", completed ? value("elapsed_ns") : std::nullopt);
    number("start_tick", completed ? value("start_tick") : std::nullopt);
    number("end_tick", completed ? value("end_tick") : std::nullopt);
    jsonl += "}\n";
}
}
std::string EncodeTraceRecords(const std::vector<TraceRecord>& records)
{
    std::string out;
    out.reserve(records.size() * 1024);
    for (const auto& record : records) {
        out += '{';
        bool first = true;
        auto key = [&](const std::string& name) {
            if (!first) out += ',';
            first = false; JsonString(out, name); out += ':';
        };
        auto label = [&](const std::string& name, const std::string& value) {
            if (record.values.count(name)) return;
            key(name); JsonString(out, value);
        };
        label("event", record.event);
        if (!record.op_name.empty()) label("op_name", record.op_name);
        if (!record.kernel_name.empty()) label("kernel_name", record.kernel_name);
        for (const auto& item : record.labels) {
            if (item.first == "event" || (item.first == "op_name" && !record.op_name.empty()) ||
                (item.first == "kernel_name" && !record.kernel_name.empty())) continue;
            label(item.first, item.second);
        }
        for (const auto& item : record.values) { key(item.first); out += std::to_string(item.second); }
        out += "}\n";
    }
    return out;
}

TraceMetadataSnapshot CurrentTraceMetadata()
{
    return active_metadata;
}
TraceMetadataSnapshot MakeCallMetadata(TraceMetadataSnapshot parent, int64_t api, int64_t handle, int64_t dispatch_count)
{
    auto value = std::make_shared<TraceMetadata>();
    value->parent = std::move(parent);
    value->has_call_ids = true;
    value->api_call_id = api;
    value->handle_id = handle;
    value->dispatch_count = dispatch_count;
    return value;
}
TraceMetadataSnapshot ExchangeTraceMetadata(TraceMetadataSnapshot metadata)
{
    std::swap(metadata, active_metadata);
    return metadata;
}
ScopedTraceMetadata::ScopedTraceMetadata(TraceMetadataSnapshot metadata)
    : previous_(ExchangeTraceMetadata(std::move(metadata)))
{
}
ScopedTraceMetadata::~ScopedTraceMetadata()
{
    ExchangeTraceMetadata(std::move(previous_));
}

void ConfigureTraceTestMode(bool metadata_only)
{
    trace_mode.store(metadata_only ? TraceMode::TestMetadataOnly : TraceMode::Test, std::memory_order_relaxed);
}

std::shared_ptr<KernelTrace> KernelTrace::Open()
{
    const char* enable = std::getenv("DEEPEP_TRACE_RECORDER_ENABLE");
    if (!enable || std::strcmp(enable, "1") != 0) return {};
    const auto mode = trace_mode.load(std::memory_order_relaxed);
    if (mode == TraceMode::TestMetadataOnly) return {};
    try {
        static std::mutex mutex;
        static std::map<aclrtContext, std::weak_ptr<KernelTrace>> sinks;
        std::lock_guard<std::mutex> lock(mutex);
        aclrtContext context{};
        int32_t device{};
        Check(aclrtGetCurrentContext(&context), "get context");
        Check(aclrtGetDevice(&device), "get device");
        if (auto alive = sinks[context].lock()) return alive;
        auto sink = std::shared_ptr<KernelTrace>(new KernelTrace);
        sink->context_ = context;
        sink->recorder_mode_ = mode == TraceMode::Recorder;
        sink->Init(device);
        sinks[context] = sink;
        return sink;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "DeepEP kernel trace disabled: %s\n", e.what());
        return {};
    }
}

void KernelTrace::Init(int device)
{
    device_ = device;
    if (recorder_mode_) {
        poll_interval_ms_ = ReadRecorderMilliseconds("DEEPEP_TRACE_POLL_INTERVAL_MS", poll_interval_ms_);
        timeout_ms_ = ReadRecorderMilliseconds("DEEPEP_TRACE_TIMEOUT_MS", timeout_ms_);
    }
    const char* value = recorder_mode_ ? nullptr : std::getenv("DEEPEP_TRACE_SLOTS");
    if (value) {
        char* end{};
        const auto count = std::strtoul(value, &end, 10);
        if (!*value || *value < '0' || *value > '9' || *end || count < 128 || count > 16384)
            throw std::runtime_error("DEEPEP_TRACE_SLOTS must be an integer in [128, 16384]");
        slot_count_ = static_cast<unsigned>(count);
    }
    slots_ = std::make_unique<Slot[]>(slot_count_);
    Check(aclrtMallocHost(&host_, slot_count_ * kSlotBytes), "allocate trace ring");
    std::memset(host_, 0, slot_count_ * kSlotBytes);
    Check(aclrtHostRegister(host_, slot_count_ * kSlotBytes, ACL_HOST_REGISTER_MAPPED, &mapped_), "map trace ring");

    aclrtStream stream{};
    Check(aclrtCreateStream(&stream), "create calibration stream");
    const auto before = NowNs<std::chrono::steady_clock>();
    device_accessed_ = true;
    kernel_trace_clock_do(stream, static_cast<uint8_t*>(mapped_));
    const auto synced = aclrtSynchronizeStream(stream);
    const auto after = NowNs<std::chrono::steady_clock>();
    const auto destroyed = aclrtDestroyStream(stream);
    Check(synced, "calibrate device clock");
    Check(destroyed, "destroy calibration stream");
    auto* first = static_cast<uint64_t*>(host_);
    const auto tick = __atomic_exchange_n(first, uint64_t{0}, __ATOMIC_ACQ_REL);
    if (!tick) throw std::runtime_error("calibration kernel did not publish a tick");
    const auto anchor0 = NowNs<std::chrono::steady_clock>();
    const auto wall = NowNs<std::chrono::system_clock>();
    const auto anchor1 = NowNs<std::chrono::steady_clock>();
    offset_ns_ =
        wall - (anchor0 + (anchor1 - anchor0) / 2) + before + (after - before) / 2 - static_cast<int64_t>(tick);
    calibration_tick_ = tick;
    window_ns_ = after - before;

}

void ConfigureHostDispatchProbe(std::size_t capacity, int (*gil_state)(), bool device_timing)
{
    if (host_probe.active) throw std::runtime_error("cannot configure an active Host dispatch probe");
    if (host_probe.device) throw std::runtime_error("take previous notify device probe before configuring again");
    if (capacity > 1000000) throw std::runtime_error("Host dispatch probe capacity exceeds 1000000");
    std::unique_ptr<NotifyProbeMemory> device;
    if (device_timing && capacity) {
        device = std::make_unique<NotifyProbeMemory>();
        Check(aclrtGetCurrentContext(&device->context), "notify probe context");
        Check(aclrtMallocHost(&device->host, capacity * kCoreBytes), "allocate notify probe");
        std::memset(device->host, 0, capacity * kCoreBytes);
        Check(aclrtHostRegister(device->host, capacity * kCoreBytes, ACL_HOST_REGISTER_MAPPED, &device->mapped),
              "map notify probe");
    }
    host_probe.samples.clear();
    host_probe.samples.reserve(capacity);
    host_probe.capacity = capacity;
    host_probe.ordinal = host_probe.dropped = 0;
    host_probe.gil_state = gil_state;
    host_probe.device = std::move(device);
}

HostDispatchReport TakeHostDispatchProbe()
{
    if (host_probe.active) throw std::runtime_error("cannot take an active Host dispatch probe");
    if (host_probe.device) {
        aclrtContext current{};
        Check(aclrtGetCurrentContext(&current), "take notify probe context");
        if (current != host_probe.device->context) throw std::runtime_error("notify probe context changed");
        // Only at explicit export, outside timed workload. Do not free a device writer's memory early.
        Check(aclrtSynchronizeDevice(), "complete notify probe before export");
        for (auto& sample : host_probe.samples) {
            if (!sample.notify_sequence) continue;
            const auto* line = reinterpret_cast<const uint64_t*>(
                static_cast<const uint8_t*>(host_probe.device->host) + (sample.ordinal - 1) * kCoreBytes);
            sample.notify_done = __atomic_load_n(line + kNotifyProbeDoneOffset / sizeof(uint64_t), __ATOMIC_ACQUIRE);
            if (sample.notify_done == sample.notify_sequence)
                std::copy_n(line, sample.notify_ticks.size(), sample.notify_ticks.begin());
        }
    }
    HostDispatchReport report{std::move(host_probe.samples), host_probe.dropped};
    host_probe.capacity = 0;
    host_probe.gil_state = nullptr;
    host_probe.device.reset();
    return report;
}

uint64_t HostDispatchProbe::NotifyAddress()
{
    if (!enabled_ || !host_probe.device || sample_.ordinal > host_probe.capacity) return 0;
    sample_.notify_sequence = sample_.ordinal;
    sample_.notify_address = reinterpret_cast<uint64_t>(host_probe.device->mapped) +
                             (sample_.ordinal - 1) * kCoreBytes;
    return sample_.notify_address;
}

HostDispatchProbe* CurrentHostDispatchProbe() { return host_probe.active; }

HostDispatchProbe::HostDispatchProbe(int rank, bool cached)
{
    if (!host_probe.capacity) return;
    enabled_ = true;
    exceptions_ = std::uncaught_exceptions();
    sample_.ordinal = ++host_probe.ordinal;
    sample_.rank = rank;
    sample_.cached = cached;
    sample_.wait_path = 0;
    sample_.api_call_id = -1;
    sample_.success = false;
    sample_.ticks.fill(-1);
    sample_.gil.fill(-1);
    previous_ = host_probe.active;
    host_probe.active = this;
    Mark(DispatchProbePoint::NativeEntry);
    for (auto metadata = CurrentTraceMetadata(); metadata; metadata = metadata->parent) {
        if (metadata->has_call_ids) { sample_.api_call_id = metadata->api_call_id; break; }
        auto found = metadata->values.find("api_call_id");
        if (found != metadata->values.end()) { sample_.api_call_id = found->second; break; }
    }
}

void HostDispatchProbe::MarkEnabled(DispatchProbePoint point)
{
    const auto index = static_cast<unsigned>(point);
    sample_.ticks[index] = NowNs<std::chrono::steady_clock>();
    sample_.gil[index] = host_probe.gil_state ? host_probe.gil_state() : -1;
}

HostDispatchProbe::~HostDispatchProbe()
{
    if (!enabled_) return;
    sample_.success = sample_.success && std::uncaught_exceptions() == exceptions_;
    Mark(DispatchProbePoint::NativeExit);
    host_probe.active = previous_;
    if (host_probe.samples.size() < host_probe.capacity) host_probe.samples.push_back(sample_);
    else ++host_probe.dropped;
}

KernelTrace::Ticket KernelTrace::Reserve(int rank, uint32_t instance, uint32_t cores, const char* op_name,
                                       const char* kernel_name, uint64_t stream, uint32_t ep_size,
                                       int64_t input_rows, int64_t output_rows)
{
    const auto& metadata = active_metadata;
    int op = -1;
    int64_t op_count = -1;
    if (recorder_mode_) {
        const auto* scope = metadata.get();
        while (scope && !scope->op_counters) scope = scope->parent.get();
        if (!scope) return {};
        const auto comm_id = scope->labels.find("comm_id");
        if (comm_id == scope->labels.end() || comm_id->second.empty()) return {};
        const auto kernel = std::find(kRecorderKernelNames.begin(), kRecorderKernelNames.end(), std::string_view(kernel_name));
        if (kernel == kRecorderKernelNames.end()) return {};
        op = static_cast<int>(kernel - kRecorderKernelNames.begin());
        op_count = scope->op_counters->next[op].fetch_add(1, std::memory_order_relaxed);
    }
    const auto index = cursor_.fetch_add(1, std::memory_order_relaxed) % slot_count_;
    auto& slot = slots_[index];
    auto expected = State::Free;
    if (cores == 0 || cores > platform::kMaxNumAiv ||
        !slot.state.compare_exchange_strong(expected, State::Reserved, std::memory_order_acquire)) {
        ++dropped_;
        return {};
    }
    slot.sequence = next_sequence.fetch_add(1, std::memory_order_relaxed);
    // Zero is the uninitialized marker; JSON exposes the sequence as int64.
    if (slot.sequence == 0 || slot.sequence > uint64_t(std::numeric_limits<int64_t>::max())) {
        slot.state.store(State::Free, std::memory_order_release);
        ++dropped_;
        return {};
    }
    slot.op_name = op_name;
    slot.kernel_name = kernel_name;
    slot.rank = rank;
    slot.instance = instance;
    slot.cores = cores;
    slot.op = op;
    slot.op_count = op_count;
    slot.metadata = metadata;
    slot.stream = stream;
    slot.ep_size = ep_size;
    slot.input_rows = input_rows;
    slot.output_rows = output_rows;
    slot.logged_launch = false;
    slot.logged_timeout = false;
    return {reinterpret_cast<uint64_t>(mapped_) + index * kSlotBytes, index, slot.sequence};
}

void KernelTrace::Submit(Ticket ticket, int64_t launch_ns, int status)
{
    auto& slot = slots_[ticket.index];
    slot.launch_ns = launch_ns;
    slot.status = status;
    if (recorder_mode_ && status == ACL_SUCCESS) slot.submit_steady_ns = NowNs<std::chrono::steady_clock>();
    slot.state.store(State::Submitted, std::memory_order_release);
}

TraceRecord KernelTrace::Calibration() const
{
    return {"calibration", "", "", {{"device", device_}, {"pid", getpid()}, {"tick_hz", kTickHz},
        {"tick", static_cast<int64_t>(calibration_tick_)}, {"offset_ns", offset_ns_}, {"window_ns", window_ns_},
        {"slots", slot_count_}, {"slot_bytes", kSlotBytes}}};
}

void KernelTrace::Synchronize()
{
    aclrtContext previous{};
    Check(aclrtGetCurrentContext(&previous), "get context");
    Check(aclrtSetCurrentContext(context_), "select trace context");
    const auto status = aclrtSynchronizeDevice();
    const auto restored = aclrtSetCurrentContext(previous);
    Check(status, "drain trace device");
    Check(restored, "restore trace context");
}

TraceRecord KernelTrace::DescribeSlot(const Slot& slot) const
{
    TraceRecord record{"", slot.op_name, slot.kernel_name,
        {{"gid", static_cast<int64_t>(slot.sequence)}, {"rank", slot.rank},
         {"local_instance", slot.instance}, {"launch_ns", slot.launch_ns}, {"aiv", slot.cores},
         {"pid", getpid()}, {"device", device_}}};
    if (slot.op >= 0) record.values["op"] = slot.op;
    if (slot.op_count >= 0) record.values["op_count"] = slot.op_count;
    if (slot.ep_size) record.values["ep_size"] = slot.ep_size;
    if (slot.input_rows >= 0) record.values["kernel_input_rows"] = slot.input_rows;
    if (slot.output_rows >= 0) record.values["kernel_output_rows"] = slot.output_rows;
    for (auto metadata = slot.metadata; metadata; metadata = metadata->parent) {
        if (metadata->has_call_ids) {
            record.values.emplace("api_call_id", metadata->api_call_id);
            record.values.emplace("handle_id", metadata->handle_id);
            if (metadata->dispatch_count >= 0) record.values.emplace("dispatch_count", metadata->dispatch_count);
        }
        record.values.insert(metadata->values.begin(), metadata->values.end());
        record.labels.insert(metadata->labels.begin(), metadata->labels.end());
    }
    record.labels["stream"] = std::to_string(slot.stream);
    return record;
}

std::vector<TraceRecord> KernelTrace::Drain(bool closing, bool recorder)
{
    std::lock_guard<std::mutex> lock(drain_mutex_);
    std::vector<TraceRecord> records;
    const auto poll_ns = recorder ? NowNs<std::chrono::steady_clock>() : 0;
    const auto timeout_ns = static_cast<int64_t>(timeout_ms_) * kNsPerMs;
    for (unsigned index = 0; index < slot_count_; ++index) {
        auto& slot = slots_[index];
        if (slot.state.load(std::memory_order_acquire) != State::Submitted) continue;
        TraceRecord record;
        if (!recorder) record = DescribeSlot(slot);
        if (!recorder && !slot.logged_launch) {
            record.event = slot.status ? "launch_error" : "launch";
            record.values["status"] = slot.status;
            records.push_back(record);
            slot.logged_launch = true;
        }
        if (slot.status != ACL_SUCCESS) {
            slot.state.store(State::Quarantined, std::memory_order_release);
            ++incomplete_;
            continue;
        }
        auto* base = static_cast<uint8_t*>(host_) + index * kSlotBytes;
        uint64_t first = std::numeric_limits<uint64_t>::max(), last = 0;
        bool ready = true;
        for (uint32_t core = 0; core < slot.cores; ++core) {
            auto* line = base + core * kCoreBytes;
            const auto sequence = __atomic_load_n(
                reinterpret_cast<uint64_t*>(line + kLineBytes), __ATOMIC_ACQUIRE);
            if (sequence != slot.sequence) {
                ready = false;
                break;
            }
            const auto start = __atomic_load_n(reinterpret_cast<uint64_t*>(line), __ATOMIC_RELAXED);
            const auto end = __atomic_load_n(reinterpret_cast<uint64_t*>(line + sizeof(uint64_t)), __ATOMIC_RELAXED);
            first = std::min(first, start);
            last = std::max(last, end);
        }
        if (!ready) {
            if (recorder && recorder_mode_ && !slot.logged_timeout &&
                poll_ns - slot.submit_steady_ns >= timeout_ns) {
                record = DescribeSlot(slot);
                record.event = "timeout";
                records.push_back(std::move(record));
                slot.logged_timeout = true;
            }
            if (closing) {
                if (!recorder) {
                    record.event = "incomplete";
                    records.push_back(std::move(record));
                }
                slot.state.store(State::Quarantined, std::memory_order_release);
                ++incomplete_;
            }
            continue;
        }
        if (recorder) record = DescribeSlot(slot);
        record.event = "end";
        record.values.insert({{"start_tick", first}, {"end_tick", last},
            {"start_ns", offset_ns_ + static_cast<int64_t>(first)},
            {"end_ns", offset_ns_ + static_cast<int64_t>(last)}, {"elapsed_ns", last - first}});
        records.push_back(std::move(record));
        ++completed_;
        // All producers published this generation. Keep old bytes; sequence identifies reuse.
        slot.state.store(State::Free, std::memory_order_release);
    }
    if (closing) {
        const auto dropped = dropped_.load();
        if (!recorder) {
            records.push_back({"summary", "", "", {{"completed", completed_},
                {"incomplete", incomplete_}, {"dropped", dropped}}});
        } else if (incomplete_ || dropped) {
            std::fprintf(stderr, "DeepEP trace recorder device %d: incomplete=%llu dropped=%llu\n", device_,
                static_cast<unsigned long long>(incomplete_), static_cast<unsigned long long>(dropped));
        }
    }
    return records;
}

std::map<std::string, std::string> KernelTrace::DrainProductionBatches(uint32_t bus_id, bool closing)
{
    std::map<std::string, std::string> batches;
    for (const auto& record : Drain(closing, true)) {
        if (record.event != "end" && record.event != "timeout") continue;
        const auto comm_id = record.labels.find("comm_id");
        if (comm_id == record.labels.end() || !IsCanonicalCommId(comm_id->second)) continue;
        AppendProductionRecord(batches[comm_id->second], record, comm_id->second, bus_id);
    }
    return batches;
}

KernelTrace::~KernelTrace()
{
    aclrtContext previous{};
    const bool have_previous = aclrtGetCurrentContext(&previous) == ACL_SUCCESS;
    const bool current = have_previous && previous == context_;
    const bool selected = current || aclrtSetCurrentContext(context_) == ACL_SUCCESS;
    const bool safe = !device_accessed_ || (selected && aclrtSynchronizeDevice() == ACL_SUCCESS);
    // If pending writes cannot be ruled out, leave the mapping to process teardown.
    if (safe && selected && (!mapped_ || aclrtHostUnregister(host_) == ACL_SUCCESS)) {
        if (host_) aclrtFreeHost(host_);
    } else if (host_) {
        std::fprintf(stderr, "DeepEP kernel trace retained an undrained host mapping\n");
    }
    if (!current && selected && have_previous) aclrtSetCurrentContext(previous);
}
}  // namespace ascend_deepep::trace
