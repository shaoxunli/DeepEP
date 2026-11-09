#pragma once

#include "kernel_operator.h"
#include "common/types.h"
#include "common/udma_schedule.h"

namespace ascend_deepep::udma {
// Diagnostic counters are per AIV and per invocation. Each AIV writes its own
// GM row. Nested sender counters must not be added to send_ns.
struct DispatchProfile {
    __aicore__ explicit DispatchProfile(bool active) : enabled(active) {}
    bool enabled;
    bool detailed = true;
    uint64_t entry_cycle = 0, self_ns = 0, output_ns = 0, total_ns = 0;
    bool trace = false;
    uint32_t rank = 0, world = 0, cores = 0, markers = 0, qps = 0;
    uint64_t origin_cycle = 0, remote_wait_ns = 0, local_sync_ns = 0;
    uint64_t credit_checks = 0, credit_stalls = 0, send_records = 0;
    uint64_t first_poll[kTraceMaxIncoming], ready_poll[kTraceMaxIncoming], poll_count[kTraceMaxIncoming];
    __aicore__ uint32_t Stride() const {
        return trace ? DispatchTraceStride(world, cores, markers, qps) : kDispatchProfileWords;
    }
    __aicore__ uint64_t Elapsed() const { return Now() - origin_cycle; }
    uint64_t begin_ns = 0, metadata_ns = 0, setup_ns = 0, plan_ns = 0, send_ns = 0;
    uint64_t build_ns = 0, credit_ns = 0, submit_ns = 0, publish_ns = 0;
    uint64_t incoming_ns = 0, quiet_ns = 0, barrier_ns = 0;
    uint64_t refills = 0, hits = 0, wqes = 0, batches = 0;
    uint64_t scale_pack_ns = 0, scale_submit_ns = 0, scale_wait_ns = 0;

    __aicore__ uint64_t Now() const {
        return enabled ? static_cast<uint64_t>(AscendC::GetSystemCycle()) : 0;
    }
    __aicore__ void Add(uint64_t& counter, uint64_t start) {
        if (enabled) counter += Now() - start;
    }
    __aicore__ void Store(uint64_t output, uint32_t core, AscendC::LocalTensor<uint64_t> scratch) const {
        if (!enabled) return;
        if (!output) { trap(); return; }
        scratch.SetValue(0, begin_ns); scratch.SetValue(1, metadata_ns);
        scratch.SetValue(2, setup_ns); scratch.SetValue(3, plan_ns);
        scratch.SetValue(4, send_ns); scratch.SetValue(5, build_ns);
        scratch.SetValue(6, credit_ns); scratch.SetValue(7, submit_ns);
        scratch.SetValue(8, publish_ns); scratch.SetValue(9, incoming_ns);
        scratch.SetValue(10, quiet_ns); scratch.SetValue(11, barrier_ns);
        scratch.SetValue(12, refills); scratch.SetValue(13, hits);
        scratch.SetValue(14, wqes); scratch.SetValue(15, batches);
        scratch.SetValue(16, origin_cycle); scratch.SetValue(17, remote_wait_ns);
        scratch.SetValue(18, local_sync_ns); scratch.SetValue(19, credit_checks);
        scratch.SetValue(20, credit_stalls);
        scratch.SetValue(21, send_records);
        scratch.SetValue(22, scale_pack_ns); scratch.SetValue(23, scale_submit_ns);
        scratch.SetValue(24, scale_wait_ns);
        scratch.SetValue(25, self_ns); scratch.SetValue(26, output_ns);
        scratch.SetValue(27, total_ns);
        for (uint32_t i = 28; i < kDispatchProfileWords; ++i) scratch.SetValue(i, 0);
        AscendC::GlobalTensor<uint64_t> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(output) + uint64_t(core) * Stride());
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID5);
        AscendC::DataCopyPad(out, scratch,
            AscendC::DataCopyExtParams{1, kDispatchProfileWords * uint32_t(sizeof(uint64_t)), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        if (!trace) return;
        for (uint32_t item = core; item < world * markers; item += cores) {
            if (item / markers == rank) continue;
            const uint32_t slot = item / cores;
            scratch.SetValue(0, item / markers); scratch.SetValue(1, item % markers);
            scratch.SetValue(2, first_poll[slot]); scratch.SetValue(3, ready_poll[slot]);
            scratch.SetValue(4, poll_count[slot]);
            for (uint32_t i = 5; i < kTraceReceiveWords; ++i) scratch.SetValue(i, 0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID5);
            AscendC::DataCopyPad(out[kDispatchProfileWords + world * qps * kTraceSendWords + slot * kTraceReceiveWords],
                scratch, AscendC::DataCopyExtParams{1, kTraceReceiveWords * uint32_t(sizeof(uint64_t)), 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        }
    }
};
} // namespace ascend_deepep::udma
