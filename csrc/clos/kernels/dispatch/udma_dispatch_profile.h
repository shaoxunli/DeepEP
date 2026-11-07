#pragma once

#include "kernel_operator.h"
#include "common/types.h"

namespace ascend_deepep::udma {
// Diagnostic counters are per AIV and per invocation. Each AIV writes its own
// GM row. Nested sender counters must not be added to send_ns.
struct DispatchProfile {
    bool enabled;
    uint64_t begin_ns = 0, metadata_ns = 0, setup_ns = 0, plan_ns = 0, send_ns = 0;
    uint64_t build_ns = 0, credit_ns = 0, submit_ns = 0, publish_ns = 0;
    uint64_t incoming_ns = 0, quiet_ns = 0, barrier_ns = 0;
    uint64_t refills = 0, hits = 0, wqes = 0, batches = 0;

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
        for (uint32_t i = 16; i < kDispatchProfileWords; ++i) scratch.SetValue(i, 0);
        AscendC::GlobalTensor<uint64_t> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(output) + uint64_t(core) * kDispatchProfileWords);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID5);
        AscendC::DataCopyPad(out, scratch,
            AscendC::DataCopyExtParams{1, kDispatchProfileWords * uint32_t(sizeof(uint64_t)), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
    }
};
} // namespace ascend_deepep::udma
