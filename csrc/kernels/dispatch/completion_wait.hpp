// Copyright (c) 2026, Lu Lu

#pragma once

#include "aggregate_packet.hpp"

template <typename Scratch>
__aicore__ inline void WaitDispatchIncoming(GM_ADDR workspace, const DispatchTiling& t, Scratch& scratch)
{
    // AIV7 finishes its local-copy/relay-assist role before becoming the sole
    // incoming poller. Kernel completion already joins the other AIVs; no
    // additional SyncAll is required and no caller collective is hidden here.
    if (GetBlockIdx() != 7U) return;
    const uint32_t local_begin = t.rank / 8U * 8U;
    const uint32_t local_end = local_begin + 8U < t.world ? local_begin + 8U : t.world;
    // Clos QP0/1 cover sources outside this server; FM covers the other local
    // ranks. There are no self or unused-lane notifications to wait for.
    const uint32_t starts[6] = {
        0U, local_end, t.world, t.world + local_end, 2U * t.world + local_begin, 2U * t.world + t.rank + 1U};
    // Benchmark mode skips FM payload, so those senders publish no terminal flag.
    const uint32_t lengths[6] = {local_begin,
                                 t.world - local_end,
                                 local_begin,
                                 t.world - local_end,
                                 t.cross_server_only ? 0U : t.rank - local_begin,
                                 t.cross_server_only ? 0U : local_end - t.rank - 1U};
    const uint32_t expected =
        2U * (t.world - (local_end - local_begin)) + (t.cross_server_only ? 0U : local_end - local_begin - 1U);
    const uint64_t begin = GetSystemCycle();
    // At most 36 batches for supported EP <= 512. A sender cannot overwrite a
    // completed slot until this rank publishes its next descriptor generation.
    uint64_t completed = 0U;
    uint32_t ready = 0U;
    for (;;) {
        uint32_t batch = 0U;
        for (uint32_t range = 0U; range < 6U; ++range) {
            for (uint32_t offset = 0U; offset < lengths[range]; offset += DispatchAggregatePacket::kPollBlocks) {
                const uint32_t left = lengths[range] - offset;
                const uint32_t count =
                    left < DispatchAggregatePacket::kPollBlocks ? left : DispatchAggregatePacket::kPollBlocks;
                const uint64_t bit = uint64_t(1) << batch++;
                if (completed & bit) continue;
                const uint32_t matches = DispatchAggregatePacket::ReadyCount(
                    workspace + t.done + uint64_t(starts[range] + offset) * 512U, count, t.generation, scratch);
                if (matches == count) {
                    completed |= bit;
                    ready += count;
                }
            }
        }
        if (ready == expected) break;
        if (GetSystemCycle() - begin >= kDispatchCompletionTimeoutCycles) {
            aclshmemi_kernel_abort("Dispatch incoming timeout rank=%u expected=%u ready=%u generation=%llu\n", t.rank,
                                   expected, ready, static_cast<unsigned long long>(t.generation));
            trap();
            return;
        }
        PauseDispatchCompletionPoll();
    }
    DataSyncBarrier<MemDsbT::DDR>();
}
