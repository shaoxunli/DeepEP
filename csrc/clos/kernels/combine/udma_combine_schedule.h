#pragma once

#include "kernels/combine/udma_combine_sender.h"
#include "common/types.h"
#include "kernels/aggregate/aggregate_pipeline.h"
#include "kernels/combine/udma_combine_weights.h"

namespace ascend_deepep::udma {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
template <uint32_t Capacity, uint32_t Phase, uint32_t Qps, class Resource>
__aicore__ inline void CombineRemoteImpl(
    Resource& resource, GM_ADDR x, GM_ADDR weights, GM_ADDR src_idx, GM_ADDR rank_prefix, GM_ADDR channel_prefix,
    GM_ADDR workspace, const CombineTilingData& td, uint32_t core, DispatchProfile* diagnostic = nullptr)
{
    DispatchProfile inactive{false};
    auto& profile = diagnostic ? *diagnostic : inactive;
    uint64_t stage = profile.Now();
    constexpr uint32_t sender_bytes = kHeadersOffset + Capacity * Qps * (2U * kWqeBytes + sizeof(uint64_t));
    static_assert(sender_bytes <= 216 * 1024);
    auto wq = resource.template GetBufferByByte<uint8_t>(0);
    const bool simt_weights = Phase == 0 && CombineSimtWeightsEnabled(td);
    if (simt_weights)
        CombineRemoteWeights(weights, src_idx, rank_prefix, workspace, td, core);
    // Dispatch owns src_idx and completes before combine. Combine only reads it;
    // the handle keeps its GM storage alive through the final send completion.
    const auto* indices = reinterpret_cast<const __gm__ int32_t*>(src_idx);
    CombineSenderT<Capacity, Qps, const __gm__ int32_t*> sender(td, core, reinterpret_cast<__ubuf__ uint8_t*>(wq.GetPhyAddr()), workspace,
        x, weights, UINT32_MAX, 0, 1, 0, !simt_weights, &profile);
    profile.Add(profile.setup_ns, stage); stage = profile.Now();
    AscendC::GlobalTensor<int32_t> rank_gm;
    rank_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
    // Dispatch stores all channels of one source rank contiguously. Ownership
    // here is per peer, so channel boundaries need not split the return stream.
    (void)channel_prefix;
    const uint32_t fields = td.has_topk_weights && !simt_weights ? 2U : 1U;
    uint32_t row[Capacity]{}, end[Capacity]{}, field[Capacity]{};
    uint32_t ready_chunk[Capacity];
    if constexpr (Phase == 3) {
        for (uint32_t p = 0; p < Capacity; ++p) ready_chunk[p] = UINT32_MAX;
    }
    for (uint32_t p = 0; p < sender.Count(); ++p) {
        const uint32_t peer = sender.Rank(p);
        row[p] = peer ? rank_gm.GetValue((peer - 1) * td.world_size + td.rank) : 0;
        end[p] = rank_gm.GetValue(peer * td.world_size + td.rank);
        // Split the original peer row interval, retaining src_idx and output addresses.
        const uint32_t begin = row[p], length = end[p] - begin;
        const uint32_t lane = core % td.udma_workers_per_peer;
        row[p] = begin + PeerRowBoundary(length, lane, td.udma_workers_per_peer,
            td.rank, peer, td.udma_ranks_per_group, td.udma_cross_lane0_weight);
        end[p] = begin + PeerRowBoundary(length, lane + 1U, td.udma_workers_per_peer,
            td.rank, peer, td.udma_ranks_per_group, td.udma_cross_lane0_weight);
    }
    profile.Add(profile.plan_ns, stage); stage = profile.Now();
    for (uint32_t first = 0; first < sender.Count();) {
        const uint32_t last = PeerWindowEnd(sender, td.rank, td.world_size, td.udma_ranks_per_group, td.udma_peer_order, first);
        uint32_t remaining = 0;
        for (uint32_t p = first; p < last; ++p) remaining += end[p] - row[p];
        while (remaining) {
            for (uint32_t p = first; p < last; ++p) {
                uint32_t visit = td.udma_wqes_per_visit;
                if (row[p] < end[p]) sender.BeginVisit(p);
                while (visit && row[p] < end[p]) {
                    if constexpr (Phase == 3) {
                        const uint32_t chunk = row[p] / td.pipeline_chunk_tokens;
                        if (ready_chunk[p] != chunk) {
                            sender.Flush(); // Publish earlier rows before waiting; never quiet per chunk.
                            WaitAggregateReady(td, core, row[p]);
                            ready_chunk[p] = chunk;
                        }
                    }
                    uint32_t available = (end[p] - row[p]) * fields - field[p];
                    if constexpr (Phase == 3) {
                        const uint32_t in_chunk = td.pipeline_chunk_tokens - row[p] % td.pipeline_chunk_tokens;
                        const uint32_t ready_fields = in_chunk * fields - field[p];
                        available = available < ready_fields ? available : ready_fields;
                    }
                    const auto* window = indices + row[p];
                    const uint32_t count =
                        sender.template PutSelectedRows<Phase == 3 ? 0 : Phase>(
                            p, window, row[p], field[p], available < visit ? available : visit);
                    const uint32_t rows = (field[p] + count) / fields;
                    field[p] = (field[p] + count) % fields;
                    row[p] += rows;
                    remaining -= rows;
                    visit -= count;
                }
            }
        }
        if (td.udma_peer_order == kGroupWindowOrder) sender.Flush();
        first = last;
    }
    profile.Add(profile.send_ns, stage); stage = profile.Now();
    if constexpr (Phase == 1) {
        sender.Flush(); // Drain WQE staging, not the asynchronous payload reads.
    } else {
        sender.Publish(workspace, td.udma_completion_signal_offset, td.rank, core % td.udma_workers_per_peer);
        profile.Add(profile.publish_ns, stage); stage = profile.Now();
        IncomingObservation observation{profile.origin_cycle};
        WaitIncoming<false>(workspace, td.udma_completion_signal_offset, td.rank, td.world_size, core,
            td.num_channels, td.udma_workers_per_peer * Qps, profile.enabled ? &observation : nullptr);
        profile.remote_wait_ns = observation.remote_ns; profile.local_sync_ns = observation.local_ns;
        profile.Add(profile.incoming_ns, stage); stage = profile.Now();
        sender.Quiet();
        profile.Add(profile.quiet_ns, stage); stage = profile.Now();
        // UDMA SO markers do not cover SIMT stores issued by other AIVs.
        // Match Dispatch's full-rank completion before Reduce/workspace reuse.
        if (simt_weights)
            aclshmemi_sync_npu_v4<true>(0);
        else
            AscendC::SyncAll<true>();
        profile.Add(profile.barrier_ns, stage);
        profile.send_records = sender.Count() * Qps;
    }
}
template <uint32_t Phase = 0, class Resource>
__aicore__ inline void CombineRemote(
    Resource& r, GM_ADDR x, GM_ADDR weights, GM_ADDR idx, GM_ADDR rp, GM_ADDR cp,
    GM_ADDR workspace, const CombineTilingData& td, uint32_t core, DispatchProfile* diagnostic = nullptr)
{
    if (td.udma_qps_per_worker == 2) {
        if (GroupOwnerOrder(td.udma_peer_order))
            CombineRemoteImpl<kGroupPeerCapacity, Phase, 2>(r, x, weights, idx, rp, cp, workspace, td, core, diagnostic);
        else
            CombineRemoteImpl<kMaxPeers, Phase, 2>(r, x, weights, idx, rp, cp, workspace, td, core, diagnostic);
    } else if (GroupOwnerOrder(td.udma_peer_order))
        CombineRemoteImpl<kGroupPeerCapacity, Phase, 1>(r, x, weights, idx, rp, cp, workspace, td, core, diagnostic);
    else
        CombineRemoteImpl<kMaxPeers, Phase, 1>(r, x, weights, idx, rp, cp, workspace, td, core, diagnostic);
}
#endif
} // namespace ascend_deepep::udma
