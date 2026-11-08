#pragma once

#include "kernels/dispatch/udma_dispatch_sender.h"
#include "common/types.h"
#include "simt_api/device_sync_functions.h"

namespace ascend_deepep::udma {
// Separate list windows from the sender's in-flight WQE staging region.
constexpr uint32_t kListsOffset = kDispatchSenderBytes;
static_assert(kListsOffset + kMaxPeers * kTileTokens * sizeof(int32_t) <= 216 * 1024);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)

// Independent peer cursors retain existing row ownership and completion.
template <uint32_t Capacity, class Sender, class Prefix>
__aicore__ inline void DispatchPeerCursors(Sender& sender, const DispatchTilingData& td, uint32_t core,
    const __gm__ int32_t* plan, Prefix& rank_gm, Prefix& channel_gm, uint32_t plan_stride, uint32_t fields)
{
    struct Cursor { uint32_t channel, pos, end, row, field; bool loaded; };
    Cursor cursor[Capacity];
    const uint32_t lane = core % td.udma_workers_per_peer;
    const bool weighted = td.udma_cross_lane0_weight != 1;
    const uint32_t step = weighted ? 1U : td.udma_workers_per_peer;
    for (uint32_t p = 0; p < sender.Count(); ++p)
        cursor[p] = {weighted ? 0U : lane, 0, 0, 0, 0, false};
    for (uint32_t first = 0; first < sender.Count();) {
        const uint32_t last = PeerWindowEnd(sender, td.rank, td.world_size, td.udma_ranks_per_group, td.udma_peer_order, first);
        bool active = true;
        while (active) {
            active = false;
            for (uint32_t p = first; p < last; ++p) {
                auto& c = cursor[p];
                const uint32_t peer = sender.Rank(p);
                while (c.channel < td.num_channels && (!c.loaded || c.pos == c.end)) {
                    if (c.loaded) c.channel += step;
                    if (c.channel >= td.num_channels) break;
                    const uint32_t previous = c.channel ? channel_gm.GetValue(peer * td.num_channels + c.channel - 1) : 0;
                    const uint32_t count = channel_gm.GetValue(peer * td.num_channels + c.channel) - previous;
                    c.pos = 0; c.end = count; c.field = 0; c.loaded = true;
                    if (weighted) {
                        const uint32_t total = channel_gm.GetValue(peer * td.num_channels + td.num_channels - 1);
                        const uint32_t first = PeerRowBoundary(total, lane, td.udma_workers_per_peer,
                            td.rank, peer, td.udma_ranks_per_group, td.udma_cross_lane0_weight);
                        const uint32_t last = PeerRowBoundary(total, lane + 1, td.udma_workers_per_peer,
                            td.rank, peer, td.udma_ranks_per_group, td.udma_cross_lane0_weight);
                        const uint32_t skip = first > previous ? first - previous : 0;
                        const uint32_t stop = last > previous ? last - previous : 0;
                        c.pos = skip < count ? skip : count; c.end = stop < count ? stop : count;
                    }
                    c.row = (td.rank ? rank_gm.GetValue((td.rank - 1) * td.world_size + peer) : 0) + previous + c.pos;
                }
                if (c.channel >= td.num_channels) continue;
                active = true;
                sender.BeginVisit(p);
                uint32_t visit = td.udma_wqes_per_visit;
                while (visit && c.pos < c.end) {
                    const uint64_t index = (uint64_t(peer) * td.num_channels + c.channel) * plan_stride + c.pos;
                    const uint32_t available = (c.end - c.pos) * fields - c.field;
                    const uint32_t n = sender.PutRows(p, plan + index, c.row, c.field,
                        available < visit ? available : visit, available);
                    const uint32_t rows = (c.field + n) / fields;
                    c.field = (c.field + n) % fields; c.pos += rows; c.row += rows; visit -= n;
                }
            }
        }
        // Submit staging before moving windows; final Publish/Quiet still owns completion.
        if (td.udma_peer_order == kGroupWindowOrder) sender.Flush();
        first = last;
    }
}

template <uint32_t Capacity, uint32_t Qps, class Resource>
__aicore__ inline void DispatchRemoteImpl(
    Resource& resource, GM_ADDR x, GM_ADDR scales, GM_ADDR plan, GM_ADDR rank_prefix, GM_ADDR channel_prefix,
    GM_ADDR workspace, const DispatchTilingData& td, uint32_t core, uint32_t cores, DispatchProfile& profile)
{
    constexpr uint32_t sender_bytes = kHeadersOffset + Capacity * Qps * (2U * kWqeBytes + kOutputRecordBytes) + 32;
    static_assert(sender_bytes + kDispatchCacheBytes + Capacity * kTileTokens * sizeof(int32_t) <= 216 * 1024);
    const uint32_t cache_bytes = td.dispatch_experiment_flags & kDispatchWqeCache ? kDispatchCacheBytes : 0U;
    auto wq = resource.template GetBufferByByte<uint8_t>(0);
    auto lists = resource.template GetBufferByByte<int32_t>(sender_bytes + cache_bytes);
    uint64_t stage = profile.Now();
    DispatchSenderT<Capacity, Qps> sender(td, core, wq, workspace, x, scales, UINT32_MAX, 0, &profile);
    profile.Add(profile.setup_ns, stage);
    stage = profile.Now();
    AscendC::GlobalTensor<int32_t> plan_gm;
    plan_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(plan));
    AscendC::GlobalTensor<int32_t> rank_gm, channel_gm;
    rank_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
    channel_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(channel_prefix));
    const uint32_t channel_tokens = (td.num_tokens + td.num_channels - 1) / td.num_channels;
    const uint32_t max_tile = 65536U / td.world_size < kTileTokens ? 65536U / td.world_size : kTileTokens;
    const uint32_t plan_stride = (channel_tokens + 7U) / 8U * 8U;
    const uint32_t scale_bytes = td.num_scales * td.scale_element_bytes;
    const uint32_t fields = td.has_x_scales && scale_bytes &&
                                   !(td.dispatch_experiment_flags & kDispatchSimtScales) ? 2U : 1U;
    const uint32_t lane = core % td.udma_workers_per_peer;
    const bool weighted = td.udma_cross_lane0_weight != 1;
    if ((td.dispatch_experiment_flags & kDispatchPeerCursor) || td.udma_peer_order == kGroupWindowOrder) {
        profile.Add(profile.plan_ns, stage); stage = profile.Now();
        DispatchPeerCursors<Capacity>(sender, td, core, reinterpret_cast<const __gm__ int32_t*>(plan),
            rank_gm, channel_gm, plan_stride, fields);
        profile.Add(profile.send_ns, stage); stage = profile.Now();
    } else {
    // Default retains channel ownership. Weighted lanes intersect one whole-peer
    // row partition with each channel, so rounding never resets at a tile boundary.
    for (uint32_t ch = weighted ? 0U : lane; sender.Count() && ch < td.num_channels;
         ch += weighted ? 1U : td.udma_workers_per_peer) {
        const uint32_t begin = ch * channel_tokens < td.num_tokens ? ch * channel_tokens : td.num_tokens;
        const uint32_t end = begin + channel_tokens < td.num_tokens ? begin + channel_tokens : td.num_tokens;
        uint32_t row[Capacity]{}, consumed[Capacity]{}, total[Capacity]{};
        for (uint32_t p = 0; p < sender.Count(); ++p) {
            const uint32_t peer = sender.Rank(p);
            const uint32_t previous = ch ? channel_gm.GetValue(peer * td.num_channels + ch - 1) : 0;
            total[p] = channel_gm.GetValue(peer * td.num_channels + ch) - previous;
            row[p] = (td.rank ? rank_gm.GetValue((td.rank - 1) * td.world_size + peer) : 0) +
                     (ch ? channel_gm.GetValue(peer * td.num_channels + ch - 1) : 0);
            if (weighted) {
                const uint32_t count = channel_gm.GetValue(peer * td.num_channels + td.num_channels - 1);
                const uint32_t first = PeerRowBoundary(count, lane, td.udma_workers_per_peer,
                    td.rank, peer, td.udma_ranks_per_group, td.udma_cross_lane0_weight);
                const uint32_t last = PeerRowBoundary(count, lane + 1U, td.udma_workers_per_peer,
                    td.rank, peer, td.udma_ranks_per_group, td.udma_cross_lane0_weight);
                const uint32_t skip = first > uint32_t(previous) ? first - previous : 0U;
                consumed[p] = skip < total[p] ? skip : total[p];
                const uint32_t stop = last > uint32_t(previous) ? last - previous : 0U;
                total[p] = stop < total[p] ? stop : total[p];
                row[p] += consumed[p];
            }
        }
        for (uint32_t start = begin; start < end; start += max_tile) {
            const uint32_t size = end - start < max_tile ? end - start : max_tile;
            uint32_t counts[Capacity]{}, cursor[Capacity]{}, field[Capacity]{}, remaining = 0;
            // Keep the original input-token tile and peer visit boundaries.
            // A sorted list can contribute at most `size` hits to this tile.
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
            for (uint32_t p = 0; p < sender.Count(); ++p) {
                const uint32_t left = total[p] - consumed[p];
                const uint32_t load = left < size ? left : size;
                if (load == 0)
                    continue;
                const uint64_t base = (uint64_t(sender.Rank(p)) * td.num_channels + ch) * plan_stride;
                AscendC::DataCopyExtParams copy{1, load * uint32_t(sizeof(int32_t)), 0, 0, 0};
                AscendC::DataCopyPad(
                    lists[p * kTileTokens], plan_gm[base + consumed[p]], copy,
                    AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
            for (uint32_t p = 0; p < sender.Count(); ++p) {
                const uint32_t left = total[p] - consumed[p];
                const uint32_t load = left < size ? left : size;
                while (counts[p] < load && uint32_t(lists.GetValue(p * kTileTokens + counts[p])) < start + size)
                    ++counts[p];
                remaining += counts[p];
                consumed[p] += counts[p];
            }
            profile.Add(profile.plan_ns, stage);
            stage = profile.Now();
            while (remaining) {
                for (uint32_t p = 0; p < sender.Count(); ++p) {
                    uint32_t visit = td.udma_wqes_per_visit;
                    if (cursor[p] < counts[p]) sender.BeginVisit(p);
                    while (visit && cursor[p] < counts[p]) {
                        const uint64_t base = (uint64_t(sender.Rank(p)) * td.num_channels + ch) * plan_stride;
                        const uint64_t index = base + consumed[p] - counts[p] + cursor[p];
                        const uint32_t available = (counts[p] - cursor[p]) * fields - field[p];
                        const uint32_t count = sender.PutRows(
                            p, reinterpret_cast<const __gm__ int32_t*>(plan) + index, row[p], field[p],
                            available < visit ? available : visit, available);
                        const uint32_t rows = (field[p] + count) / fields;
                        field[p] = (field[p] + count) % fields;
                        cursor[p] += rows;
                        row[p] += rows;
                        remaining -= rows;
                        visit -= count;
                    }
                }
            }
            profile.Add(profile.send_ns, stage);
            stage = profile.Now();
        }
    }
    }
    profile.Add(profile.plan_ns, stage);
    stage = profile.Now();
    sender.Publish(workspace, td.udma_completion_signal_offset, td.rank, core % td.udma_workers_per_peer);
    profile.Add(profile.publish_ns, stage);
    stage = profile.Now();
    IncomingObservation observation{profile.origin_cycle};
    if (profile.trace) {
        observation.first = profile.first_poll; observation.ready = profile.ready_poll;
        observation.polls = profile.poll_count;
    }
    WaitIncoming<true>(workspace, td.udma_completion_signal_offset, td.rank, td.world_size, core, cores,
        td.udma_workers_per_peer * Qps, profile.enabled ? &observation : nullptr);
    profile.remote_wait_ns = observation.remote_ns; profile.local_sync_ns = observation.local_ns;
    profile.Add(profile.incoming_ns, stage);
    stage = profile.Now();
    sender.Quiet();
    profile.Add(profile.quiet_ns, stage);
    stage = profile.Now();
    // SO markers cover payload only. Metadata AIVs can still be doing remote
    // SIMT stores on another rank; all ranks/cores must join before outputs are
    // read or the workspace is reused. The collective includes local sync.
    aclshmemi_sync_npu_v4<true>(0);
    profile.Add(profile.barrier_ns, stage);
    // All sender/metadata readers have drained; the initial WQE area can now
    // stage this AIV's private diagnostic record before output-copy reuse.
    profile.send_records = sender.Count() * Qps;
    profile.Store(td.udma_profile_output, core, resource.template GetBufferByByte<uint64_t>(0));
    sender.StoreTrace(td.udma_profile_output, core, resource.template GetBufferByByte<uint64_t>(0));
}
template <class Resource>
__aicore__ inline void DispatchRemote(
    Resource& r, GM_ADDR x, GM_ADDR scales, GM_ADDR plan, GM_ADDR rp, GM_ADDR cp,
    GM_ADDR workspace, const DispatchTilingData& td, uint32_t core, uint32_t cores, DispatchProfile& profile)
{
    if (td.udma_qps_per_worker == 2) {
        if (GroupOwnerOrder(td.udma_peer_order))
            DispatchRemoteImpl<kGroupPeerCapacity, 2>(r, x, scales, plan, rp, cp, workspace, td, core, cores, profile);
        else
            DispatchRemoteImpl<kMaxPeers, 2>(r, x, scales, plan, rp, cp, workspace, td, core, cores, profile);
    } else if (GroupOwnerOrder(td.udma_peer_order))
        DispatchRemoteImpl<kGroupPeerCapacity, 1>(r, x, scales, plan, rp, cp, workspace, td, core, cores, profile);
    else
        DispatchRemoteImpl<kMaxPeers, 1>(r, x, scales, plan, rp, cp, workspace, td, core, cores, profile);
}
#endif
} // namespace ascend_deepep::udma
