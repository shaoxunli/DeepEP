#pragma once

#include "ascend_deepep/kernels/udma_combine_sender.h"
#include "ascend_deepep/kernels/types.h"

namespace ascend_deepep::udma {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
template <class Resource>
__aicore__ inline void CombineRemote(
    Resource& resource, GM_ADDR x, GM_ADDR weights, GM_ADDR src_idx, GM_ADDR rank_prefix, GM_ADDR channel_prefix,
    GM_ADDR workspace, const CombineTilingData& td, uint32_t core)
{
    auto wq = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
    auto indices = resource.ubBuf.template GetBufferByByte<int32_t>(kCombineSenderBytes);
    CombineSender sender(td, core, reinterpret_cast<__ubuf__ uint8_t*>(wq.GetPhyAddr()), workspace, x, weights);
    AscendC::GlobalTensor<int32_t> src_gm, rank_gm;
    src_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(src_idx));
    rank_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rank_prefix));
    // Dispatch stores all channels of one source rank contiguously. Ownership
    // here is per peer, so channel boundaries need not split the return stream.
    (void)channel_prefix;
    const uint32_t fields = td.has_topk_weights ? 2U : 1U;
    uint32_t row[kMaxPeers]{}, end[kMaxPeers]{}, loaded[kMaxPeers]{}, cursor[kMaxPeers]{}, field[kMaxPeers]{};
    uint32_t remaining = 0;
    for (uint32_t p = 0; p < sender.Count(); ++p) {
        const uint32_t peer = sender.Rank(p);
        row[p] = peer ? rank_gm.GetValue((peer - 1) * td.world_size + td.rank) : 0;
        end[p] = rank_gm.GetValue(peer * td.world_size + td.rank);
        remaining += end[p] - row[p];
    }
    while (remaining) {
        for (uint32_t p = 0; p < sender.Count(); ++p) {
            uint32_t visit = td.udma_wqes_per_visit;
            while (visit && row[p] < end[p]) {
                if (cursor[p] == loaded[p]) {
                    // Pending spans can still read this window in SIMT.
                    sender.Flush();
                    const uint32_t aligned = row[p] / 8 * 8;
                    const uint32_t leading = row[p] - aligned;
                    const uint32_t count =
                        end[p] - row[p] < kTileTokens - leading ? end[p] - row[p] : kTileTokens - leading;
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
                    AscendC::DataCopyExtParams copy{1, (count + leading) * uint32_t(sizeof(int32_t)), 0, 0, 0};
                    AscendC::DataCopyPad(
                        indices[p * kTileTokens], src_gm[aligned], copy,
                        AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                    cursor[p] = leading;
                    loaded[p] = leading + count;
                }
                const uint32_t available = (loaded[p] - cursor[p]) * fields - field[p];
                auto* window = reinterpret_cast<__ubuf__ int32_t*>(indices.GetPhyAddr()) + p * kTileTokens + cursor[p];
                const uint32_t count =
                    sender.PutRows(p, window, row[p], field[p], available < visit ? available : visit);
                const uint32_t rows = (field[p] + count) / fields;
                field[p] = (field[p] + count) % fields;
                cursor[p] += rows;
                row[p] += rows;
                remaining -= rows;
                visit -= count;
            }
        }
    }
    sender.Publish(workspace, td.udma_completion_signal_offset, td.rank);
    WaitIncoming<false>(workspace, td.udma_completion_signal_offset, td.rank, td.world_size, core, td.num_channels);
    sender.Quiet();
    AscendC::SyncAll<true>();
}
#endif
} // namespace ascend_deepep::udma
