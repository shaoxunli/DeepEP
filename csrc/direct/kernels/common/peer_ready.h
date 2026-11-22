#pragma once

#include "common/types.h"
#include "kernel_operator.h"
#include "shmem.h"

namespace ascend_deepep::udma {
// Buffer-owned comm_meta: three cache-line-isolated tables. Table 2 retains
// 64 per-QP completion words per source. No payload or handle aliases this area.
inline constexpr uint64_t kPeerSlotBytes = 512;
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__ inline __gm__ uint64_t* PeerSlot(uint64_t control, uint32_t world, uint32_t table, uint32_t peer)
{
    return reinterpret_cast<__gm__ uint64_t*>(control + (uint64_t(table) * world + peer) * kPeerSlotBytes);
}
__aicore__ inline uint64_t PeerLoad(__gm__ uint64_t* address)
{
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(address));
    AscendC::GlobalTensor<uint64_t> word;
    word.SetGlobalBuffer(address);
    return word.GetValue(0);
}
template <class T>
__aicore__ inline bool PeerReady(const T& td, uint32_t peer, uint64_t started)
{
    if (!td.peer_epoch || peer == td.rank) return true;
    const uint64_t value = PeerLoad(PeerSlot(td.peer_control, td.world_size, 0, peer));
    if (value >= td.peer_epoch) return true;
    if (uint64_t(AscendC::GetSystemCycle()) - started >= td.npu_timeout_cycles) {
        AscendC::printf("[PEER_READY_TIMEOUT] rank=%u peer=%u epoch=%llu observed=%llu\n", td.rank, peer,
                        (unsigned long long)td.peer_epoch, (unsigned long long)value);
        trap();
    }
    return false;
}
template <class T>
__aicore__ inline void WaitPeerReady(const T& td, uint32_t peer)
{
    const uint64_t started = AscendC::GetSystemCycle();
    while (!PeerReady(td, peer, started)) {
    }
}
// Each AIV owns distinct destinations. No control SQ is shared with payload.
// MTE3 completes its UB read before the scalar value can be reused.
__aicore__ inline void PushPeerEpoch(uint64_t control, uint32_t world, uint32_t table, uint32_t rank, uint32_t core,
                                     uint32_t cores, uint64_t epoch, AscendC::LocalTensor<uint64_t> scratch)
{
    scratch.SetValue(0, epoch);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    for (uint32_t peer = core; peer < world; peer += cores) {
        AscendC::GlobalTensor<uint64_t> destination;
        destination.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint64_t*>(aclshmem_ptr(PeerSlot(control, world, table, rank), peer)));
        AscendC::DataCopyPad(destination, scratch, AscendC::DataCopyExtParams{1, 8, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID4);
    }
}
template <class T>
__aicore__ inline void PublishMetadataDone(const T& td, uint32_t core, uint32_t cores,
                                           AscendC::LocalTensor<uint64_t> scratch)
{
    if (!td.peer_epoch) return;
    // All SIMT VFs have V_S drains at their call sites. Drain memory writes on
    // every issuing AIV before a local join delegates notification to any AIV.
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    AscendC::SyncAll<true>();
    PushPeerEpoch(td.peer_control, td.world_size, 1, td.rank, core, cores, td.peer_epoch, scratch);
}
template <class T>
__aicore__ inline void WaitMetadataDone(const T& td, uint32_t core, uint32_t cores)
{
    const uint64_t started = AscendC::GetSystemCycle();
    for (uint32_t source = core; source < td.world_size; source += cores) {
        if (source == td.rank) continue;
        auto* address = PeerSlot(td.peer_control, td.world_size, 1, source);
        while (PeerLoad(address) < td.peer_epoch) {
            if (uint64_t(AscendC::GetSystemCycle()) - started >= td.npu_timeout_cycles) {
                AscendC::printf("[PEER_META_TIMEOUT] rank=%u source=%u epoch=%llu\n", td.rank, source,
                                (unsigned long long)td.peer_epoch);
                trap();
            }
        }
    }
    AscendC::SyncAll<true>();
}
#endif
}  // namespace ascend_deepep::udma
