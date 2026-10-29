#pragma once

#include <cstdint>

#ifdef ASCEND_DEEPEP_DEVICE_SCHEDULE
#define ASCEND_DEEPEP_SCHEDULE_FN __aicore__ inline
#else
#define ASCEND_DEEPEP_SCHEDULE_FN inline constexpr
#endif

namespace ascend_deepep::udma {
inline constexpr uint32_t kQpCount = 32;
inline constexpr uint32_t kMaxPeers = 8;
inline constexpr uint32_t kTileTokens = 256;
inline constexpr uint32_t kBatchWqes = 32;
inline constexpr uint32_t kWqeBytes = 64;
inline constexpr uint32_t kSignalBytes = 512;

ASCEND_DEEPEP_SCHEDULE_FN bool ValidPeers(uint32_t peers)
{
    return peers == 1 || peers == 2 || peers == 4 || peers == 8;
}
ASCEND_DEEPEP_SCHEDULE_FN bool ValidVisit(uint32_t wqes)
{
    return wqes == 4 || wqes == 8 || wqes == 16 || wqes == 32;
}
ASCEND_DEEPEP_SCHEDULE_FN uint32_t Workers(uint32_t world, uint32_t peers)
{
    return (world - 1 + peers - 1) / peers;
}
// Token mode partitions channels/return rows across half the AIVs. The other
// half retains SIMT metadata/self-copy parallelism. QP ownership never depends
// on whether a worker's assigned token range is empty.
ASCEND_DEEPEP_SCHEDULE_FN uint32_t TokenWorkers(uint32_t cores)
{
    return cores / 2U < kQpCount ? cores / 2U : kQpCount;
}
ASCEND_DEEPEP_SCHEDULE_FN uint64_t TokenPeerMask(uint32_t rank, uint32_t peer)
{
    return rank == peer ? 0 : (uint64_t{1} << kQpCount) - 1U;
}
// Spare AIVs own all metadata channels; if every AIV owns a QP, each core
// handles its metadata before sending payload, without changing QP ownership.
ASCEND_DEEPEP_SCHEDULE_FN uint32_t MetadataFirst(uint32_t world, uint32_t peers, uint32_t cores)
{
    const uint32_t payload = Workers(world, peers);
    return payload < cores ? payload : 0U;
}
ASCEND_DEEPEP_SCHEDULE_FN uint32_t Peer(uint32_t rank, uint32_t world, uint32_t index)
{
    return (rank + 1 + index) % world;
}
ASCEND_DEEPEP_SCHEDULE_FN uint32_t Owner(uint32_t rank, uint32_t peer, uint32_t world, uint32_t peers)
{
    return ((peer + world - rank - 1) % world) / peers;
}
// Both endpoints must create the same channel/QP set. Include every supported
// schedule, so switching visit/group size after a drained call needs no reinit.
ASCEND_DEEPEP_SCHEDULE_FN uint64_t PeerMask(uint32_t rank, uint32_t peer, uint32_t world)
{
    if (rank == peer)
        return 0;
    uint64_t mask = 1; // SDK control QP.
    for (uint32_t peers = 1; peers <= kMaxPeers; peers *= 2) {
        if (Workers(world, peers) <= kQpCount) {
            mask |= uint64_t{1} << Owner(rank, peer, world, peers);
            mask |= uint64_t{1} << Owner(peer, rank, world, peers);
        }
    }
    return mask;
}
} // namespace ascend_deepep::udma

#undef ASCEND_DEEPEP_SCHEDULE_FN
