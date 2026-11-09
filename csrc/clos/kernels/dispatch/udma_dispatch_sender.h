#pragma once

#include "kernels/common/udma_shared_sender.h"
#include "common/types.h"
#include "kernels/dispatch/udma_dispatch_output.h"
#include "kernels/dispatch/udma_dispatch_profile.h"
#include "utils/debug/asc_assert.h"

namespace ascend_deepep::udma {
inline constexpr uint32_t kDispatchBasesOffset = kSenderScratchBytes;
inline constexpr uint32_t kDispatchEpochOffset = kDispatchBasesOffset + kMaxPeers * kOutputRecordBytes;
inline constexpr uint32_t kDispatchSenderBytes = kDispatchEpochOffset + 32;
static_assert(kDispatchSenderBytes % 32 == 0);
// Eight direct-mapped entries per AIV; collisions flush pending readers before
// replacement. Each entry holds 32 complete, unsignaled WQEs (16 KiB total).
inline constexpr uint32_t kDispatchCacheEntries = 8;
inline constexpr uint32_t kDispatchCacheBytes = kDispatchCacheEntries * kBatchWqes * kWqeBytes;

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// Spans reference the immutable GM plan, so a list-window refill never forces
// an extra short SQ batch or leaves SIMT reading overwritten UB indices.
__simt_vf__ __launch_bounds__(32) inline void BuildDispatchWqes(
    __ubuf__ uint64_t* wqes, __ubuf__ uint64_t* spans, __ubuf__ uint64_t* headers, __ubuf__ uint64_t* remote_bases,
    uint64_t x, uint64_t scales, uint32_t x_bytes, uint32_t scale_bytes,
    uint32_t tokens, uint32_t fields, uint32_t span_count, uint32_t count)
{
    const uint32_t i = threadIdx.x;
    if (i >= count)
        return;
    for (uint32_t s = 0; s < span_count; ++s) {
        auto* span = spans + s * 4U;
        const uint32_t begin = static_cast<uint32_t>(span[3]);
        const uint32_t length = static_cast<uint32_t>(span[1] >> 32U);
        if (i < begin || i - begin >= length)
            continue;
        const uint32_t peer = static_cast<uint32_t>(span[2]);
        const uint32_t position = static_cast<uint32_t>(span[2] >> 32U) + i - begin;
        const uint64_t row = uint64_t(static_cast<uint32_t>(span[1])) + position / fields;
        const int32_t token = reinterpret_cast<const __gm__ int32_t*>(span[0])[position / fields];
        if (token < 0 || static_cast<uint32_t>(token) >= tokens) {
            __asc_simt_vf::__trap();
            return;
        }
        const bool scale = position % fields != 0;
        const uint32_t bytes = scale ? scale_bytes : x_bytes;
        auto* output = remote_bases + peer * kOutputRecordWords;
        if (uint64_t(row) >= output[2]) {
            __asc_simt_vf::__trap();
            return;
        }
        auto* header = headers + (peer * 2U + uint32_t(i + 1U == count)) * 8U;
        auto* wqe = wqes + i * 8U;
        wqe[0] = header[0];
        wqe[1] = header[1];
        wqe[2] = header[2];
        wqe[3] = header[3];
        wqe[4] = header[4];
        wqe[5] = output[scale ? 1 : 0] + uint64_t(row) * bytes;
        wqe[6] = bytes;
        wqe[7] = (scale ? scales : x) + uint64_t(token) * bytes;
        return;
    }
    __asc_simt_vf::__trap();
}

__simt_vf__ __launch_bounds__(32) inline void GatherDispatchCachedWqes(
    __ubuf__ uint64_t* wqes, __ubuf__ uint64_t* spans, uint32_t span_count, uint32_t count)
{
    const uint32_t i = threadIdx.x;
    if (i >= count) return;
    for (uint32_t s = 0; s < span_count; ++s) {
        auto* span = spans + s * 4U;
        const uint32_t begin = static_cast<uint32_t>(span[3]);
        const uint32_t length = static_cast<uint32_t>(span[1]);
        if (i < begin || i - begin >= length) continue;
        auto* src = reinterpret_cast<__ubuf__ uint64_t*>(span[0]) + (i - begin) * 8U;
        for (uint32_t word = 0; word < 8; ++word) wqes[i * 8U + word] = src[word];
        return;
    }
    __asc_simt_vf::__trap();
}

template <uint32_t Capacity = kMaxPeers, uint32_t Qps = 1>
class DispatchSenderT : public SenderT<Capacity, Qps> {
    using Base = SenderT<Capacity, Qps>;
    using Base::scratch_; using Base::count_; using Base::peers_; using Base::qp_;
    using Base::active_state_; using Base::last_; using Base::qp_base_;
    static constexpr uint32_t kDispatchBasesOffset = kHeadersOffset + Capacity * Qps * 2U * kWqeBytes;
    static constexpr uint32_t kDispatchEpochOffset = kDispatchBasesOffset + Capacity * Qps * kOutputRecordBytes;
    static constexpr uint32_t kCacheOffset = kDispatchEpochOffset + 32;
public:
    __aicore__ DispatchSenderT(
        const DispatchTilingData& td, uint32_t core, AscendC::LocalTensor<uint8_t> scratch, GM_ADDR workspace, GM_ADDR x,
        GM_ADDR scales, uint32_t peer_start = UINT32_MAX, uint32_t rotation = 0, DispatchProfile* profile = nullptr)
        : Base(td.rank, td.world_size, core,
              peer_start == UINT32_MAX ? td.udma_peers_per_worker : kMaxPeers,
              reinterpret_cast<__ubuf__ uint8_t*>(scratch.GetPhyAddr()), workspace, td.udma_completion_signal_offset, peer_start, rotation, td.udma_peer_order, td.udma_ranks_per_group,
              peer_start == UINT32_MAX ? td.udma_workers_per_peer : 1U),
          td_(td),
          x_(reinterpret_cast<uint64_t>(x)),
          scales_(reinterpret_cast<uint64_t>(scales)), profile_(profile),
          cache_enabled_((td.dispatch_experiment_flags & kDispatchWqeCache) != 0)
    {
        if (Tracing()) for (uint32_t lane = 0; lane < Qps; ++lane)
            for (uint32_t i = 0; i < count_; ++i) {
                const uint32_t p = lane * Capacity + i;
                for (uint32_t w = 0; w < kTraceSendWords; ++w) trace_[p][w] = 0;
                trace_[p][0] = peers_[p].peer; trace_[p][1] = qp_base_ + lane;
                uint32_t alias = 0;
                while (alias < i && peers_[lane * Capacity + alias].state != peers_[p].state) ++alias;
                trace_[p][11] = lane * Capacity + alias;
            }
        auto* bases = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kDispatchBasesOffset);
        if (!td.udma_output_control) {
            // Internal staging baseline. Native UDMA calls require the
            // mailbox; no public option can skip publication accidentally.
            for (uint32_t lane = 0; lane < Qps; ++lane)
                for (uint32_t i = 0; i < count_; ++i) {
                    const uint32_t p = lane * Capacity + i;
                    bases[p * kOutputRecordWords] = peers_[p].remote_base + td.x_buffer_offset;
                    bases[p * kOutputRecordWords + 1] = peers_[p].remote_base + td.x_scales_buffer_offset;
                    bases[p * kOutputRecordWords + 2] = UINT32_MAX;
                }
            return;
        }
        if (count_ == 0)
            return;
        AscendC::GlobalTensor<uint8_t> control;
        control.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(td.udma_output_control));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID6);
        AscendC::DataCopyPad(scratch[kDispatchEpochOffset], control,
            AscendC::DataCopyExtParams{1, 32, 0, 0, 0},
            AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        for (uint32_t lane = 0; lane < Qps; ++lane)
            for (uint32_t i = 0; i < count_; ++i) {
                const uint32_t p = lane * Capacity + i;
                AscendC::DataCopyPad(scratch[kDispatchBasesOffset + p * kOutputRecordBytes],
                    control[OutputRecordOffset(peers_[p].peer)],
                    AscendC::DataCopyExtParams{1, kOutputRecordBytes, 0, 0, 0},
                    AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
            }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID6);
        const uint64_t epoch = *reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kDispatchEpochOffset);
        const uint64_t x_bytes = uint64_t(td.hidden) * td.x_element_bytes;
        const uint64_t scale_bytes = td.has_x_scales ? uint64_t(td.num_scales) * td.scale_element_bytes : 0;
        for (uint32_t lane = 0; lane < Qps; ++lane)
            for (uint32_t i = 0; i < count_; ++i) {
                const uint32_t p = lane * Capacity + i;
                auto* out = bases + p * kOutputRecordWords;
                if (!epoch || out[kOutputEpochWord] != epoch || out[2] > UINT32_MAX ||
                    out[3] != x_bytes || out[4] != scale_bytes || (out[2] && !out[0]) ||
                    (out[2] && scale_bytes && !out[1]) ||
                    out[0] > UINT64_MAX - out[2] * x_bytes || out[1] > UINT64_MAX - out[2] * scale_bytes) {
                    AscendC::printf("dispatch output descriptor mismatch rank=%u peer=%u epoch=%llu remote=%llu\n",
                        td.rank, peers_[p].peer, static_cast<unsigned long long>(epoch),
                        static_cast<unsigned long long>(out[kOutputEpochWord]));
                    trap();
                }
            }
    }

    __aicore__ uint32_t PutRows(uint32_t p, const __gm__ int32_t* indices, uint32_t row, uint32_t field,
        uint32_t count, uint32_t available = 0)
    {
        p = Base::QueuePeer(p);
        if (cache_enabled_) return PutCachedRows(p, indices, row, field, count, available);
        if (rows_pending_ && active_state_ != peers_[p].state)
            Flush();
        if (!rows_pending_) {
            active_state_ = peers_[p].state;
            Credit(p);
        }
        const uint32_t n = count < kBatchWqes - rows_pending_ ? count : kBatchWqes - rows_pending_;
        auto* span = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset) + spans_ * 4U;
        span[0] = reinterpret_cast<uint64_t>(indices);
        span[1] = uint64_t(row) | (uint64_t(n) << 32U);
        span[2] = uint64_t(p) | (uint64_t(field) << 32U);
        span[3] = rows_pending_;
        ++spans_;
        rows_pending_ += n;
        TraceRows(p, n);
        last_ = p;
        if (rows_pending_ == kBatchWqes)
            Flush();
        return n;
    }

    __aicore__ void Flush()
    {
        if (!rows_pending_)
            return;
        const uint64_t started = Now();
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
        if (cache_enabled_) {
            asc_vf_call<GatherDispatchCachedWqes>(dim3(32), reinterpret_cast<__ubuf__ uint64_t*>(scratch_),
                reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset), spans_, rows_pending_);
        } else asc_vf_call<BuildDispatchWqes>(
            dim3(32), reinterpret_cast<__ubuf__ uint64_t*>(scratch_),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kHeadersOffset),
            reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kDispatchBasesOffset), x_, scales_,
            td_.hidden * td_.x_element_bytes, td_.num_scales * td_.scale_element_bytes,
            td_.num_tokens, td_.has_x_scales && td_.num_scales && td_.scale_element_bytes &&
                                    !(td_.dispatch_experiment_flags & (kDispatchSimtScales | kDispatchPackedScales)) ? 2U : 1U, spans_,
            rows_pending_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        if (profile_ && profile_->enabled && profile_->detailed) {
            // V_MTE3 orders the copy engine, not the scalar timer. Diagnostic
            // runs drain SIMT here so its execution is not charged to submit.
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
            profile_->Add(profile_->build_ns, started);
        }
        const uint64_t submitted = Now();
        Base::SubmitPayload(rows_pending_);
        if (Tracing()) {
            const uint64_t at = profile_->Elapsed();
            for (uint32_t lane = 0; lane < Qps; ++lane) for (uint32_t i = 0; i < count_; ++i) {
                const uint32_t p = lane * Capacity + i;
                if (!(trace_pending_[p / 32] & (1U << (p % 32)))) continue;
                if (!trace_[p][7]) trace_[p][2] = at;
                trace_[p][3] = at; ++trace_[p][7];
            }
            for (auto& bits : trace_pending_) bits = 0;
        }
        if (profile_ && profile_->enabled && profile_->detailed) {
            profile_->Add(profile_->submit_ns, submitted);
            profile_->wqes += rows_pending_;
            ++profile_->batches;
        }
        rows_pending_ = 0;
        spans_ = 0;
        cache_readers_ = 0;
    }

    // One visit is at most udma_wqes_per_visit WQEs for one peer.
    // Baseline Qps=1 keeps its original batching, with no extra flush.
    __aicore__ void BeginVisit(uint32_t p)
    {
        if constexpr (Qps == 2) {
            Flush();
            Base::SelectQp(next_qp_[p]);
            next_qp_[p] ^= 1U;
        }
    }

    // Compact scale intervals use the same SQ and terminal SO as payload.
    // Flush both request builders before their common UB scratch changes owner.
    __aicore__ void PutPackedScales(uint32_t p, uint32_t first, uint32_t count, uint32_t rank_base)
    {
        if (!count) return;
        Flush();
        p = Base::QueuePeer(p);
        const uint64_t bytes = uint64_t(td_.num_scales) * td_.scale_element_bytes;
        const uint64_t row = uint64_t(rank_base) + first;
        auto* output = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kDispatchBasesOffset) + p * kOutputRecordWords;
        if (uint64_t(first) + count > td_.num_tokens || row + count > output[2] ||
            uint64_t(count) * bytes > UINT32_MAX) { trap(); return; }
        active_state_ = peers_[p].state;
        Credit(p);
        auto* request = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset);
        request[0] = output[1] + row * bytes;
        request[1] = td_.packed_peer_scales + (uint64_t(peers_[p].peer) * td_.num_tokens + first) * bytes;
        request[2] = uint64_t(count) * bytes;
        request[3] = p;
        last_ = p;
        Base::pending_ = 1;
        const uint64_t submitted = Now();
        Base::Flush();
        if (Tracing()) {
            const uint64_t at = profile_->Elapsed();
            if (!trace_[p][7]) trace_[p][2] = at;
            trace_[p][3] = at; ++trace_[p][6]; ++trace_[p][7];
        }
        if (profile_ && profile_->enabled && profile_->detailed) {
            profile_->Add(profile_->scale_submit_ns, submitted);
            ++profile_->wqes; ++profile_->batches;
        }
    }

    __aicore__ void Publish(GM_ADDR base, uint64_t offset, uint32_t rank, uint32_t worker = 0)
    {
        Flush();
        if (!Tracing()) { Base::Publish(base, offset, rank, worker); return; }
        for (uint32_t lane = 0; lane < Qps; ++lane) {
            auto* one = Signal(base, offset, rank) + worker * Qps + lane;
            for (uint32_t i = 0; i < count_; ++i) {
                const uint32_t p = lane * Capacity + i;
                aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kDone>(one, one,
                    reinterpret_cast<__ubuf__ uint64_t*>(scratch_), 1U, peers_[p].peer, qp_base_ + lane, EVENT_ID5);
                trace_[p][4] = profile_->Elapsed();
            }
        }
    }
    __aicore__ void Quiet() {
        if (!Tracing()) { Base::Quiet(); return; }
        for (uint32_t lane = 0; lane < Qps; ++lane) for (uint32_t i = 0; i < count_; ++i) {
            const uint32_t p = lane * Capacity + i;
            const uint32_t alias = trace_[p][11];
            if (alias == p) {
                const uint64_t start = Now();
                aclshmemx_udma_qp_quiet(peers_[p].peer, qp_base_ + lane);
                trace_[p][5] = profile_->Elapsed(); trace_[p][10] = Now() - start;
            } else { trace_[p][5] = trace_[alias][5]; trace_[p][10] = trace_[alias][10]; }
        }
    }
    __aicore__ void StoreTrace(uint64_t output, uint32_t core, AscendC::LocalTensor<uint64_t> scratch) const {
        if (!Tracing()) return;
        AscendC::GlobalTensor<uint64_t> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(output) + uint64_t(core) * profile_->Stride());
        for (uint32_t lane = 0; lane < Qps; ++lane) for (uint32_t i = 0; i < count_; ++i) {
            for (uint32_t w = 0; w < kTraceSendWords; ++w) scratch.SetValue(w, trace_[lane * Capacity + i][w]);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID5);
            AscendC::DataCopyPad(out[kDispatchProfileWords + (lane * count_ + i) * kTraceSendWords], scratch,
                AscendC::DataCopyExtParams{1, kTraceSendWords * uint32_t(sizeof(uint64_t)), 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        }
    }

private:
    __aicore__ bool Tracing() const { return profile_ && profile_->trace; }
    __aicore__ void TraceRows(uint32_t p, uint32_t n) {
        if (!Tracing()) return;
        trace_[p][6] += n; trace_pending_[p / 32] |= 1U << (p % 32);
    }
    __aicore__ uint64_t Now() const { return profile_ && profile_->detailed ? profile_->Now() : 0; }
    __aicore__ void Credit(uint32_t p) {
        const uint64_t started = Now();
        const bool pressure = Base::EnsurePayloadCredit(p);
        if (profile_ && profile_->enabled && profile_->detailed) {
            profile_->Add(profile_->credit_ns, started);
            ++profile_->credit_checks; profile_->credit_stalls += pressure;
        }
        if (Tracing() && pressure) { ++trace_[p][8]; trace_[p][9] += Now() - started; }
    }
    struct CacheEntry {
        uint64_t next_indices = 0;
        uint32_t peer = UINT32_MAX, next_row = 0, next_field = 0, cursor = 0, count = 0;
    };
    __aicore__ uint32_t PutCachedRows(uint32_t p, const __gm__ int32_t* indices, uint32_t row,
        uint32_t field, uint32_t count, uint32_t available)
    {
        const uint32_t fields = td_.has_x_scales && td_.num_scales && td_.scale_element_bytes &&
                                       !(td_.dispatch_experiment_flags & (kDispatchSimtScales | kDispatchPackedScales)) ? 2U : 1U;
        if (!count || available < count || field >= fields) { trap(); return 0; }
        const uint32_t slot = p % kDispatchCacheEntries;
        auto& entry = cache_[slot];
        auto* built = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kCacheOffset + slot * kBatchWqes * kWqeBytes);
        const bool hit = entry.peer == p && entry.next_indices == reinterpret_cast<uint64_t>(indices) &&
            entry.next_row == row && entry.next_field == field && entry.cursor < entry.count;
        if (!hit) {
            // Only an entry referenced by this pending batch needs a drain.
            // Use the consumed epoch scratch as a separate refill descriptor,
            // so a first visit to another peer never destroys pending spans.
            if (cache_readers_ & (1U << slot)) Flush();
            const uint64_t started = Now();
            const uint32_t fill = available < kBatchWqes ? available : kBatchWqes;
            auto* span = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kDispatchEpochOffset);
            span[0] = reinterpret_cast<uint64_t>(indices);
            span[1] = uint64_t(row) | (uint64_t(fill) << 32U);
            span[2] = uint64_t(p) | (uint64_t(field) << 32U);
            span[3] = 0;
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
            asc_vf_call<BuildDispatchWqes>(dim3(32), built, span,
                reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kHeadersOffset),
                reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kDispatchBasesOffset), x_, scales_,
                td_.hidden * td_.x_element_bytes, td_.num_scales * td_.scale_element_bytes,
                td_.num_tokens, fields, 1U, fill);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
            entry = {reinterpret_cast<uint64_t>(indices), p, row, field, 0, fill};
            if (profile_ && profile_->enabled && profile_->detailed) {
                profile_->Add(profile_->build_ns, started);
                ++profile_->refills;
            }
        } else if (profile_ && profile_->enabled && profile_->detailed) ++profile_->hits;
        if (rows_pending_ && active_state_ != peers_[p].state) Flush();
        if (!rows_pending_) {
            active_state_ = peers_[p].state;
            // The checkpoint may overwrite the batch buffer, but not cache.
            Credit(p);
        }
        uint32_t n = count < entry.count - entry.cursor ? count : entry.count - entry.cursor;
        n = n < kBatchWqes - rows_pending_ ? n : kBatchWqes - rows_pending_;
        auto* span = reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset) + spans_ * 4U;
        span[0] = reinterpret_cast<uint64_t>(built + entry.cursor * 8U);
        span[1] = n;
        span[2] = 0;
        span[3] = rows_pending_;
        ++spans_;
        rows_pending_ += n;
        TraceRows(p, n);
        cache_readers_ |= 1U << slot;
        entry.cursor += n;
        entry.next_indices += uint64_t((field + n) / fields) * sizeof(int32_t);
        entry.next_row += (field + n) / fields;
        entry.next_field = (field + n) % fields;
        last_ = p;
        if (rows_pending_ == kBatchWqes) Flush();
        return n;
    }
    const DispatchTilingData& td_;
    uint64_t x_, scales_;
    DispatchProfile* profile_;
    bool cache_enabled_;
    uint64_t trace_[Capacity * Qps][kTraceSendWords];
    uint32_t trace_pending_[(Capacity * Qps + 31) / 32]{};
    CacheEntry cache_[kDispatchCacheEntries]{};
    uint32_t cache_readers_ = 0;
    uint32_t rows_pending_ = 0, spans_ = 0;
    uint32_t next_qp_[Qps == 2 ? Capacity : 1]{};
};
using DispatchSender = DispatchSenderT<>;
#endif
} // namespace ascend_deepep::udma
