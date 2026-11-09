/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Instrumented adaptation; the upstream license is in third-party/shmem/LICENSE.
#pragma once

#include "shmem.h"

namespace ascend_deepep::udma {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// Diagnostic-only copy of the pinned SDK V4 barrier. Signal addresses,
// generations, dissemination order, waits and local collectives are unchanged.
// Each AIV retains its own scalar timestamps until the ordinary profile store.
// Restricted to EP64/AIV32: radix 8, two rounds, one source per active AIV/round.
// Cross-device counters must not be directly compared.
template <bool IS_AIV_ONLY = true>
ACLSHMEM_DEVICE void ProfileCombineBeginSync(aclshmem_team_t team_idx, uint64_t* ticks)
{
    ticks[3] = AscendC::GetSystemCycle();
    aclshmemx_team_t* team = aclshmemi_get_state()->team_pools[team_idx];
    int32_t vec_id = AscendC::GetBlockIdx();
    // AIC and AIV must derive identical radices and local-barrier round counts.
    // Use logical blocks for communication work, not the core-dependent ratio.
    int32_t vec_size = AscendC::GetBlockNum();

    MSTX_FUSE_SCOPE_START();

    int32_t size = team->size;
    int32_t my_pe_in_team = team->mype;
    auto sync_pool = aclshmemi_get_team_sync_pool(team->team_idx);
    auto sync_counter = aclshmemi_get_team_sync_counter(team->team_idx);
    int32_t count = 0;

    int32_t k_max = aclshmemi_sync_v4_k_cap(size);
    int32_t runtime_k_cap = vec_size < size ? vec_size : size;
    if (size > 1) {
        runtime_k_cap = runtime_k_cap < SYNC_DISSEM_KVAL_MIN ? SYNC_DISSEM_KVAL_MIN : runtime_k_cap;
    }
    while (k_max > runtime_k_cap && k_max > 1) {
        k_max >>= 1;
    }

    // Minimize the number of dissemination rounds first. Among radices with
    // the same round count, use the smallest one to reduce remote signals and
    // local slot polling. Keep all selection work inside the V4 path so V3 has
    // no additional runtime overhead.
    int32_t phases = 0;
    int32_t temp = size > 1 ? size - 1 : 0;
    while (temp) {
        phases++;
        temp /= k_max;
    }

    int32_t k = k_max;
    for (int32_t candidate_k = SYNC_DISSEM_KVAL_MIN; candidate_k < k_max; candidate_k <<= 1) {
        int32_t candidate_phases = 0;
        temp = size > 1 ? size - 1 : 0;
        while (temp) {
            candidate_phases++;
            temp /= candidate_k;
        }
        if (candidate_phases == phases) {
            k = candidate_k;
            break;
        }
    }
    int32_t slot_stride = k - 1;
    if (size != 64 || vec_size != 32 || k != 8 || phases != 2) { trap(); return; }
    ticks[16] = k; ticks[17] = phases;

    // AIV0 advances one shared generation. Every AIV invalidates its stale
    // scalar-cache line before the generation is published.
    if ASCEND_IS_AIV {
        dcci_cacheline((__gm__ uint8_t*)sync_counter);
    }
    ticks[4] = AscendC::GetSystemCycle();
    aclshmemi_sync_core<IS_AIV_ONLY>();
    ticks[5] = AscendC::GetSystemCycle();
    if ASCEND_IS_AIV {
        if (vec_id == 0) {
            count = aclshmemi_load((__gm__ int32_t*)sync_counter) + 1;
            aclshmemi_signal_set((__gm__ int32_t*)sync_counter, count);
        }
    }
    aclshmemi_sync_core<IS_AIV_ONLY>();
    if ASCEND_IS_AIV {
        count = aclshmemi_load((__gm__ int32_t*)sync_counter);
    }

    ticks[6] = AscendC::GetSystemCycle();
    ticks[18] = count;
    int32_t pow_k = 1;
    for (int32_t round = 0; round < phases; round++) {
        if ASCEND_IS_AIV {
            for (int32_t i = vec_id + 1; i <= slot_stride; i += vec_size) {
                int32_t shift = i * pow_k;
                if (shift >= size) {
                    break;
                }
                int32_t to_nbr_idx = (my_pe_in_team + shift) % size;
                int32_t to_pe = team->pe_mapping[to_nbr_idx];
                int32_t slot = round * slot_stride + (i - 1);
                aclshmemi_signal_set((__gm__ int32_t*)aclshmem_ptr(sync_pool + slot, to_pe), count);
            }
            ticks[7 + round * 4] = AscendC::GetSystemCycle();
            for (int32_t i = vec_id + 1; i <= slot_stride; i += vec_size) {
                int32_t shift = i * pow_k;
                if (shift >= size) {
                    break;
                }
                int32_t slot = round * slot_stride + (i - 1);
                ticks[10 + round * 4] = team->pe_mapping[(my_pe_in_team - shift + size) % size] + 1;
                aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t*)(sync_pool + slot), count);
            }
        }
        ticks[8 + round * 4] = AscendC::GetSystemCycle();
        aclshmemi_sync_core<IS_AIV_ONLY>();
        ticks[9 + round * 4] = AscendC::GetSystemCycle();
        pow_k *= k;
    }

    ticks[15] = AscendC::GetSystemCycle();
    MSTX_FUSE_SCOPE_END();
    // All launched cores participate in the barrier, including AIVs outside
    // the communication worker budget. Report their actual per-side count.
    MSTX_BARRIER_NPU_REPORT(size, AscendC::GetBlockNum() * AscendC::GetTaskRation());
}

#endif
} // namespace ascend_deepep::udma
