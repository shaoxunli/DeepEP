#pragma once

#include <cstdint>

#ifndef CATLASS_ARCH
#error "CATLASS_ARCH must be provided by the build system"
#endif

namespace ascend_deepep::platform {

#if CATLASS_ARCH == 3510
inline constexpr std::int64_t kMaxNumAiv = 64;
inline constexpr std::int64_t kKernelUbBytes = 216 * 1024;
inline constexpr std::int64_t kDispatchStageBytes = 192 * 1024;
#elif CATLASS_ARCH == 2201
inline constexpr std::int64_t kMaxNumAiv = 48;
inline constexpr std::int64_t kKernelUbBytes = 192 * 1024;
inline constexpr std::int64_t kDispatchStageBytes = 176 * 1024;
#else
#error "Unsupported CATLASS_ARCH for ascend-deepep"
#endif

static_assert(kMaxNumAiv > 0 && kMaxNumAiv % 2 == 0);
static_assert(kKernelUbBytes > 0 && kKernelUbBytes % 32 == 0);
static_assert(
    kDispatchStageBytes > 0 &&
    kDispatchStageBytes < kKernelUbBytes &&
    kDispatchStageBytes % 32 == 0);

}  // namespace ascend_deepep::platform
