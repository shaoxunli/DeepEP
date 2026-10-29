#pragma once

#include "ascend_deepep/platform.h"
#include "catlass/arch/arch.hpp"

#ifndef CATLASS_ARCH
#error "CATLASS_ARCH must be provided by the kernel build"
#endif

namespace ascend_deepep {

#if CATLASS_ARCH == 3510
using KernelArchBase = Catlass::Arch::Ascend950;
#elif CATLASS_ARCH == 2201
// Catlass names the shared 2201 architecture tag AtlasA2. Atlas A3 uses the
// same dav-2201 instruction architecture and the same 192 KiB UB capacity.
using KernelArchBase = Catlass::Arch::AtlasA2;
#else
#error "Unsupported CATLASS_ARCH for ascend-deepep"
#endif

struct KernelArch : KernelArchBase {
    static constexpr std::uint32_t UB_SIZE =
        static_cast<std::uint32_t>(platform::kKernelUbBytes);
};

static_assert(KernelArch::UB_SIZE <= KernelArchBase::UB_SIZE);

}  // namespace ascend_deepep
