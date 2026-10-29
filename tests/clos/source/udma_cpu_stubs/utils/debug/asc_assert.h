#pragma once
#include "kernel_operator.h"
namespace __asc_simt_vf {
inline void __trap()
{
    trap();
}
}  // namespace __asc_simt_vf
