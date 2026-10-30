#pragma once

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace ascend_deepep::runtime {

// Address translation alone is insufficient: SHMEM reserves peer VA slots
// even for unreachable peers. Query the actual imported allocation range.
inline std::string MteMappingError(void* address, std::size_t bytes)
{
    if (address == nullptr || bytes == 0) {
        return "missing symmetric address or size";
    }
    void* base = nullptr;
    std::size_t size = 0;
    const auto status = aclrtMemGetAddressRange(address, &base, &size);
    if (status != ACL_SUCCESS) {
        return "aclrtMemGetAddressRange failed: " + std::to_string(status);
    }
    const auto begin = reinterpret_cast<uintptr_t>(base);
    const auto pointer = reinterpret_cast<uintptr_t>(address);
    if (base == nullptr || pointer < begin || pointer - begin > size || bytes > size - (pointer - begin)) {
        return "imported allocation does not cover the symmetric region";
    }
    return {};
}

}  // namespace ascend_deepep::runtime
