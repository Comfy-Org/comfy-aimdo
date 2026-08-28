#pragma once

#include "gpu_abi.h"

#include <stddef.h>

typedef struct VirtualRange {
    CUdeviceptr address;
    size_t bytes;
    size_t refs;
} VirtualRange;

VirtualRange *virtual_range_alloc(size_t bytes, size_t alignment);
VirtualRange *virtual_range_ref(VirtualRange *range);
CUresult virtual_range_unref(VirtualRange *range);

static inline CUdeviceptr virtual_range_get(VirtualRange *range) {
    return range->address;
}
