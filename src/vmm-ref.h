#pragma once

#include "gpu_abi.h"

#include <stddef.h>

typedef struct VirtualRange {
    CUdeviceptr address;
    size_t bytes;
    size_t refs;
} VirtualRange;

typedef struct PhysicalAllocation {
    CUmemGenericAllocationHandle handle;
    size_t bytes;
    size_t refs;
    struct PhysicalPage *references;
} PhysicalAllocation;

typedef struct PhysicalPage {
    PhysicalAllocation *allocation;
    CUdeviceptr address;
    struct PhysicalPage *next;
} PhysicalPage;

VirtualRange *virtual_range_alloc(size_t bytes, size_t alignment);
VirtualRange *virtual_range_ref(VirtualRange *range);
CUresult virtual_range_unref(VirtualRange *range);
CUresult physical_page_alloc(PhysicalPage **page, size_t bytes, int device);
PhysicalPage *physical_page_ref(PhysicalPage *page, CUdeviceptr address);
CUresult physical_page_unref(PhysicalPage *page);

static inline CUdeviceptr virtual_range_get(VirtualRange *range) {
    return range->address;
}

static inline CUmemGenericAllocationHandle physical_page_get(PhysicalPage *page) {
    return page->allocation->handle;
}
