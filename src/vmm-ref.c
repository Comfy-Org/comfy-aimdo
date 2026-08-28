#include "plat.h"
#include "vmm-ref.h"

VirtualRange *virtual_range_alloc(size_t bytes, size_t alignment) {
    VirtualRange *range = malloc(sizeof(*range));

    if (!range) {
        return NULL;
    }
    if (cuMemAddressReserve(&range->address, bytes, alignment, 0, 0)) {
        free(range);
        return NULL;
    }
    range->bytes = bytes;
    range->refs = 1;
    return range;
}

VirtualRange *virtual_range_ref(VirtualRange *range) {
    allocations_lock();
    range->refs++;
    allocations_unlock();
    return range;
}

CUresult virtual_range_unref(VirtualRange *range) {
    allocations_lock();
    if (--range->refs) {
        allocations_unlock();
        return CUDA_SUCCESS;
    }

    CUresult result = cuMemAddressFree(range->address, range->bytes);
    if (result) {
        range->refs++;
    } else {
        free(range);
    }
    allocations_unlock();
    return result;
}

CUresult physical_page_alloc(PhysicalPage **page, size_t bytes, int device) {
    PhysicalPage *reference = malloc(sizeof(*reference));
    PhysicalAllocation *allocation = malloc(sizeof(*allocation));

    if (!reference || !allocation) {
        free(reference);
        free(allocation);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUmemAllocationProp prop = {.type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .location = {CU_MEM_LOCATION_TYPE_DEVICE, device}};
    CUresult result = cuMemCreate(&allocation->handle, bytes, &prop, 0);
    if (result) {
        free(reference);
        free(allocation);
        return result;
    }

    allocation->bytes = bytes;
    allocation->refs = 1;
    *reference = (PhysicalPage){.allocation = allocation};
    allocation->references = reference;
    *page = reference;
    allocations_lock();
    total_vram_usage += bytes;
    allocations_unlock();
    return CUDA_SUCCESS;
}

PhysicalPage *physical_page_ref(PhysicalPage *page, CUdeviceptr address) {
    PhysicalPage *reference = malloc(sizeof(*reference));

    if (!reference) {
        return NULL;
    }
    allocations_lock();
    page->allocation->refs++;
    *reference = (PhysicalPage){.allocation = page->allocation, .address = address,
                                .next = page->allocation->references};
    page->allocation->references = reference;
    allocations_unlock();
    return reference;
}

CUresult physical_page_unref(PhysicalPage *page) {
    CUresult result = CUDA_SUCCESS;

    allocations_lock();
    PhysicalPage **entry = &page->allocation->references;

    while (*entry != page) {
        entry = &(*entry)->next;
    }

    if (page->address) {
        PhysicalPage *reference = page->allocation->references;
        while (reference &&
               (reference == page || reference->address != page->address)) {
            reference = reference->next;
        }
        if (!reference) {
            result = cuMemUnmap(page->address, page->allocation->bytes);
            unmap_workaround(page->address, page->allocation->bytes);
            if (result) {
                allocations_unlock();
                return result;
            }
        }
        page->address = 0;
    }
    if (page->allocation->refs == 1 &&
        (result = cuMemRelease(page->allocation->handle))) {
        allocations_unlock();
        return result;
    }
    *entry = page->next;
    if (--page->allocation->refs == 0) {
        total_vram_usage -= page->allocation->bytes;
        free(page->allocation);
    }
    free(page);
    allocations_unlock();
    return CUDA_SUCCESS;
}
