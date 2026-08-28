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
