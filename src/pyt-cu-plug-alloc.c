#include "plat.h"
#include "vrambuf.h"

#define VMM_HASH_SHIFT  21

static inline unsigned int vmm_hash(CUdeviceptr ptr) {
    return ((uintptr_t)(void *)ptr >> VMM_HASH_SHIFT) % VMM_HASH_SIZE;
}

void allocations_analyze(bool only_dirty) {
    size_t total_size = 0;
    int count = 0;

    if (only_dirty && !allocations_dirty) {
        return;
    }
    allocations_dirty = false;

    log(DEBUG, "--- Allocation Analysis Start ---\n");

    for (int i = 0; i < VMM_HASH_SIZE; i++) {
        VramBuffer *entry = vmm_table[i];
        while (entry) {
            void* ptr = (void*)vrambuf_get(entry);
            size_t s = entry->allocated;

            log(DEBUG, "  [Bucket %4d] Ptr: %p | Size: %7zuk\n",
                i, ptr, s / K);

            total_size += s;
            count++;

            entry = entry->next;
        }
    }

    log(DEBUG, "%d Active Allocations for a total of %7zu MB\n", count, total_size / M);
}

SHARED_EXPORT
void *alloc_fn(size_t size, int device, cudaStream_t stream) {
    VramBuffer *entry;

    log(VERBOSE, "%s (start): size=%zuk, device=%d\n", __func__, size / K, device);

    if (!set_devctx_for_device(device)) {
        return NULL;
    }

    entry = vrambuf_create(device, size);
    if (!entry) {
        return NULL;
    }
    if (!vrambuf_grow(entry, size)) {
        vrambuf_destroy(entry);
        return NULL;
    }

    {
        unsigned int h = vmm_hash(vrambuf_get(entry));
        entry->next = vmm_table[h];
        vmm_table[h] = entry;
        allocations_dirty = true;
    }

    log(VERBOSE, "%s (return): ptr=%p\n", __func__, (void *)vrambuf_get(entry));
    return (void *)vrambuf_get(entry);
}

SHARED_EXPORT
void free_fn(void* ptr, size_t size, int device, cudaStream_t stream) {
    log_shot(DEBUG, "Pytorch is freeing VRAM ...\n");
    log(VERBOSE, "%s (start) ptr=%p size=%zuk, device=%d\n", __func__, ptr, size / K, device);
    if (ptr == NULL || !set_devctx_for_device(device)) {
        return;
    }

    for (VramBuffer **curr = &vmm_table[vmm_hash((CUdeviceptr)ptr)]; *curr; curr = &(*curr)->next) {
        VramBuffer *entry = *curr;
        if (vrambuf_get(entry) != (CUdeviceptr)ptr || entry->device != device) {
            continue;
        }

        *curr = entry->next;
        allocations_dirty = true;
        vrambuf_destroy(entry);
        log(VERBOSE, "Freed: ptr=%p, size=%zuk, stream=%p\n", ptr, size / K, stream);
        return;
    }

    log(ERROR, "%s could not find VRAM@%p\n", __func__, ptr);
}
