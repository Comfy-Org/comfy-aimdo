#include "plat.h"

void aimdo_vbars_free(size_t size);

#define MIN_ALLOC_SHIFT 21
#define MIN_ALLOC       (1 << MIN_ALLOC_SHIFT) /*2 MB as per Cuda page sizes and pytorch cacheing alloc */

#define VMM_HASH_SIZE   (1 << 12) 

typedef struct VMMEntry {
    CUdeviceptr ptr;
    CUmemGenericAllocationHandle handle;
    size_t size;
    struct VMMEntry* next;
} VMMEntry;

static VMMEntry* vmm_table[VMM_HASH_SIZE];

static inline unsigned int vmm_hash(CUdeviceptr ptr) {
    return ((uintptr_t)(void *)ptr >> MIN_ALLOC_SHIFT) % VMM_HASH_SIZE;
}

SHARED_EXPORT
void *alloc_fn(size_t size, int device, cudaStream_t stream) {
    CUresult err;
    VMMEntry* entry = calloc(1, sizeof(*entry));

    if (!entry) {
        fprintf(stderr, "FATAL: Host OOM\n");
        return NULL;
    }

    entry->size = size;

    if (!CHECK_CU(err = cuMemAddressReserve(&entry->ptr, size, 0, 0, 0))) {
        goto fail;
    }
    /* FIXME: Think about looping this by chunk. Ideally we want to consume
        * what we can from cuda before we OOM so that the VBAR free routine
        * doesn't over-free
        */
    if (three_stooges(entry->ptr, size, device, &entry->handle) != CUDA_SUCCESS) {
        if (err != CUDA_ERROR_OUT_OF_MEMORY) {
            goto fail1;
        }
        fprintf(stderr, "DEBUG: OOMED\n");
        aimdo_vbars_free(size);
        if (three_stooges(entry->ptr, size, device, &entry->handle) != CUDA_SUCCESS) {
            goto fail1;
        }
    }

    {
        unsigned int h = vmm_hash(entry->ptr);
        entry->next = vmm_table[h];
        vmm_table[h] = entry;
    }

    printf("FK2 Custom Alloc: ptr=%p, size=%zu, device=%d phys(%llx)\n", entry->ptr, size, device, (unsigned long long)entry->handle);
    return (void *)entry->ptr;

fail1:
    cuMemAddressFree(entry->ptr, size);
fail:
    free(entry);
    return NULL;
}

SHARED_EXPORT
void free_fn(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (ptr == NULL) {
        return;
    }

    for (VMMEntry **curr = &vmm_table[vmm_hash(ptr)]; *curr; curr = &(*curr)->next) {
        VMMEntry *entry = *curr;
        if (entry->ptr != ptr) {
            continue;
        }

        CHECK_CU(cuMemUnmap(entry->ptr, entry->size));
        CHECK_CU(cuMemRelease(entry->handle));
        CHECK_CU(cuMemAddressFree(entry->ptr, entry->size));

        *curr = entry->next;
        free(entry);
        printf("Custom Free: ptr=%p, size=%zu, stream=%p\n", ptr, size, stream);
        return;
    }

    fprintf(stderr, "WARNING: free_fn could not find pointer %p in lookup table\n", ptr);
}