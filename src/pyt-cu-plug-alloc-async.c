#include "plat.h"

#define CUDA_PAGE_SIZE (2 << 20)
#define ALIGN_UP(s) (((s) + CUDA_PAGE_SIZE - 1) & ~(CUDA_PAGE_SIZE - 1))
#define SIZE_HASH_SIZE 1024

typedef struct SizeEntry {
    void *ptr;
    size_t size;
    struct SizeEntry *next;
} SizeEntry;

static SizeEntry *size_table[SIZE_HASH_SIZE];

static inline unsigned int size_hash(void *ptr) {
    return ((uintptr_t)ptr >> 10 ^ (uintptr_t)ptr >> 21) % SIZE_HASH_SIZE;
}

int aimdo_cuda_malloc_async(void **devPtr, size_t size, void *hStream) {
    CUdeviceptr dptr;
    CUresult status;
    CUdevice device;

    log(VVERBOSE, "%s (start) size=%zuk stream=%p\n", __func__, size / K, hStream);
    if (!devPtr ||
        !CHECK_CU(cuCtxGetDevice(&device))) {
        return 1;
    }
    vbars_free(wddm_budget_deficit(device, size));

    if (CHECK_CU(cuMemAllocAsync(&dptr, size, (CUstream)hStream))) {
        *devPtr = (void *)dptr;
        goto success;
    }
    vbars_free(size);
    status = cuMemAllocAsync(&dptr, size, (CUstream)hStream);
    if (CHECK_CU(status)) {
        *devPtr = (void *)dptr;
        goto success;
    }

    *devPtr = NULL;
    return status; /* Fail */

success:

    total_vram_usage += ALIGN_UP(size);

    {
        unsigned int h = size_hash(*devPtr);
        SizeEntry *entry = (SizeEntry *)malloc(sizeof(*entry));
        if (entry) {
            entry->ptr = *devPtr;
            entry->size = size;
            entry->next = size_table[h];
            size_table[h] = entry;
        }
    }

    log(VVERBOSE, "%s (return): ptr=%p\n", __func__, *devPtr);
    return 0;
}

int aimdo_cuda_free_async(void *devPtr, void *hStream) {
    SizeEntry *entry;
    SizeEntry **prev;
    unsigned int h;
    CUresult status;

    log(VVERBOSE, "%s (start) ptr=%p\n", __func__, devPtr);

    if (!devPtr) {
        return 0;
    }

    h = size_hash(devPtr);
    entry = size_table[h];
    prev = &size_table[h];

    while (entry) {
        if (entry->ptr == devPtr) {
            *prev = entry->next;

            log(VVERBOSE, "Freed: ptr=%p, size=%zuk, stream=%p\n", devPtr, entry->size / K, hStream);
            status = cuMemFreeAsync((CUdeviceptr)devPtr, (CUstream)hStream);
            if (CHECK_CU(status)) {
                total_vram_usage -= ALIGN_UP(entry->size);
            }

            free(entry);
            return status;
        }
        prev = &entry->next;
        entry = entry->next;
    }

    log(ERROR, "%s: could not account free at %p\n", __func__, devPtr);
    return cuMemFreeAsync((CUdeviceptr)devPtr, (CUstream)hStream);
}
