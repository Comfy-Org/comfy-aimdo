#include "plat.h"

#ifdef _WIN32
#include <windows.h>
#define GET_TICK() GetTickCount64()
#else
#include <sys/time.h>
static inline uint64_t get_tick_linux() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
#define GET_TICK() get_tick_linux()
#endif

#define CUDA_PAGE_SIZE (2 << 20)
#define ALIGN_UP(s) (((s) + CUDA_PAGE_SIZE - 1) & ~(CUDA_PAGE_SIZE - 1))
#define SIZE_HASH_SIZE 1024

typedef struct SizeEntry {
    CUdeviceptr ptr;
    size_t size;
    struct SizeEntry *next;
} SizeEntry;

static SizeEntry *size_table[SIZE_HASH_SIZE];

static inline unsigned int size_hash(CUdeviceptr ptr) {
    return ((uintptr_t)ptr >> 10 ^ (uintptr_t)ptr >> 21) % SIZE_HASH_SIZE;
}

CUresult aimdo_cuda_malloc_async(CUdeviceptr *devPtr, size_t size, CUstream hStream,
                            CUresult (*true_cuMemAllocAsync)(CUdeviceptr*, size_t, CUstream)) {
    static uint64_t last_check = 0;
    uint64_t now = GET_TICK();
    CUdeviceptr dptr;
    CUresult status = 0;

    log(VVERBOSE, "%s (start) size=%zuk stream=%p\n", __func__, size / K, hStream);

    if (!devPtr) {
        return 1;
    }

    if (now - last_check >= 2000) {
        last_check = now;
        CUdevice device;
        if (!CHECK_CU(cuCtxGetDevice(&device))) {
            return 1;
        }
        vbars_free(wddm_budget_deficit(device, size));
    }

    if (CHECK_CU(true_cuMemAllocAsync(&dptr, size, hStream))) {
        *devPtr = dptr;
        goto success;
    }
    vbars_free(size);
    status = true_cuMemAllocAsync(&dptr, size, hStream);
    if (CHECK_CU(status)) {
        *devPtr = dptr;
        goto success;
    }

    *devPtr = 0;
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

CUresult aimdo_cuda_free_async(CUdeviceptr devPtr, CUstream hStream,
                          CUresult (*true_cuMemFreeAsync)(CUdeviceptr, CUstream)) {
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
            status = true_cuMemFreeAsync(devPtr, hStream);
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

#if !defined(_WIN32) && !defined(_WIN64)

CUresult cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
    if (!devPtr) {
        return 1; /* cudaErrorInvalidValue */
    }

    return aimdo_cuda_malloc_async((CUdeviceptr*)devPtr, size,
                                   (CUstream)stream, cuMemAllocAsync) ?
                2 /* cudaErrorMemoryAllocation */ : 0;
}

CUresult cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    return aimdo_cuda_free_async((CUdeviceptr)devPtr, (CUstream)stream, cuMemFreeAsync) ?
                400 /* cudaErrorInvalidDevicePointer */ : 0;
}

#endif
