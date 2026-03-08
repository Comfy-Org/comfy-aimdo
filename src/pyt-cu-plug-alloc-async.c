#include "plat.h"

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

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
static CRITICAL_SECTION size_table_lock;
static volatile LONG size_table_lock_init;

static inline void st_lock(void) {
    if (!InterlockedCompareExchange(&size_table_lock_init, 1, 0)) {
        InitializeCriticalSection(&size_table_lock);
        InterlockedExchange(&size_table_lock_init, 2);
    }
    while (size_table_lock_init != 2) { /* spin until init done */ }
    EnterCriticalSection(&size_table_lock);
}
static inline void st_unlock(void) { LeaveCriticalSection(&size_table_lock); }
#else
#include <pthread.h>
static pthread_mutex_t size_table_lock = PTHREAD_MUTEX_INITIALIZER;
static inline void st_lock(void) { pthread_mutex_lock(&size_table_lock); }
static inline void st_unlock(void) { pthread_mutex_unlock(&size_table_lock); }
#endif

int aimdo_cuda_malloc_async(CUdeviceptr *devPtr, size_t size, CUstream hStream,
                            int (*true_cuMemAllocAsync)(CUdeviceptr*, size_t, CUstream)) {
    CUdeviceptr dptr;
    CUresult status = 0;

    log(VVERBOSE, "%s (start) size=%zuk stream=%p\n", __func__, size / K, hStream);

    if (!devPtr) {
        return 1;
    }

    vbars_free(budget_deficit(size));

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

    st_lock();
    total_vram_usage += CUDA_ALIGN_UP(size);

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
    st_unlock();

    log(VVERBOSE, "%s (return): ptr=%p\n", __func__, *devPtr);
    return 0;
}

int aimdo_cuda_free_async(CUdeviceptr devPtr, CUstream hStream,
                          int (*true_cuMemFreeAsync)(CUdeviceptr, CUstream)) {
    SizeEntry *entry;
    SizeEntry **prev;
    unsigned int h;
    CUresult status;

    log(VVERBOSE, "%s (start) ptr=%p\n", __func__, devPtr);

    if (!devPtr) {
        return 0;
    }

    st_lock();
    h = size_hash(devPtr);
    entry = size_table[h];
    prev = &size_table[h];

    while (entry) {
        if (entry->ptr == devPtr) {
            *prev = entry->next;

            log(VVERBOSE, "Freed: ptr=0x%llx, size=%zuk, stream=%p\n", devPtr, entry->size / K, hStream);
            status = true_cuMemFreeAsync(devPtr, hStream);
            if (CHECK_CU(status)) {
                total_vram_usage -= CUDA_ALIGN_UP(entry->size);
            }

            st_unlock();
            free(entry);
            return status;
        }
        prev = &entry->next;
        entry = entry->next;
    }
    st_unlock();

    log(ERROR, "%s: could not account free at %p\n", __func__, devPtr);
    return true_cuMemFreeAsync(devPtr, hStream);
}

#if !defined(_WIN32) && !defined(_WIN64)

cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
    if (!devPtr) {
        return 1; /* cudaErrorInvalidValue */
    }

    return aimdo_cuda_malloc_async((CUdeviceptr*)devPtr, size,
                                   (CUstream)stream, cuMemAllocAsync) ?
                2 /* cudaErrorMemoryAllocation */ : 0;
}

cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    /* CUresult and cudaError_t values are identical in CUDA 12+ for all
     * errors cuMemFreeAsync can return (1, 3, 4, 101, 201, 801).
     */
    return (cudaError_t)aimdo_cuda_free_async((CUdeviceptr)devPtr, (CUstream)stream, cuMemFreeAsync);
}

#endif
