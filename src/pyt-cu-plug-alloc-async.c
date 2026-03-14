#include "plat.h"

#define SIZE_HASH_SIZE 1024
/* cudaMalloc does not guarantee fragmentation handling as well as cudaMallocAsync,
 * so we reserve a small extra headroom when forcing budget pressure.
 */
#define CUDA_MALLOC_HEADROOM (128 * M)

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

static inline void account_alloc(CUdeviceptr ptr, size_t size) {
    unsigned int h = size_hash(ptr);
    SizeEntry *entry;

    st_lock();
    total_vram_usage += CUDA_ALIGN_UP(size);

    entry = (SizeEntry *)malloc(sizeof(*entry));
    if (entry) {
        entry->ptr = ptr;
        entry->size = size;
        entry->next = size_table[h];
        size_table[h] = entry;
    }
    st_unlock();
}

static inline void account_free(CUdeviceptr ptr, CUstream hStream) {
    SizeEntry *entry;
    SizeEntry **prev;
    unsigned int h = size_hash(ptr);

    st_lock();
    entry = size_table[h];
    prev = &size_table[h];

    while (entry) {
        if (entry->ptr == ptr) {
            *prev = entry->next;

            log(VVERBOSE, "Freed: ptr=0x%llx, size=%zuk, stream=%p\n", ptr, entry->size / K, hStream);
            total_vram_usage -= CUDA_ALIGN_UP(entry->size);

            st_unlock();
            free(entry);
            return;
        }
        prev = &entry->next;
        entry = entry->next;
    }
    st_unlock();

    log(ERROR, "%s: could not account free at %p\n", __func__, ptr);
}

int aimdo_cuda_malloc(CUdeviceptr *devPtr, size_t size,
                      CUresult (*true_cuMemAlloc_v2)(CUdeviceptr*, size_t)) {
    CUdeviceptr dptr;
    CUresult status = 0;

    if (!devPtr || !true_cuMemAlloc_v2) {
        return 1;
    }

    vbars_free(budget_deficit(size + CUDA_MALLOC_HEADROOM));

    if (CHECK_CU(true_cuMemAlloc_v2(&dptr, size))) {
        *devPtr = dptr;
        account_alloc(*devPtr, size);
        return 0;
    }

    vbars_free(size + CUDA_MALLOC_HEADROOM);
    status = true_cuMemAlloc_v2(&dptr, size);
    if (CHECK_CU(status)) {
        *devPtr = dptr;
        account_alloc(*devPtr, size);
        return 0;
    }

    *devPtr = 0;
    return status;
}

int aimdo_cuda_free(CUdeviceptr devPtr,
                    CUresult (*true_cuMemFree_v2)(CUdeviceptr)) {
    CUresult status;

    if (!devPtr) {
        return 0;
    }
    if (!true_cuMemFree_v2) {
        return 1;
    }

    status = true_cuMemFree_v2(devPtr);
    if (!CHECK_CU(status)) {
        return status;
    }

    account_free(devPtr, NULL);
    return status;
}

int aimdo_cuda_malloc_async(CUdeviceptr *devPtr, size_t size, CUstream hStream,
                            CUresult (*true_cuMemAllocAsync)(CUdeviceptr*, size_t, CUstream)) {
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
    account_alloc(*devPtr, size);

    log(VVERBOSE, "%s (return): ptr=%p\n", __func__, *devPtr);
    return 0;
}

int aimdo_cuda_free_async(CUdeviceptr devPtr, CUstream hStream,
                          CUresult (*true_cuMemFreeAsync)(CUdeviceptr, CUstream)) {
    CUresult status;

    log(VVERBOSE, "%s (start) ptr=%p\n", __func__, devPtr);

    if (!devPtr) {
        return 0;
    }

    status = true_cuMemFreeAsync(devPtr, hStream);
    if (!CHECK_CU(status)) {
        return status;
    }

    account_free(devPtr, hStream);
    return status;
}

#if !defined(_WIN32) && !defined(_WIN64)

static inline void ensure_ctx(void) {
    CUcontext ctx = NULL;

    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) {
        cuCtxSetCurrent(aimdo_cuda_ctx);
    }
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (!devPtr) {
        return 1; /* cudaErrorInvalidValue */
    }

    ensure_ctx();
    return aimdo_cuda_malloc((CUdeviceptr*)devPtr, size, cuMemAlloc_v2) ?
                2 /* cudaErrorMemoryAllocation */ : 0;
}

cudaError_t cudaFree(void* devPtr) {
    ensure_ctx();
    return (cudaError_t)aimdo_cuda_free((CUdeviceptr)devPtr, cuMemFree_v2);
}

cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
    if (!devPtr) {
        return 1; /* cudaErrorInvalidValue */
    }

    ensure_ctx();
    return aimdo_cuda_malloc_async((CUdeviceptr*)devPtr, size,
                                   (CUstream)stream, cuMemAllocAsync) ?
                2 /* cudaErrorMemoryAllocation */ : 0;
}

cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    ensure_ctx();
    /* CUresult and cudaError_t values are identical in CUDA 12+ for all
     * errors cuMemFreeAsync can return (1, 3, 4, 101, 201, 801).
     */
    return (cudaError_t)aimdo_cuda_free_async((CUdeviceptr)devPtr, (CUstream)stream, cuMemFreeAsync);
}

#endif
