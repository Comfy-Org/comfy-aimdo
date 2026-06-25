#include "plat.h"

/* cudaMalloc does not guarantee fragmentation handling as well as cudaMallocAsync,
 * so we reserve a small extra headroom when forcing budget pressure.
 */
#define CUDA_MALLOC_HEADROOM (128 * M)

typedef struct SizeEntry {
    CUdeviceptr ptr;
    size_t size;
    struct SizeEntry *next;
} SizeEntry;

static inline unsigned int size_hash(CUdeviceptr ptr) {
    return ((uintptr_t)ptr >> 10 ^ (uintptr_t)ptr >> 21) % SIZE_HASH_SIZE;
}

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>

static inline bool st_init(void) {
    CRITICAL_SECTION *lock = (CRITICAL_SECTION *)malloc(sizeof(*lock));

    if (!lock) {
        return false;
    }

    InitializeCriticalSection(lock);
    size_table_lock = lock;
    return true;
}

static inline void st_cleanup(void) {
    CRITICAL_SECTION *lock = (CRITICAL_SECTION *)size_table_lock;

    if (!lock) {
        return;
    }

    DeleteCriticalSection(lock);
    free(lock);
    size_table_lock = NULL;
}
static inline void st_lock(void) {
    EnterCriticalSection((CRITICAL_SECTION *)size_table_lock);
}
static inline void st_unlock(void) { LeaveCriticalSection((CRITICAL_SECTION *)size_table_lock); }
#else
#include <pthread.h>

static inline bool st_init(void) {
    pthread_mutex_t *lock = (pthread_mutex_t *)malloc(sizeof(*lock));

    if (!lock || pthread_mutex_init(lock, NULL) != 0) {
        free(lock);
        return false;
    }

    size_table_lock = lock;
    return true;
}

static inline void st_cleanup(void) {
    pthread_mutex_t *lock = (pthread_mutex_t *)size_table_lock;

    if (!lock) {
        return;
    }

    pthread_mutex_destroy(lock);
    free(lock);
    size_table_lock = NULL;
}

static inline void st_lock(void) { pthread_mutex_lock((pthread_mutex_t *)size_table_lock); }
static inline void st_unlock(void) { pthread_mutex_unlock((pthread_mutex_t *)size_table_lock); }
#endif

bool allocations_init(void) {
    return st_init();
}

void allocations_cleanup(void) {
    st_cleanup();
}

static inline size_t accounted_alloc_size(size_t size) {
    size_t rounded = size;

    /* A best guess at cuda async allocator small allocation handling.
     * we know it can sub-page small allocations, but we still want to
     * be conservative and not trust it all the way so clamp the lower
     * end. Assume page granularity when it makes sense to round up
     * as so. Just estimate with the actual for everything else in
     * between.
     */

    if (rounded <= K) {
        return K;
    }
    if (rounded > CUDA_PAGE_SIZE / 2) {
        return CUDA_ALIGN_UP(rounded);
    }

    return rounded;
}

static inline void account_alloc(CUdeviceptr ptr, size_t size) {
    unsigned int h = size_hash(ptr);
    SizeEntry *entry;

    st_lock();
    total_vram_usage += accounted_alloc_size(size);

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
            total_vram_usage -= accounted_alloc_size(entry->size);

            st_unlock();
            free(entry);
            return;
        }
        prev = &entry->next;
        entry = entry->next;
    }
    st_unlock();

    log(DEBUG, "%s: could not account free at %p\n", __func__, (void *)(uintptr_t)ptr);
}

/* Exposed accounting hooks for backends that intercept allocation at a
 * different API layer (e.g. the Level Zero detour in src-xpu/ze-detour.c).
 */
void aimdo_account_alloc(CUdeviceptr ptr, size_t size) {
    account_alloc(ptr, size);
}

void aimdo_account_free(CUdeviceptr ptr) {
    account_free(ptr, NULL);
}

int aimdo_cuda_malloc(CUdeviceptr *devPtr, size_t size,
                      CUresult (*true_cuMemAlloc_v2)(CUdeviceptr*, size_t)) {
    CUdeviceptr dptr;
    CUresult status = 0;

    if (!devPtr || !true_cuMemAlloc_v2) {
        return 1;
    }
    if (!set_devctx_for_current_cuda_device()) {
        /* this is not our device at all - straight passthrough */
        return true_cuMemAlloc_v2(devPtr, size);
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
    if (!set_devctx_for_current_cuda_device()) {
        return true_cuMemFree_v2(devPtr);
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

    if (!devPtr || !true_cuMemAllocAsync) {
        return 1;
    }
    if (!set_devctx_for_current_cuda_device()) {
        return true_cuMemAllocAsync(devPtr, size, hStream);
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

    log(VVERBOSE, "%s (return): ptr=%p\n", __func__, (void *)(uintptr_t)*devPtr);
    return 0;
}

int aimdo_cuda_free_async(CUdeviceptr devPtr, CUstream hStream,
                          CUresult (*true_cuMemFreeAsync)(CUdeviceptr, CUstream)) {
    CUresult status;

    log(VVERBOSE, "%s (start) ptr=%p\n", __func__, (void *)(uintptr_t)devPtr);

    if (!devPtr) {
        return 0;
    }
    if (!true_cuMemFreeAsync) {
        return 1;
    }
    if (!set_devctx_for_current_cuda_device()) {
        return true_cuMemFreeAsync(devPtr, hStream);
    }

    status = true_cuMemFreeAsync(devPtr, hStream);
    if (!CHECK_CU(status)) {
        return status;
    }

    account_free(devPtr, hStream);
    return status;
}
