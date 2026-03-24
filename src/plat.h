#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#  include <hip/hip_runtime.h>
#  include "plat_hip.h"
#else
#  include <cuda.h>
#  include "plat_cuda.h"
#endif

/* NOTE: cuda_runtime.h is banned here. Always use the driver APIs.
 * Add duck-types in plat_cuda.h
 */

#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

/* control.c */
bool cuda_budget_deficit();

#if defined(_WIN32) || defined(_WIN64)

#define SHARED_EXPORT __declspec(dllexport)

#include <BaseTsd.h>

typedef SSIZE_T ssize_t;

/* shmem-detect.c */
bool aimdo_wddm_init(CUdevice dev);
void aimdo_wddm_cleanup();
bool poll_budget_deficit();
/* cuda-detour.c */
bool aimdo_setup_hooks();
void aimdo_teardown_hooks();

#else

#define SHARED_EXPORT

static inline bool aimdo_wddm_init(CUdevice dev) { return true; }
static inline void aimdo_wddm_cleanup() {}

static inline bool poll_budget_deficit() {
    return cuda_budget_deficit();
}

#endif

static inline bool plat_init(CUdevice dev) {
    return aimdo_wddm_init(dev) &&
           aimdo_setup_hooks();
}
static inline void plat_cleanup() {
    aimdo_wddm_cleanup();
    aimdo_teardown_hooks();
}

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/* NOTE: align_to must be power of 2 */
#define ALIGN_UP(x, align_to) (((x) + (align_to) - 1) & ~((align_to) - 1))

#define CUDA_PAGE_SIZE   (2 << 20)
#define CUDA_ALIGN_UP(s) ALIGN_UP(s, CUDA_PAGE_SIZE)

typedef unsigned long long ull;
#define K 1024
#define M (K * K)
#define G (M * K)

enum DebugLevels {
    __NONE__ = -1,
    /* Default to everything so if python itegration is hosed, we see prints. */
    ALL = 0,
    CRITICAL,
    ERROR,
    WARNING,
    INFO,
    DEBUG,
    VERBOSE,
    VVERBOSE,
};

/* debug.c */
extern int log_level;
extern uint64_t log_shot_counter;
const char *get_level_str(int level);
void log_reset_shots();

#define do_log(do_shot_counter, level, ...) {                                                   \
    static uint64_t _sc_;                                                                       \
    if ((!log_level || log_level >= (level)) && _sc_ < log_shot_counter) {                      \
        _sc_ = (do_shot_counter) ? log_shot_counter : 0;                                        \
        fprintf(stderr, "aimdo: %s:%d:%s:", __FILE__, __LINE__, get_level_str(level));          \
        fprintf(stderr, __VA_ARGS__);                                                           \
        fflush(stderr);                                                                         \
    }                                                                                           \
}

#define log(level, ...) do_log(false, level, __VA_ARGS__)
#define log_shot(level, ...) do_log(true, level, __VA_ARGS__)

/* The default VRAM headroom. Different deficit methods with BYO headroom */
#define VRAM_HEADROOM (256 * 1024 * 1024)

/* control.c */
extern uint64_t vram_capacity;
extern uint64_t total_vram_usage;
extern uint64_t total_vram_last_check;
extern ssize_t deficit_sync;
extern const char *prevailing_deficit_method;

static inline size_t budget_deficit(size_t size) {
    ssize_t deficit_simple, deficit_delta;
    size_t deficit;

    poll_budget_deficit();
    deficit_simple = (ssize_t)(total_vram_usage + VRAM_HEADROOM + size) - (ssize_t)vram_capacity;
    deficit_delta = deficit_sync + (ssize_t)total_vram_usage - (ssize_t)total_vram_last_check + size;
    deficit = (size_t)MAX(MAX(deficit_simple, deficit_delta), (ssize_t)0);
    if (deficit) {
        log(DEBUG, "%s: Prevailing Method: %s Deficit: %zu Alloc Size %zu\n", __func__,
            deficit_simple > deficit_delta ? "simple" : prevailing_deficit_method,
            deficit / M, size / M);
    }
    return deficit;
}

static inline int check_cu_impl(CUresult res, const char *label) {
    if (res != CUDA_SUCCESS && res != CUDA_ERROR_OUT_OF_MEMORY) {
        const char* desc;
        if (cuGetErrorString(res, &desc) != CUDA_SUCCESS) {
            desc = "<FATAL - CANNOT PARSE CUDA ERROR CODE>";

        }
        log(DEBUG, "CUDA API FAILED : %s : %s\n", label, desc);
    }
    return (res == CUDA_SUCCESS);
}
#define CHECK_CU(x) check_cu_impl((x), #x)

static inline CUresult three_stooges(CUdeviceptr vaddr, size_t size, int device,
                                     CUmemGenericAllocationHandle *handle) {
    CUmemGenericAllocationHandle h = 0;
    CUresult err;

    CUmemAllocationProp prop = {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .location.type = CU_MEM_LOCATION_TYPE_DEVICE,
        .location.id = device,
    };

    CUmemAccessDesc accessDesc = {
        .location.type = CU_MEM_LOCATION_TYPE_DEVICE,
        .location.id = device,
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };

    if (!CHECK_CU(err = cuMemCreate(&h, size, &prop, 0))) {
        goto fail;
    }
    if (!CHECK_CU(err = cuMemMap(vaddr, size, 0, h, 0))) {
        goto fail_mmap;
    }
    if (!CHECK_CU(err = cuMemSetAccess(vaddr, size, &accessDesc, 1))) {
        goto fail_access;
    }
    total_vram_usage += size;

    *handle = h;
    return CUDA_SUCCESS;

fail_access:
    CHECK_CU(cuMemUnmap(vaddr, size));
    unmap_workaround(vaddr, size);
fail_mmap:
    CHECK_CU(cuMemRelease(h));
fail:
    return err;
}

/* model_vbar.c */
size_t vbars_free(size_t size);
SHARED_EXPORT
uint64_t vbars_analyze(bool only_dirty);

/* pyt-cu-alloc.c */
int aimdo_cuda_malloc(CUdeviceptr *dptr, size_t size,
                      CUresult (*true_cuMemAlloc_v2)(CUdeviceptr*, size_t));
int aimdo_cuda_free(CUdeviceptr dptr,
                    CUresult (*true_cuMemFree_v2)(CUdeviceptr));

int aimdo_cuda_malloc_async(CUdeviceptr *devPtr, size_t size, CUstream hStream,
                            CUresult (*true_cuMemAllocAsync)(CUdeviceptr*, size_t, CUstream));
int aimdo_cuda_free_async(CUdeviceptr devPtr, CUstream hStream,
                          CUresult (*true_cuMemFreeAsync)(CUdeviceptr, CUstream));

void allocations_analyze();

/* control.c */
extern CUcontext aimdo_cuda_ctx;
SHARED_EXPORT
void aimdo_analyze();
