#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

/* control.c */
size_t cuda_budget_deficit(int device, size_t bytes);

#if defined(_WIN32) || defined(_WIN64)

#define SHARED_EXPORT __declspec(dllexport)

#include <BaseTsd.h>

typedef SSIZE_T ssize_t;

/* shmem-detect.c */
bool plat_init(CUdevice dev);
void plat_cleanup();
size_t wddm_budget_deficit(int device, size_t bytes);

#else

#define SHARED_EXPORT

static inline bool plat_init(CUdevice dev) { return true; }
static inline void plat_cleanup() {}
static inline size_t wddm_budget_deficit(int device, size_t bytes) {
    return cuda_budget_deficit(device, bytes);
}

#endif

typedef unsigned long long ull;
#define K 1024
#define M (K * K)

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

/* control.c */
extern uint64_t vram_capacity;
extern uint64_t total_vram_usage;

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
fail_mmap:
    CHECK_CU(cuMemRelease(h));
fail:
    return err;
}

/* model_vbar.c */
void vbars_free(size_t size);
