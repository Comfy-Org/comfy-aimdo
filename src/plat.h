#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

#if defined(_WIN32) || defined(_WIN64)
#define SHARED_EXPORT __declspec(dllexport)
#else
#define SHARED_EXPORT
#endif

typedef unsigned long long ull;

enum DebugLevels {
    __NONE__ = 0,
    CRITICAL,
    ERROR,
    WARNING,
    INFO,
    DEBUG
};

/* debug.c */
extern int log_level;
const char *get_level_str(int level);

#define log(level, ...) if (log_level >= (level)) {                                         \
    fprintf(stderr, "aimdo: %s:%d:%s:", __FILE__, __LINE__, get_level_str(level));          \
    fprintf(stderr, __VA_ARGS__);                                                           \
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

    *handle = h;
    return CUDA_SUCCESS;

fail_access:
    CHECK_CU(cuMemUnmap(vaddr, size));
fail_mmap:
    CHECK_CU(cuMemRelease(h));
fail:
    return err;
}

void vbars_free(size_t size); /* model-vbar.c */
