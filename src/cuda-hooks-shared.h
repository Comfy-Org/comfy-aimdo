/* Shared hook declarations for the CUDA hook backends.
 *
 * This is intentionally an includable implementation header, not a normal
 * interface header. It defines static functions and static data, and should
 * only be included directly by one backend .c file per translation unit.
 */

#pragma once

typedef struct {
    void **true_ptr;
    void **target_ptr;
    void *hook_ptr;
    const char *name;
} HookEntry;

static CUresult (CUDAAPI *true_cuMemAlloc_v2)(CUdeviceptr*, size_t);
static CUresult (CUDAAPI *true_cuMemFree_v2)(CUdeviceptr);
static CUresult (CUDAAPI *true_cuMemAllocAsync)(CUdeviceptr*, size_t, CUstream);
static CUresult (CUDAAPI *true_cuMemAllocAsync_ptsz)(CUdeviceptr*, size_t, CUstream);
static CUresult (CUDAAPI *true_cuMemFreeAsync)(CUdeviceptr, CUstream);
static CUresult (CUDAAPI *true_cuMemFreeAsync_ptsz)(CUdeviceptr, CUstream);

static CUresult CUDAAPI aimdo_cuMemAlloc_v2(CUdeviceptr *dptr, size_t size) {
    return aimdo_cuda_malloc(dptr, size, true_cuMemAlloc_v2);
}

static CUresult CUDAAPI aimdo_cuMemFree_v2(CUdeviceptr dptr) {
    return aimdo_cuda_free(dptr, true_cuMemFree_v2);
}

static CUresult CUDAAPI aimdo_cuMemAllocAsync(CUdeviceptr *dptr, size_t size, CUstream hStream) {
    return aimdo_cuda_malloc_async(dptr, size, hStream, true_cuMemAllocAsync);
}

static CUresult CUDAAPI aimdo_cuMemAllocAsync_ptsz(CUdeviceptr *dptr, size_t size, CUstream hStream) {
    return aimdo_cuda_malloc_async(dptr, size, hStream, true_cuMemAllocAsync_ptsz);
}

static CUresult CUDAAPI aimdo_cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    return aimdo_cuda_free_async(dptr, hStream, true_cuMemFreeAsync);
}

static CUresult CUDAAPI aimdo_cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream) {
    return aimdo_cuda_free_async(dptr, hStream, true_cuMemFreeAsync_ptsz);
}

static const HookEntry hooks[] = {
    { (void **)&true_cuMemAlloc_v2,        (void **)&g_cuda.p_cuMemAlloc_v2,        aimdo_cuMemAlloc_v2,        "cuMemAlloc_v2" },
    { (void **)&true_cuMemFree_v2,         (void **)&g_cuda.p_cuMemFree_v2,         aimdo_cuMemFree_v2,         "cuMemFree_v2" },
    { (void **)&true_cuMemAllocAsync,      (void **)&g_cuda.p_cuMemAllocAsync,      aimdo_cuMemAllocAsync,      "cuMemAllocAsync" },
    { (void **)&true_cuMemAllocAsync_ptsz, (void **)&g_cuda.p_cuMemAllocAsync_ptsz, aimdo_cuMemAllocAsync_ptsz, "cuMemAllocAsync_ptsz" },
    { (void **)&true_cuMemFreeAsync,       (void **)&g_cuda.p_cuMemFreeAsync,       aimdo_cuMemFreeAsync,       "cuMemFreeAsync" },
    { (void **)&true_cuMemFreeAsync_ptsz,  (void **)&g_cuda.p_cuMemFreeAsync_ptsz,  aimdo_cuMemFreeAsync_ptsz,  "cuMemFreeAsync_ptsz" },
};
