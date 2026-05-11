#include "plat.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static void *g_cuda_module;

AimdoCudaDispatch g_cuda;

/* Keep the ABI target pinned to the minimum supported CUDA version so
 * cuGetProcAddress does not hand us newer function revisions with different
 * signatures.
 */
#define AIMDO_CUDA_ABI_VERSION 12060

typedef struct {
    void **slot;
    const char *symbol;
    cuuint64_t flags;
} DispatchSymbol;

static const DispatchSymbol dispatch_symbols[] = {
    { (void **)&g_cuda.p_cuInit, "cuInit", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuGetErrorString, "cuGetErrorString", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuCtxGetDevice, "cuCtxGetDevice", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuCtxSynchronize, "cuCtxSynchronize", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuDeviceGet, "cuDeviceGet", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuDeviceGetAttribute, "cuDeviceGetAttribute", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuDeviceTotalMem, "cuDeviceTotalMem", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuDeviceGetName, "cuDeviceGetName", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemGetInfo, "cuMemGetInfo", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemAlloc_v2, "cuMemAlloc", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemFree_v2, "cuMemFree", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemAllocAsync, "cuMemAllocAsync", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemAllocAsync_ptsz, "cuMemAllocAsync", CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM },
    { (void **)&g_cuda.p_cuMemFreeAsync, "cuMemFreeAsync", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemFreeAsync_ptsz, "cuMemFreeAsync", CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM },
    { (void **)&g_cuda.p_cuMemAllocHost, "cuMemAllocHost", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemFreeHost, "cuMemFreeHost", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemHostRegister, "cuMemHostRegister", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemHostUnregister, "cuMemHostUnregister", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemAddressReserve, "cuMemAddressReserve", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemAddressFree, "cuMemAddressFree", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemCreate, "cuMemCreate", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemMap, "cuMemMap", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemSetAccess, "cuMemSetAccess", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemUnmap, "cuMemUnmap", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemRelease, "cuMemRelease", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
#if defined(_WIN32) || defined(_WIN64)
    { (void **)&g_cuda.p_cuDeviceGetLuid, "cuDeviceGetLuid", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
#endif
};

static const char *const cuda_library_names[] = {
#if defined(_WIN32) || defined(_WIN64)
    "nvcuda.dll",
    "nvcuda64.dll",
#else
    "libcuda.so.1",
    "libcuda.so",
#endif
};

bool aimdo_cuda_runtime_init(void) {
    if (g_cuda.p_cuInit) {
        return true;
    }

    if (!(g_cuda_module = aimdo_find_loaded_module(
              cuda_library_names, ARRAY_SIZE(cuda_library_names)))) {
        return false;
    }

#if defined(_WIN32) || defined(_WIN64)
    g_cuda.p_cuGetProcAddress = (PFN_cuGetProcAddress)GetProcAddress((HMODULE)g_cuda_module,
                                                                     "cuGetProcAddress");
#else
    g_cuda.p_cuGetProcAddress = (PFN_cuGetProcAddress)dlsym(g_cuda_module,
                                                            "cuGetProcAddress");
#endif
    if (!g_cuda.p_cuGetProcAddress) {
        log(ERROR, "%s: failed to resolve cuGetProcAddress\n", __func__);
        aimdo_cuda_runtime_cleanup();
        return false;
    }

    for (size_t i = 0; i < ARRAY_SIZE(dispatch_symbols); i++) {
        void *resolved = NULL;

        if (g_cuda.p_cuGetProcAddress(dispatch_symbols[i].symbol, &resolved,
                                      AIMDO_CUDA_ABI_VERSION,
                                      dispatch_symbols[i].flags, NULL) != CUDA_SUCCESS) {
            resolved = NULL;
        }
        if (!resolved) {
            log(ERROR, "%s: failed to resolve required CUDA symbol %s\n", __func__,
                dispatch_symbols[i].symbol);
            aimdo_cuda_runtime_cleanup();
            return false;
        }
        *dispatch_symbols[i].slot = resolved;
    }

    if (g_cuda.p_cuInit(0) != CUDA_SUCCESS) {
        log(ERROR, "%s: cuInit failed\n", __func__);
        aimdo_cuda_runtime_cleanup();
        return false;
    }

    return true;
}

void aimdo_cuda_runtime_cleanup(void) {
    memset(&g_cuda, 0, sizeof(g_cuda));

    if (!g_cuda_module) {
        return;
    }

#if !defined(_WIN32) && !defined(_WIN64)
    dlclose(g_cuda_module);
#endif
    g_cuda_module = NULL;
}
