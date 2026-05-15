#include "plat.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static void *g_hip_module;

AimdoCudaDispatch g_cuda;
PFN_deviceGetProperties g_device_get_properties;

typedef CUresult (CUDAAPI *PFN_hipHostMalloc)(void **pp, size_t bytesize, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_hipHostRegister)(void *p, size_t bytesize, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_hipHostUnregister)(void *p);

typedef struct {
    void **slot;
    const char *symbol;
} DispatchSymbol;

static PFN_hipHostMalloc g_hip_host_malloc;
static PFN_hipHostRegister g_hip_host_register;
static PFN_hipHostUnregister g_hip_host_unregister;

static CUresult CUDAAPI aimdo_hip_mem_alloc_host(void **pp, size_t bytesize) {
    return g_hip_host_malloc(pp, bytesize, 0);
}

static const DispatchSymbol dispatch_symbols[] = {
    { (void **)&g_cuda.p_cuInit, "hipInit" },
    { (void **)&g_cuda.p_cuGetErrorString, "hipDrvGetErrorString" },
    { (void **)&g_cuda.p_cuCtxGetDevice, "hipGetDevice" },
    { (void **)&g_cuda.p_cuCtxSynchronize, "hipDeviceSynchronize" },
    { (void **)&g_cuda.p_cuDeviceGet, "hipDeviceGet" },
    { (void **)&g_cuda.p_cuDeviceTotalMem, "hipDeviceTotalMem" },
    { (void **)&g_cuda.p_cuDeviceGetName, "hipDeviceGetName" },
    { (void **)&g_cuda.p_cuMemGetInfo, "hipMemGetInfo" },
    { (void **)&g_cuda.p_cuMemAlloc_v2, "hipMalloc" },
    { (void **)&g_cuda.p_cuMemFree_v2, "hipFree" },
    { (void **)&g_cuda.p_cuMemAllocAsync, "hipMallocAsync" },
    { (void **)&g_cuda.p_cuMemFreeAsync, "hipFreeAsync" },
    { (void **)&g_hip_host_malloc, "hipHostMalloc" },
    { (void **)&g_hip_host_register, "hipHostRegister" },
    { (void **)&g_hip_host_unregister, "hipHostUnregister" },
    { (void **)&g_cuda.p_cuMemFreeHost, "hipHostFree" },
    { (void **)&g_cuda.p_cuMemAddressReserve, "hipMemAddressReserve" },
    { (void **)&g_cuda.p_cuMemAddressFree, "hipMemAddressFree" },
    { (void **)&g_cuda.p_cuMemCreate, "hipMemCreate" },
    { (void **)&g_cuda.p_cuMemMap, "hipMemMap" },
    { (void **)&g_cuda.p_cuMemSetAccess, "hipMemSetAccess" },
    { (void **)&g_cuda.p_cuMemUnmap, "hipMemUnmap" },
    { (void **)&g_cuda.p_cuMemRelease, "hipMemRelease" },
    { (void **)&g_cuda.p_cuMemcpyHtoDAsync, "hipMemcpyHtoDAsync" },
};

static const char *const hip_library_names[] = {
#if defined(_WIN32) || defined(_WIN64)
    "amdhip64.dll",
    "amdhip64_7.dll",
#else
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so",
#endif
};

static void *aimdo_hip_resolve_symbol(const char *symbol) {
#if defined(_WIN32) || defined(_WIN64)
    return (void *)GetProcAddress((HMODULE)g_hip_module, symbol);
#else
    return dlsym(g_hip_module, symbol);
#endif
}

bool aimdo_cuda_runtime_init(void) {
    if (g_cuda.p_cuInit) {
        return true;
    }

    if (!(g_hip_module = aimdo_find_loaded_module(
              hip_library_names, ARRAY_SIZE(hip_library_names)))) {
        return false;
    }

    for (size_t i = 0; i < ARRAY_SIZE(dispatch_symbols); i++) {
        void *resolved = aimdo_hip_resolve_symbol(dispatch_symbols[i].symbol);

        if (!resolved) {
            log(ERROR, "%s: failed to resolve required HIP symbol %s\n", __func__,
                dispatch_symbols[i].symbol);
            aimdo_cuda_runtime_cleanup();
            return false;
        }
        *dispatch_symbols[i].slot = resolved;
    }

    g_cuda.p_cuMemAllocHost = aimdo_hip_mem_alloc_host;
    g_cuda.p_cuMemHostRegister = g_hip_host_register;
    g_cuda.p_cuMemHostUnregister = g_hip_host_unregister;
    g_cuda.p_cuMemAllocAsync_ptsz = g_cuda.p_cuMemAllocAsync;
    g_cuda.p_cuMemFreeAsync_ptsz = g_cuda.p_cuMemFreeAsync;

    g_device_get_properties = (PFN_deviceGetProperties)aimdo_hip_resolve_symbol("hipGetDevicePropertiesR0600");
    if (!g_device_get_properties) {
        g_device_get_properties = (PFN_deviceGetProperties)aimdo_hip_resolve_symbol("hipGetDeviceProperties");
    }

    {
        CUresult err = g_cuda.p_cuInit(0);

        if (err != CUDA_SUCCESS) {
            const char *desc = NULL;

            if (g_cuda.p_cuGetErrorString) {
                g_cuda.p_cuGetErrorString(err, &desc);
            }
            log(ERROR, "%s: hipInit failed with code %d%s%s\n", __func__, (int)err,
                desc ? ": " : "", desc ? desc : "");
            aimdo_cuda_runtime_cleanup();
            return false;
        }
    }

    return true;
}

void aimdo_cuda_runtime_cleanup(void) {
    memset(&g_cuda, 0, sizeof(g_cuda));
    g_device_get_properties = NULL;
    g_hip_host_malloc = NULL;
    g_hip_host_register = NULL;
    g_hip_host_unregister = NULL;

    if (!g_hip_module) {
        return;
    }

#if !defined(_WIN32) && !defined(_WIN64)
    dlclose(g_hip_module);
#endif
    g_hip_module = NULL;
}
