/* src-xpu/dispatch.c
 *
 * Intel XPU (Level Zero) backend's implementation of the GPU dispatch contract
 * (src/gpu_dispatch.h). Unlike the CUDA/HIP backends, which resolve a vendor
 * driver module via cuGetProcAddress/dlsym, the XPU backend's entry points are
 * the ze_cu* functions in ze-shim.c — so aimdo_cuda_runtime_init() simply wires
 * them straight into the g_cuda table (no module discovery), then runs zeInit
 * via ze_cuInit.
 */
#include "plat.h"
#include "ze-shim.h"

AimdoCudaDispatch g_cuda;
PFN_deviceGetProperties g_device_get_properties;

bool aimdo_cuda_runtime_init(void) {
    if (g_cuda.p_cuInit) {
        return true;
    }

    g_cuda.p_cuInit              = ze_cuInit;
    g_cuda.p_cuGetErrorString    = ze_cuGetErrorString;
    g_cuda.p_cuCtxGetDevice      = ze_cuCtxGetDevice;
    g_cuda.p_cuCtxSynchronize    = ze_cuCtxSynchronize;
    g_cuda.p_cuDeviceGet         = ze_cuDeviceGet;
    g_cuda.p_cuDeviceGetAttribute = ze_cuDeviceGetAttribute;
    g_cuda.p_cuDeviceTotalMem    = ze_cuDeviceTotalMem;
    g_cuda.p_cuDeviceGetName     = ze_cuDeviceGetName;
    g_cuda.p_cuDeviceGetLuid     = ze_cuDeviceGetLuid;
    g_cuda.p_cuMemGetInfo        = ze_cuMemGetInfo;
    g_cuda.p_cuMemAllocHost      = ze_cuMemAllocHost;
    g_cuda.p_cuMemFreeHost       = ze_cuMemFreeHost;
    g_cuda.p_cuMemHostRegister   = ze_cuMemHostRegister;
    g_cuda.p_cuMemHostUnregister = ze_cuMemHostUnregister;
    g_cuda.p_cuMemAddressReserve = ze_cuMemAddressReserve;
    g_cuda.p_cuMemAddressFree    = ze_cuMemAddressFree;
    g_cuda.p_cuMemCreate         = ze_cuMemCreate;
    g_cuda.p_cuMemMap            = ze_cuMemMap;
    g_cuda.p_cuMemSetAccess      = ze_cuMemSetAccess;
    g_cuda.p_cuMemUnmap          = ze_cuMemUnmap;
    g_cuda.p_cuMemRelease        = ze_cuMemRelease;
    g_cuda.p_cuMemcpyHtoDAsync   = ze_cuMemcpyHtoDAsync;
    g_cuda.p_cuEventCreate       = ze_cuEventCreate;
    g_cuda.p_cuEventDestroy      = ze_cuEventDestroy;
    g_cuda.p_cuEventRecord       = ze_cuEventRecord;
    g_cuda.p_cuEventSynchronize  = ze_cuEventSynchronize;

    /* Entry points only the CUDA/HIP allocator detour uses; the XPU build hooks
     * torch allocations through the ze_loader detour instead. Wire fail-loud
     * stubs so the table never has NULL slots. */
    g_cuda.p_cuMemAlloc_v2        = ze_cuMemAlloc_v2;
    g_cuda.p_cuMemFree_v2         = ze_cuMemFree_v2;
    g_cuda.p_cuMemAllocAsync      = ze_cuMemAllocAsync;
    g_cuda.p_cuMemAllocAsync_ptsz = ze_cuMemAllocAsync;
    g_cuda.p_cuMemFreeAsync       = ze_cuMemFreeAsync;
    g_cuda.p_cuMemFreeAsync_ptsz  = ze_cuMemFreeAsync;

    if (g_cuda.p_cuInit(0) != CUDA_SUCCESS) {
        log(ERROR, "%s: zeInit (via ze_cuInit) failed\n", __func__);
        aimdo_cuda_runtime_cleanup();
        return false;
    }

    return true;
}

void aimdo_cuda_runtime_cleanup(void) {
    memset(&g_cuda, 0, sizeof(g_cuda));
    g_device_get_properties = NULL;
}
