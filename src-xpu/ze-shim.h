/* src-xpu/ze-shim.h
 *
 * Private header for the Intel XPU (Level Zero) backend of comfy-aimdo.
 *
 * Upstream's src/gpu_abi.h is now the canonical CUDA-driver ABI used by the
 * portable core (types, handles, the CUmem* descriptor structs, and the enum
 * values CUDA_SUCCESS / CUDA_ERROR_OUT_OF_MEMORY). This header only adds the
 * pieces the Level Zero shim needs on top of that:
 *
 *   - the few CUresult codes gpu_abi.h does not define
 *   - prototypes for the ze_cu* entry points implemented in ze-shim.c, which
 *     src-xpu/dispatch.c wires into the g_cuda dispatch table
 *   - helpers shared with the ze_loader detour (ze-detour.c)
 *
 * The cu* names themselves are macros in plat.h that expand to g_cuda.p_cuXxx,
 * so the implementations MUST use the distinct ze_cu* names to avoid being
 * rewritten by the preprocessor.
 */
#pragma once

#include "gpu_abi.h"

#include <level_zero/ze_api.h>

/* CUresult codes used by the shim that gpu_abi.h's cudaError_enum omits. These
 * are plain integers (not enumerators) so they cannot collide with the enum. */
#ifndef CUDA_ERROR_INVALID_VALUE
#define CUDA_ERROR_INVALID_VALUE 1
#endif
#ifndef CUDA_ERROR_NOT_INITIALIZED
#define CUDA_ERROR_NOT_INITIALIZED 3
#endif
#ifndef CUDA_ERROR_UNKNOWN
#define CUDA_ERROR_UNKNOWN 999
#endif

/* ---- Dispatch entry points (implemented in ze-shim.c) ------------------ */

CUresult CUDAAPI ze_cuInit(unsigned int flags);
CUresult CUDAAPI ze_cuGetErrorString(CUresult error, const char **pStr);

/* Device / context */
CUresult CUDAAPI ze_cuDeviceGet(CUdevice *device, int ordinal);
CUresult CUDAAPI ze_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult CUDAAPI ze_cuDeviceTotalMem(size_t *bytes, CUdevice dev);
CUresult CUDAAPI ze_cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult CUDAAPI ze_cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev);
CUresult CUDAAPI ze_cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult CUDAAPI ze_cuCtxGetDevice(CUdevice *device);
CUresult CUDAAPI ze_cuCtxSetCurrent(CUcontext ctx);
CUresult CUDAAPI ze_cuCtxSynchronize(void);

/* Virtual Memory Management */
CUresult CUDAAPI ze_cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                                        CUdeviceptr addr, unsigned long long flags);
CUresult CUDAAPI ze_cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult CUDAAPI ze_cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                                const CUmemAllocationProp *prop, unsigned long long flags);
CUresult CUDAAPI ze_cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult CUDAAPI ze_cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                             CUmemGenericAllocationHandle handle, unsigned long long flags);
CUresult CUDAAPI ze_cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult CUDAAPI ze_cuMemSetAccess(CUdeviceptr ptr, size_t size,
                                   const CUmemAccessDesc *desc, size_t count);

/* Misc memory */
CUresult CUDAAPI ze_cuMemGetInfo(size_t *free, size_t *total);
CUresult CUDAAPI ze_cuMemAllocHost(void **pp, size_t size);
CUresult CUDAAPI ze_cuMemFreeHost(void *p);
CUresult CUDAAPI ze_cuMemHostRegister(void *p, size_t size, unsigned int flags);
CUresult CUDAAPI ze_cuMemHostUnregister(void *p);

/* Async H2D copy + events (blocking implementations; see ze-shim.c) */
CUresult CUDAAPI ze_cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, size_t size,
                                      CUstream hStream);
CUresult CUDAAPI ze_cuEventCreate(CUevent *phEvent, unsigned int flags);
CUresult CUDAAPI ze_cuEventDestroy(CUevent hEvent);
CUresult CUDAAPI ze_cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult CUDAAPI ze_cuEventSynchronize(CUevent hEvent);

/* Fail-loud stubs for entry points only the CUDA/HIP detour uses (the XPU build
 * intercepts torch allocations via the ze_loader detour instead, so the core
 * never calls these — but we register them so the table has no NULL slots). */
CUresult CUDAAPI ze_cuMemAlloc_v2(CUdeviceptr *dptr, size_t size);
CUresult CUDAAPI ze_cuMemFree_v2(CUdeviceptr dptr);
CUresult CUDAAPI ze_cuMemAllocAsync(CUdeviceptr *dptr, size_t size, CUstream hStream);
CUresult CUDAAPI ze_cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);

/* ---- Helpers shared with the ze_loader detour (ze-detour.c) ------------ */

/* aimdo-managed Level Zero device ordinal, or -1 if not initialized. */
int aimdo_ze_managed_ordinal(void);
ze_context_handle_t aimdo_ze_context(void);
ze_device_handle_t  aimdo_ze_device(void);
