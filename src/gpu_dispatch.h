#pragma once

#include "gpu_abi.h"

#include <stdbool.h>

typedef CUresult (CUDAAPI *PFN_cuInit)(unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuGetProcAddress)(const char *symbol, void **pfn, int cudaVersion,
                                                 cuuint64_t flags,
                                                 CUdriverProcAddressQueryResult *symbolStatus);
typedef CUresult (CUDAAPI *PFN_cuGetErrorString)(CUresult error, const char **pStr);
typedef CUresult (CUDAAPI *PFN_cuCtxGetDevice)(CUdevice *device);
typedef CUresult (CUDAAPI *PFN_cuCtxSynchronize)(void);
typedef CUresult (CUDAAPI *PFN_cuDeviceGet)(CUdevice *device, int ordinal);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib,
                                                     CUdevice dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceTotalMem)(size_t *bytes, CUdevice dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetName)(char *name, int len, CUdevice dev);
typedef CUresult (CUDAAPI *PFN_cuMemGetInfo)(size_t *free_bytes, size_t *total_bytes);
typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v2)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (CUDAAPI *PFN_cuMemFree_v2)(CUdeviceptr dptr);
typedef CUresult (CUDAAPI *PFN_cuMemAllocAsync)(CUdeviceptr *dptr, size_t bytesize,
                                                CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemFreeAsync)(CUdeviceptr dptr, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemAllocHost)(void **pp, size_t bytesize);
typedef CUresult (CUDAAPI *PFN_cuMemFreeHost)(void *p);
typedef CUresult (CUDAAPI *PFN_cuMemHostRegister)(void *p, size_t bytesize,
                                                 unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuMemHostUnregister)(void *p);
typedef CUresult (CUDAAPI *PFN_cuMemAddressReserve)(CUdeviceptr *ptr, size_t size,
                                                    size_t alignment, CUdeviceptr addr,
                                                    unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemAddressFree)(CUdeviceptr ptr, size_t size);
typedef CUresult (CUDAAPI *PFN_cuMemCreate)(CUmemGenericAllocationHandle *handle, size_t size,
                                            const CUmemAllocationProp *prop,
                                            unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemMap)(CUdeviceptr ptr, size_t size, size_t offset,
                                         CUmemGenericAllocationHandle handle,
                                         unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemSetAccess)(CUdeviceptr ptr, size_t size,
                                               const CUmemAccessDesc *desc, size_t count);
typedef CUresult (CUDAAPI *PFN_cuMemUnmap)(CUdeviceptr ptr, size_t size);
typedef CUresult (CUDAAPI *PFN_cuMemRelease)(CUmemGenericAllocationHandle handle);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoDAsync)(CUdeviceptr dst, const void *src,
                                                  size_t bytes, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetLuid)(char *luid, unsigned int *deviceNodeMask,
                                                CUdevice dev);

typedef struct AimdoCudaDispatch {
    PFN_cuInit p_cuInit;
    PFN_cuGetProcAddress p_cuGetProcAddress;
    PFN_cuGetErrorString p_cuGetErrorString;
    PFN_cuCtxGetDevice p_cuCtxGetDevice;
    PFN_cuCtxSynchronize p_cuCtxSynchronize;
    PFN_cuDeviceGet p_cuDeviceGet;
    PFN_cuDeviceGetAttribute p_cuDeviceGetAttribute;
    PFN_cuDeviceTotalMem p_cuDeviceTotalMem;
    PFN_cuDeviceGetName p_cuDeviceGetName;
    PFN_cuMemGetInfo p_cuMemGetInfo;
    PFN_cuMemAlloc_v2 p_cuMemAlloc_v2;
    PFN_cuMemFree_v2 p_cuMemFree_v2;
    PFN_cuMemAllocAsync p_cuMemAllocAsync;
    PFN_cuMemAllocAsync p_cuMemAllocAsync_ptsz;
    PFN_cuMemFreeAsync p_cuMemFreeAsync;
    PFN_cuMemFreeAsync p_cuMemFreeAsync_ptsz;
    PFN_cuMemAllocHost p_cuMemAllocHost;
    PFN_cuMemFreeHost p_cuMemFreeHost;
    PFN_cuMemHostRegister p_cuMemHostRegister;
    PFN_cuMemHostUnregister p_cuMemHostUnregister;
    PFN_cuMemAddressReserve p_cuMemAddressReserve;
    PFN_cuMemAddressFree p_cuMemAddressFree;
    PFN_cuMemCreate p_cuMemCreate;
    PFN_cuMemMap p_cuMemMap;
    PFN_cuMemSetAccess p_cuMemSetAccess;
    PFN_cuMemUnmap p_cuMemUnmap;
    PFN_cuMemRelease p_cuMemRelease;
    PFN_cuMemcpyHtoDAsync p_cuMemcpyHtoDAsync;
    PFN_cuDeviceGetLuid p_cuDeviceGetLuid;
} AimdoCudaDispatch;

extern AimdoCudaDispatch g_cuda;

typedef CUresult (CUDAAPI *PFN_deviceGetProperties)(void *prop, CUdevice dev);

extern PFN_deviceGetProperties g_device_get_properties;

bool aimdo_cuda_runtime_init(void);
void aimdo_cuda_runtime_cleanup(void);
