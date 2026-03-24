#include <stdbool.h>
bool aimdo_setup_hooks();
void aimdo_teardown_hooks();

// Map CUDA functions and constants to HIP equivalents
#define CUDA_ERROR_OUT_OF_MEMORY hipErrorOutOfMemory
#define CUDA_SUCCESS hipSuccess
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
// hipCtx* are deprecated; cuCtx* functions are too, to my knowledge
// fortunately they have the same interface
#define cuCtxGetDevice hipGetDevice
#define cuCtxSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaFreeAsync hipFreeAsync
#define cudaFree hipFree
#define cudaMallocAsync hipMallocAsync
#define cudaMalloc hipMalloc
#define cudaStream_t hipStream_t
#define cuDeviceGet hipDeviceGet
#define cuDeviceGetName hipDeviceGetName
#define CUdevice hipDevice_t
#define CUdeviceptr hipDeviceptr_t
#define cuDeviceTotalMem hipDeviceTotalMem
#define cuGetErrorString hipDrvGetErrorString
#define CUmemAccessDesc hipMemAccessDesc
#define cuMemAddressFree hipMemAddressFree
#define cuMemAddressReserve hipMemAddressReserve
#define CUmemAllocationProp hipMemAllocationProp
#define cuMemCreate hipMemCreate
#define CUmemGenericAllocationHandle hipMemGenericAllocationHandle_t
#define cuMemGetInfo hipMemGetInfo
#define cuMemMap hipMemMap
#define cuMemRelease hipMemRelease
#define cuMemSetAccess hipMemSetAccess
#define cuMemUnmap hipMemUnmap
#define CUresult hipError_t
#define CUstream hipStream_t
#define CUcontext hipCtx_t
#define cuCtxGetCurrent hipCtxGetCurrent
#define cuCtxSetCurrent hipCtxSetCurrent
#define cuDevicePrimaryCtxRetain hipDevicePrimaryCtxRetain
#define cuMemAllocHost(p, size) hipHostMalloc((p), (size), 0)
#define cuMemFreeHost hipHostFree

CUresult cuMemFreeAsync(CUdeviceptr, CUstream);
CUresult cuMemFree_v2(CUdeviceptr);
CUresult cuMemAllocAsync(CUdeviceptr*, size_t, CUstream);
CUresult cuMemAlloc_v2(CUdeviceptr*, size_t);

// On Windows we can define these without problems. On Linux this will cause recursion in
// the implementations.
#if defined(_WIN32) || defined(_WIN64)
#define cuMemFreeAsync hipFreeAsync
#define cuMemAllocAsync hipMallocAsync
#define unmap_workaround(va, size)
#else
#include <sys/mman.h>
// Workaround for https://github.com/ROCm/rocm-systems/
// On systems where it's fixed, calling mmap twice is harmless.
#define unmap_workaround(va, size) mmap((va), (size), PROT_NONE, MAP_PRIVATE | MAP_FIXED | MAP_NORESERVE | MAP_ANONYMOUS, -1, 0)
#endif
