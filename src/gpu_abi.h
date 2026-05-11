#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__HIP_PLATFORM_AMD__)
/* HIP runtime exports use the platform C ABI. */
#define CUDAAPI
#elif defined(_WIN32) || defined(_WIN64)
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

typedef unsigned long long cuuint64_t;

#if defined(_WIN64) || defined(__LP64__) || defined(_LP64)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif

typedef int CUdevice;
typedef int CUresult;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef unsigned long long CUmemGenericAllocationHandle;

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
} CUdevice_attribute;

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
} cudaError_enum;

typedef enum CUmemAllocationHandleType_enum {
    CU_MEM_HANDLE_TYPE_NONE = 0x0,
} CUmemAllocationHandleType;

typedef enum CUmemAccess_flags_enum {
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0x0,
    CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3,
} CUmemAccess_flags;

typedef enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0x0,
    CU_MEM_LOCATION_TYPE_DEVICE = 0x1,
    CU_MEM_LOCATION_TYPE_HOST = 0x2,
    CU_MEM_LOCATION_TYPE_HOST_NUMA = 0x3,
    CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 0x4,
} CUmemLocationType;

typedef enum CUmemAllocationType_enum {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,
    CU_MEM_ALLOCATION_TYPE_PINNED = 0x1,
} CUmemAllocationType;

typedef struct CUmemLocation_st {
    CUmemLocationType type;
    int id;
} CUmemLocation;

typedef struct CUmemAllocationProp_st {
    CUmemAllocationType type;
    CUmemAllocationHandleType requestedHandleTypes;
    CUmemLocation location;
    void *win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
} CUmemAllocationProp;

typedef struct CUmemAccessDesc_st {
    CUmemLocation location;
    CUmemAccess_flags flags;
} CUmemAccessDesc;

typedef enum CUdriverProcAddress_flags_enum {
    CU_GET_PROC_ADDRESS_DEFAULT = 0x0,
    CU_GET_PROC_ADDRESS_LEGACY_STREAM = 0x1,
    CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = 0x2,
} CUdriverProcAddress_flags;

typedef enum CUdriverProcAddressQueryResult_enum {
    CU_GET_PROC_ADDRESS_SUCCESS = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2,
} CUdriverProcAddressQueryResult;
