#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stddef.h>
#include <exception>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

static inline void check_cu_result(CUresult err, std::string fn){
    if (err == CUresult::CUDA_SUCCESS) {
        return;
    }
    const char* desc;
    if (cuGetErrorString(err, &desc) != CUresult::CUDA_SUCCESS) {
        desc = "<FATAL - CANNOT PARSE CUDA ERROR CODE>";
    }
    std::string err_msg = "CUDA " + fn + " failed: " + desc;
    fprintf(stderr, "%s\n", err_msg.c_str()); //Todo remove
    throw std::exception(err_msg.c_str());
}

__declspec(dllexport)
void *alloc_fn(size_t size, int device, cudaStream_t stream) {
    void *ptr = NULL;
    CUresult err;

    err = cuMemAddressReserve((CUdeviceptr*)&ptr, size, 0, 0, 0);
    check_cu_result(err, "cuMemAddressReserve");

    CUmemGenericAllocationHandle alloc_handle = 0;
    {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        err = cuMemCreate(&alloc_handle, size, &prop, 0);
        check_cu_result(err, "cuMemCreate");
    }

    err = cuMemMap((CUdeviceptr)ptr, size, 0, alloc_handle, 0);
    check_cu_result(err, "cuMemMap");

    {
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = device;
        // Set to Read/Write
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        err = cuMemSetAccess((CUdeviceptr)ptr, size, &accessDesc, 1);
        check_cu_result(err, "cuMemSetAccess");
    }

    printf("FK2 Custom Alloc: ptr=%p, size=%td, device=%d\n", ptr, size, device);
    return ptr;
}

__declspec(dllexport)
void free_fn(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (ptr == NULL) {
        return;
    }

    cudaFree(ptr);

    printf("Custom Free: ptr=%p, size=%td, stream=%p\n", ptr, size, stream);
}

#ifdef __cplusplus
} // End of the extern "C" block
#endif
