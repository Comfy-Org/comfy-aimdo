#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllexport)
void *alloc_fn(size_t size, int device, cudaStream_t stream) {
    void *ptr = NULL;
    cudaError_t err;

    err = cudaMalloc(&ptr, size);

    if (err != cudaSuccess) {
        // Use standard C error output
        fprintf(stderr, "CUDA Malloc Failed: %s\n", cudaGetErrorString(err));
        return NULL; // Return NULL on failure as per allocator contract
    }

    printf("Custom Alloc: ptr=%p, size=%td, device=%d\n", ptr, size, device);
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