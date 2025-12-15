#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stddef.h>
#include <exception> // New: Include C++ exception header
#include <string>    // New: For building the error message

#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllexport)
void *alloc_fn(size_t size, int device, cudaStream_t stream) {
    void *ptr = NULL;
    cudaError_t err;

    err = cudaMalloc(&ptr, size);

    if (err != cudaSuccess) {
        // Build an error message string
        std::string err_msg = "CUDA C++ Malloc Failed: ";
        err_msg += cudaGetErrorString(err);

        // 1. Use standard C error output (kept per request)
        fprintf(stderr, "%s\n", err_msg.c_str());

        // 2. Throw std::exception instead of returning NULL
        throw std::exception(err_msg.c_str());
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