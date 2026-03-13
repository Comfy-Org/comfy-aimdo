#include "../src/plat.h"
#include <dlfcn.h>
#include <stdbool.h>


CUresult (*hipMallocAsyncOriginal)(CUdeviceptr*, size_t, CUstream);
CUresult (*hipMallocOriginal)(CUdeviceptr*, size_t);
CUresult (*hipFreeAsyncOriginal)(CUdeviceptr*, CUstream);
CUresult (*hipFreeOriginal)(CUdeviceptr*);

CUresult cuMemAlloc_v2(CUdeviceptr* ptr, size_t size) {
	return hipMallocOriginal(ptr, size);
}

CUresult cuMemFree_v2(CUdeviceptr ptr) {
	return hipFreeOriginal(ptr);
}

// Provide these as stubs to call the original allocation functions
CUresult cuMemAllocAsync(CUdeviceptr* ptr, size_t size, CUstream h) {
	return hipMallocAsyncOriginal(ptr, size, h);
};

CUresult cuMemFreeAsync(CUdeviceptr ptr, CUstream h) {
	return hipFreeAsyncOriginal(ptr, h);
}

bool aimdo_setup_hooks() {
	log(VERBOSE, "loading libamdhip64.so\n");
	void* handle = dlopen("libamdhip64.so", RTLD_LAZY|RTLD_LOCAL);
	if (!handle) return false;
	hipMallocAsyncOriginal = dlsym(handle, "hipMallocAsync");
	hipFreeAsyncOriginal = dlsym(handle, "hipFreeAsync");
	hipMallocOriginal = dlsym(handle, "hipMalloc");
	hipFreeOriginal = dlsym(handle, "hipFree");
	return true;
}

void aimdo_teardown_hooks() {
	printf("No teardown\n");
};

