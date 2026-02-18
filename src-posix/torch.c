#define _GNU_SOURCE  // Must be at the very top
#include "plat.h"
#include <dlfcn.h>
#include <stddef.h>

bool torch_init() {
    void* handle = RTLD_DEFAULT;

    // Use the exact signature for the cast to satisfy GCC
    empty_cache = (void(*)(MempoolId_t))dlsym(handle, "_ZN3c104cuda21CUDACachingAllocator10emptyCacheENS0_11MempoolId_tE");

    if (!empty_cache) {
        empty_cache = (void(*)(MempoolId_t))dlsym(handle, "_ZN3c104cuda21CUDACachingAllocator10emptyCacheEv");
    }

    if (!empty_cache) {
        log(ERROR, "torch_init: Could not resolve Linux C++ emptyCache symbol.\n");
    } else {
        log(DEBUG, "torch_init: Hooked Linux C++ emptyCache at %p\n", (void*)empty_cache);
    }

    return true;
}
