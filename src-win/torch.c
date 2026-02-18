#include "plat.h"

#include <windows.h>

#include <windows.h>
#include "plat.h"

// Pointer to the raw C++ function (no GIL needed)
void (*empty_cache)(void) = NULL;

bool torch_init() {
    // 1. Target the actual CUDA backend DLL
    HMODULE hModule = GetModuleHandleA("torch_cuda.dll");
    if (!hModule) hModule = GetModuleHandleA("c10_cuda.dll");

    if (!hModule) {
        log(ERROR, "torch_init: torch_cuda.dll not found. Memory recovery disabled.\n");
        return true;
    }

    // 2. Use the C++ mangled name for the emptyCache routine
    // This is the static call that doesn't touch Python.
    empty_cache = (void*)GetProcAddress(hModule, "?CachingHostAllocator_emptyCache@cuda@at@@YAXXZ");

    if (!empty_cache) {
        log(ERROR, "torch_init: Could not resolve C++ emptyCache symbol.\n");
    } else {
        log(DEBUG, "torch_init: Hooked C++ emptyCache at %p\n", empty_cache);
    }
    
    return true;
}
