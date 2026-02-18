#include "plat.h"

#include <windows.h>

bool torch_init() {
    // 1. Get a handle to the already-loaded DLL
    // PyTorch on Windows uses 'c10_cuda.dll' instead of 'libc10_cuda.so'
    HMODULE hModule = GetModuleHandleA("c10_cuda.dll");
 
    if (hModule == NULL) {
        log(ERROR, "%s: c10_cuda.dll not found in process memory. Running without on-fly GC. This lowers your usable VRAM.\n", __func__);
        return true;
    }

    // 2. Find the symbol
    // WARNING: Windows symbols are decorated differently. 
    // You must check the exports of c10_cuda.dll to get the exact name.
    empty_cache = (void*)GetProcAddress(hModule, "?emptyCache@CUDACachingAllocator@cuda@c10@@SAXXZ");

    if (!empty_cache) {
        log(ERROR, "%s: c10_cuda.dll does not contain emptyCache. Running without on-fly GC. This lowers your usable VRAM.\n", __func__);
    }
    log(DEBUG, "%s: Torch empty_cache function found\n", __func__);
    return true;
}

