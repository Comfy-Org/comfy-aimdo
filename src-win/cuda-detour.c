#include "plat.h"
#include <windows.h>
#include <detours.h>

typedef struct {
    void **true_ptr;
    void *hook_ptr;
    const char *name;
} HookEntry;

/* Driver API Versioned Function Pointers */
static int (*true_cuMemAlloc_v2)(CUdeviceptr*, size_t);
static int (*true_cuMemFree_v2)(CUdeviceptr);
static int (*true_cuMemAllocAsync)(CUdeviceptr*, size_t, CUstream);
static int (*true_cuMemFreeAsync)(CUdeviceptr, CUstream);

static int aimdo_cuMemAlloc_v2(CUdeviceptr* dptr, size_t size) {
    return aimdo_cuda_malloc(dptr, size);
}

static int aimdo_cuMemFree_v2(CUdeviceptr dptr) {
    return aimdo_cuda_free(dptr);
}

static int aimdo_cuMemAllocAsync(CUdeviceptr* dptr, size_t size, CUstream hStream) {
    return aimdo_cuda_malloc_async(dptr, size, hStream, true_cuMemAllocAsync);
}

static int aimdo_cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    return aimdo_cuda_free_async(dptr, hStream, true_cuMemFreeAsync);
}

static const HookEntry hooks[] = {
    { (void**)&true_cuMemAlloc_v2,    aimdo_cuMemAlloc_v2,    "cuMemAlloc_v2"    },
    { (void**)&true_cuMemFree_v2,     aimdo_cuMemFree_v2,     "cuMemFree_v2"     },
    { (void**)&true_cuMemAllocAsync,  aimdo_cuMemAllocAsync,  "cuMemAllocAsync"  },
    { (void**)&true_cuMemFreeAsync,   aimdo_cuMemFreeAsync,   "cuMemFreeAsync"   },
};

static inline bool install_hook_entrys(HMODULE h, HookEntry *hooks, size_t num_hooks) {
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    int status;

    for (int i = 0; i < num_hooks; i++) {
        *hooks[i].true_ptr = (void*)GetProcAddress(h, hooks[i].name);
        if (!*hooks[i].true_ptr ||
            DetourAttach(hooks[i].true_ptr, hooks[i].hook_ptr) != 0) {
            log(ERROR, "%s: Hook %s failed %p", __func__, hooks[i].name, *hooks[i].true_ptr);
            DetourTransactionAbort();
            return false;
        }
    }

    status = (int)DetourTransactionCommit();
    if (status != 0) {
        log(ERROR, "%s: DetourTransactionCommit failed: %d", __func__, status);
        return false;
    }

    log(DEBUG, "%s: hooks successfully installed\n", __func__);
    return true;
}

bool aimdo_setup_hooks() {
    HMODULE h_real_cuda = GetModuleHandleA("nvcuda64.dll");
    if (h_real_cuda == NULL) {
        h_real_cuda = GetModuleHandleA("nvcuda.dll");
    }

    if (h_real_cuda == NULL) {
        log(ERROR, "%s: nvcuda driver not found in process memory", __func__);
        return false;
    }

    log(INFO, "%s: found driver at %p, installing %zu hooks",
        __func__, h_real_cuda, sizeof(hooks) / sizeof(HookEntry));

    return install_hook_entrys(h_real_cuda, (HookEntry*)hooks, sizeof(hooks) / sizeof(HookEntry));
}

void aimdo_teardown_hooks() {
    int status;

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());

    for (int i = 0; i < sizeof(hooks) / sizeof(hooks[0]); i++) {
        /* Only detach if we actually successfully resolved the pointer */
        if (*hooks[i].true_ptr) {
            DetourDetach(hooks[i].true_ptr, hooks[i].hook_ptr);
        }
    }

    status = (int)DetourTransactionCommit();
    if (status != 0) {
        log(ERROR, "%s: DetourDetach failed: %d", __func__, status);
    } else {
        log(DEBUG, "%s: hooks successfully removed\n", __func__);
    }
}
