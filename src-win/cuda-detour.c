#include "plat.h"
#include <windows.h>
#include <detours.h>

#define TARGET_DLL "cudart64_12.dll"

static int (*true_cuda_malloc)(void**, size_t) = NULL;
static int (*true_cuda_free)(void*) = NULL;
static int (*true_cuda_malloc_async)(void **devPtr, size_t size, void *hStream);
static int (*true_cuda_free_async)(void *devPtr, void *hStream);

typedef struct {
    void **true_ptr;
    void *hook_ptr;
    const char *name;
} HookEntry;

static const HookEntry hooks[] = {
    { (void**)&true_cuda_malloc,       aimdo_cuda_malloc,       "cudaMalloc"       },
    { (void**)&true_cuda_free,         aimdo_cuda_free,         "cudaFree"         },
    { (void**)&true_cuda_malloc_async, aimdo_cuda_malloc_async, "cudaMallocAsync"  },
    { (void**)&true_cuda_free_async,   aimdo_cuda_free_async,   "cudaFreeAsync"    },
};

bool aimdo_setup_hooks() {
    HMODULE h_real_cuda;
    int status;

    h_real_cuda = GetModuleHandleA(TARGET_DLL);
    if (h_real_cuda == NULL) {
        h_real_cuda = LoadLibraryExA(TARGET_DLL, NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
    }

    if (h_real_cuda == NULL) {
        log(ERROR, "%s: %s not found", __func__, TARGET_DLL);
        return false;
    }

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());

    for (int i = 0; i < sizeof(hooks)/sizeof(hooks[0]); i++) {
        *hooks[i].true_ptr = (void*)GetProcAddress(h_real_cuda, hooks[i].name);
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

    log(DEBUG, "%s: hooks successfully installed", __func__);
    return true;
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
        log(DEBUG, "%s: hooks successfully removed", __func__);
    }
}
