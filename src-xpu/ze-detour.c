/* src-xpu/ze-detour.c
 *
 * Windows (Detours) interception layer for the Intel XPU build, mirroring
 * src-win/cuda-detour.c but hooking the Level Zero loader instead of nvcuda.
 *
 * Phase-0 tracing showed torch's XPU device allocations bottom out on
 * zeMemAllocDevice (urUSMDeviceAlloc -> zeMemAllocDevice, 1:1, on the default
 * context). Hooking that single entry point lets aimdo apply eviction
 * back-pressure and keep its VRAM accounting in sync with torch, exactly like
 * the CUDA path hooks cuMemAlloc_v2 / cuMemFree_v2.
 */
#include "plat.h"
#include "ze-shim.h"

#include <windows.h>
#include <detours.h>

/* cudaMalloc reserves extra headroom under pressure; mirror that here since the
 * XPU device allocator (like cudaMalloc) does not defragment for us. */
#define XPU_ALLOC_HEADROOM (128 * M)

typedef struct {
    void       **true_ptr;
    void        *hook_ptr;
    const char  *name;
} HookEntry;

static ze_result_t (*true_zeMemAllocDevice)(ze_context_handle_t,
                                            const ze_device_mem_alloc_desc_t *,
                                            size_t, size_t,
                                            ze_device_handle_t, void **);
static ze_result_t (*true_zeMemFree)(ze_context_handle_t, void *);

/* Select the aimdo device context for the managed XPU device. The eviction and
 * accounting macros (vbars_free / aimdo_account_*) dereference the thread-local
 * g_devctx, so this MUST succeed before we touch them; otherwise we'd account
 * against a NULL/wrong context. Single managed device only for now. Returns
 * false when aimdo isn't managing a device yet (allocation predates init), in
 * which case the caller passes through untouched — mirroring the CUDA hook's
 * set_devctx_for_current_cuda_device() guard. */
static bool select_managed_devctx(void) {
    int ord = aimdo_ze_managed_ordinal();
    return ord >= 0 && set_devctx_for_device(ord);
}

static ze_result_t aimdo_zeMemAllocDevice(ze_context_handle_t hContext,
                                          const ze_device_mem_alloc_desc_t *device_desc,
                                          size_t size, size_t alignment,
                                          ze_device_handle_t hDevice, void **pptr) {
    ze_result_t status;

    if (!pptr || !true_zeMemAllocDevice) {
        return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (!select_managed_devctx()) {
        /* not our device at all - straight passthrough */
        return true_zeMemAllocDevice(hContext, device_desc, size, alignment, hDevice, pptr);
    }

    vbars_free(budget_deficit(size + XPU_ALLOC_HEADROOM));

    status = true_zeMemAllocDevice(hContext, device_desc, size, alignment, hDevice, pptr);
    if (status != ZE_RESULT_SUCCESS) {
        /* Out of room: evict harder and retry once, like aimdo_cuda_malloc. */
        vbars_free(size + XPU_ALLOC_HEADROOM);
        status = true_zeMemAllocDevice(hContext, device_desc, size, alignment, hDevice, pptr);
    }

    if (status == ZE_RESULT_SUCCESS) {
        aimdo_account_alloc((CUdeviceptr)(uintptr_t)*pptr, size);
    }
    return status;
}

static ze_result_t aimdo_zeMemFree(ze_context_handle_t hContext, void *ptr) {
    ze_result_t status;

    if (!ptr) {
        return ZE_RESULT_SUCCESS;
    }
    if (!true_zeMemFree) {
        return ZE_RESULT_ERROR_UNINITIALIZED;
    }

    /* zeMemFree carries no device handle; with single-device support the managed
     * context is unambiguous. If aimdo isn't managing a device, pass through. */
    if (!select_managed_devctx()) {
        return true_zeMemFree(hContext, ptr);
    }

    status = true_zeMemFree(hContext, ptr);
    if (status == ZE_RESULT_SUCCESS) {
        aimdo_account_free((CUdeviceptr)(uintptr_t)ptr);
    }
    return status;
}

static const HookEntry hooks[] = {
    { (void **)&true_zeMemAllocDevice, aimdo_zeMemAllocDevice, "zeMemAllocDevice" },
    { (void **)&true_zeMemFree,        aimdo_zeMemFree,        "zeMemFree"        },
};

static inline bool install_hook_entrys(HMODULE h, const HookEntry *entries, size_t num) {
    int status;

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());

    for (size_t i = 0; i < num; i++) {
        *entries[i].true_ptr = (void *)GetProcAddress(h, entries[i].name);
        if (!*entries[i].true_ptr ||
            DetourAttach(entries[i].true_ptr, entries[i].hook_ptr) != 0) {
            log(ERROR, "%s: Hook %s failed %p", __func__, entries[i].name, *entries[i].true_ptr);
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
    HMODULE h_ze = GetModuleHandleA("ze_loader.dll");
    if (h_ze == NULL) {
        h_ze = GetModuleHandleA("ze_loader");
    }
    if (h_ze == NULL) {
        log(ERROR, "%s: ze_loader not found in process memory\n", __func__);
        return false;
    }

    log(INFO, "%s: found ze_loader at %p, installing %zu hooks\n",
        __func__, h_ze, sizeof(hooks) / sizeof(HookEntry));

    return install_hook_entrys(h_ze, hooks, sizeof(hooks) / sizeof(HookEntry));
}

void aimdo_teardown_hooks() {
    int status;

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());

    for (size_t i = 0; i < sizeof(hooks) / sizeof(hooks[0]); i++) {
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
