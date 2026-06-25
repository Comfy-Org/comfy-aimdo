/* Shared hook declarations for the Intel XPU (Level Zero) hook backends.
 *
 * Like src/cuda-hooks-shared.h this is intentionally an includable
 * implementation header, not a normal interface header: it defines static
 * functions and static data and must be included by exactly one backend .c per
 * translation unit (src-xpu/ze-detour.c on Windows, src-posix/ze-funchooks.c on
 * Linux). The including .c must first include "plat.h" and "ze-shim.h".
 *
 * Phase-0 tracing showed torch's XPU device allocations bottom out on
 * zeMemAllocDevice (urUSMDeviceAlloc -> zeMemAllocDevice, 1:1, on the default
 * context). Hooking that single entry point (plus zeMemFree) lets aimdo apply
 * eviction back-pressure and keep its VRAM accounting in sync with torch,
 * exactly like the CUDA path hooks cuMemAlloc_v2 / cuMemFree_v2.
 */
#pragma once

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
 * which case the caller passes through untouched - mirroring the CUDA hook's
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
