/* src-xpu/ze-shim.c
 *
 * Implements the CUDA-driver-API subset that comfy-aimdo's portable core calls
 * (through the g_cuda dispatch table) on top of Intel oneAPI Level Zero
 * (ze_loader). This is the XPU backend; src-xpu/dispatch.c wires the ze_cu*
 * functions below into g_cuda, and the cu* names used by the core are plat.h
 * macros expanding to g_cuda.p_cuXxx.
 *
 * Key invariant (validated in Phase 0 on an Arc B570): we operate on the
 * driver's *default context* so the VMM-mapped device pointers we hand out are
 * the same ones torch (which also uses the default context) consumes via DLPack.
 * ze_cuDevicePrimaryCtxRetain therefore returns the default context.
 *
 * Async semantics: ze_cuMemcpyHtoDAsync is implemented as a *blocking* copy on a
 * synchronous immediate command list, and the cuEvent* entry points are tiny
 * host objects that are always "complete". This is correct because the copy has
 * fully landed before the call returns, so the hostbuf ring-slot reuse and the
 * vbar page-in consumers never observe stale data. True stream overlap with
 * torch's SYCL queue is a later optimization (needs SYCL/UR interop).
 */
#include <level_zero/ze_api.h>

#include "plat.h"
#include "thread-plat.h"
#include "ze-shim.h"

/* ---- Shim global state ------------------------------------------------- */

static ze_driver_handle_t  g_driver;
static ze_device_handle_t  g_device;
static ze_context_handle_t g_context;
static int                 g_inited;
static int                 g_managed_ordinal = -1;

/* Lazily-created synchronous immediate command list for blocking H2D copies,
 * protected by g_copy_lock (the copy path may be entered from xfer worker
 * callers as well as the main thread). */
static ze_command_list_handle_t g_copy_cmdlist;
static Mutex                    g_copy_lock;

static CUresult ze2cu(ze_result_t r) {
    switch (r) {
    case ZE_RESULT_SUCCESS:
        return CUDA_SUCCESS;
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
        return CUDA_ERROR_OUT_OF_MEMORY;
    case ZE_RESULT_ERROR_UNINITIALIZED:
        return CUDA_ERROR_NOT_INITIALIZED;
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
    case ZE_RESULT_ERROR_INVALID_SIZE:
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
        return CUDA_ERROR_INVALID_VALUE;
    default:
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CUDAAPI ze_cuInit(unsigned int flags) {
    ze_result_t r;

    (void)flags;
    if (g_inited) {
        return CUDA_SUCCESS;
    }
    r = zeInit(0);
    if (r != ZE_RESULT_SUCCESS) {
        log(ERROR, "%s: zeInit failed (0x%x)\n", __func__, r);
        return ze2cu(r);
    }
    g_inited = 1;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuGetErrorString(CUresult error, const char **pStr) {
    static const char *unknown = "<unknown CUDA(->ZE) error>";
    if (!pStr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    switch (error) {
    case CUDA_SUCCESS:               *pStr = "CUDA_SUCCESS"; break;
    case CUDA_ERROR_INVALID_VALUE:   *pStr = "CUDA_ERROR_INVALID_VALUE"; break;
    case CUDA_ERROR_OUT_OF_MEMORY:   *pStr = "CUDA_ERROR_OUT_OF_MEMORY"; break;
    case CUDA_ERROR_NOT_INITIALIZED: *pStr = "CUDA_ERROR_NOT_INITIALIZED"; break;
    default:                         *pStr = unknown; break;
    }
    return CUDA_SUCCESS;
}

/* ---- Device / init ----------------------------------------------------- */

/* Enumerate all Level Zero GPU devices across all drivers and select the
 * ordinal-th one, recording its owning driver and the driver default context. */
CUresult CUDAAPI ze_cuDeviceGet(CUdevice *device, int ordinal) {
    ze_result_t r;
    uint32_t driver_count = 0;
    ze_driver_handle_t *drivers = NULL;
    int seen = 0;

    if (!device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!g_inited) {
        CUresult cr = ze_cuInit(0);
        if (cr != CUDA_SUCCESS) {
            return cr;
        }
    }

    if ((r = zeDriverGet(&driver_count, NULL)) != ZE_RESULT_SUCCESS || driver_count == 0) {
        log(ERROR, "%s: zeDriverGet found no drivers (0x%x)\n", __func__, r);
        return ze2cu(r);
    }
    drivers = (ze_driver_handle_t *)calloc(driver_count, sizeof(*drivers));
    if (!drivers) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if ((r = zeDriverGet(&driver_count, drivers)) != ZE_RESULT_SUCCESS) {
        free(drivers);
        return ze2cu(r);
    }

    for (uint32_t d = 0; d < driver_count; d++) {
        uint32_t dev_count = 0;
        ze_device_handle_t *devs;

        if (zeDeviceGet(drivers[d], &dev_count, NULL) != ZE_RESULT_SUCCESS || dev_count == 0) {
            continue;
        }
        devs = (ze_device_handle_t *)calloc(dev_count, sizeof(*devs));
        if (!devs) {
            free(drivers);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        if (zeDeviceGet(drivers[d], &dev_count, devs) != ZE_RESULT_SUCCESS) {
            free(devs);
            continue;
        }

        for (uint32_t i = 0; i < dev_count; i++) {
            ze_device_properties_t props = { ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES };
            if (zeDeviceGetProperties(devs[i], &props) != ZE_RESULT_SUCCESS) {
                continue;
            }
            if (props.type != ZE_DEVICE_TYPE_GPU) {
                continue;
            }
            if (seen == ordinal) {
                g_driver = drivers[d];
                g_device = devs[i];
                free(devs);
                free(drivers);
                goto found;
            }
            seen++;
        }
        free(devs);
    }

    free(drivers);
    log(ERROR, "%s: no Level Zero GPU device at ordinal %d (found %d)\n", __func__, ordinal, seen);
    return CUDA_ERROR_INVALID_VALUE;

found:
    /* Single-device only for now: the shim keeps one global driver/device/
     * context. Refuse a second, different managed device rather than silently
     * operating on the wrong one. */
    if (g_managed_ordinal != -1 && g_managed_ordinal != ordinal) {
        log(ERROR, "%s: multi-device XPU not supported (already managing ordinal %d, "
                   "asked for %d)\n", __func__, g_managed_ordinal, ordinal);
        return CUDA_ERROR_INVALID_VALUE;
    }

    /* Use the driver's default context so our VMM pointers are the same ones
     * torch operates on. */
    g_context = zeDriverGetDefaultContext(g_driver);
    if (!g_context) {
        ze_context_desc_t cdesc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC };
        log(WARNING, "%s: no default context; creating a private one "
                     "(torch interop via DLPack may not work)\n", __func__);
        if ((r = zeContextCreate(g_driver, &cdesc, &g_context)) != ZE_RESULT_SUCCESS) {
            return ze2cu(r);
        }
    }

    g_managed_ordinal = ordinal;
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    ze_device_properties_t props = { ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES };
    ze_result_t r;

    (void)dev;
    if (!pi || !g_device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if ((r = zeDeviceGetProperties(g_device, &props)) != ZE_RESULT_SUCCESS) {
        return ze2cu(r);
    }
    switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_INTEGRATED:
        *pi = (props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) ? 1 : 0;
        return CUDA_SUCCESS;
    default:
        *pi = 0;
        return CUDA_SUCCESS;
    }
}

CUresult CUDAAPI ze_cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    ze_result_t r;
    uint32_t count = 0;
    ze_device_memory_properties_t *props;
    uint64_t total = 0;

    (void)dev;
    if (!bytes || !g_device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if ((r = zeDeviceGetMemoryProperties(g_device, &count, NULL)) != ZE_RESULT_SUCCESS || !count) {
        return ze2cu(r);
    }
    props = (ze_device_memory_properties_t *)calloc(count, sizeof(*props));
    if (!props) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    for (uint32_t i = 0; i < count; i++) {
        props[i].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    }
    if ((r = zeDeviceGetMemoryProperties(g_device, &count, props)) != ZE_RESULT_SUCCESS) {
        free(props);
        return ze2cu(r);
    }
    for (uint32_t i = 0; i < count; i++) {
        total += props[i].totalSize;
    }
    free(props);
    *bytes = (size_t)total;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    (void)dev;
    if (!pctx) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!g_context) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    *pctx = (CUcontext)g_context;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuCtxGetDevice(CUdevice *device) {
    if (!device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (g_managed_ordinal < 0) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    *device = g_managed_ordinal;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuCtxSetCurrent(CUcontext ctx) {
    /* Level Zero has no thread-current-context concept; contexts are explicit. */
    (void)ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuDeviceGetName(char *name, int len, CUdevice dev) {
    ze_device_properties_t props = { ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES };
    ze_result_t r;

    (void)dev;
    if (!name || len <= 0 || !g_device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if ((r = zeDeviceGetProperties(g_device, &props)) != ZE_RESULT_SUCCESS) {
        return ze2cu(r);
    }
    strncpy(name, props.name, (size_t)len - 1);
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev) {
    ze_device_luid_ext_properties_t luid_props = { ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES };
    ze_device_properties_t props = { ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, &luid_props };
    ze_result_t r;

    (void)dev;
    if (!luid || !g_device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if ((r = zeDeviceGetProperties(g_device, &props)) != ZE_RESULT_SUCCESS) {
        return ze2cu(r);
    }
    /* CUDA LUID is 8 bytes; Level Zero ZE_MAX_DEVICE_LUID_SIZE_EXT == 8. */
    memcpy(luid, luid_props.luid.id, sizeof(luid_props.luid.id));
    if (deviceNodeMask) {
        *deviceNodeMask = luid_props.nodeMask;
    }
    return CUDA_SUCCESS;
}

/* ---- Virtual Memory Management ----------------------------------------- */

CUresult CUDAAPI ze_cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                                        CUdeviceptr addr, unsigned long long flags) {
    void *p = NULL;
    ze_result_t r;

    (void)alignment;
    (void)flags;
    if (!ptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    r = zeVirtualMemReserve(g_context, (const void *)(uintptr_t)addr, size, &p);
    if (r != ZE_RESULT_SUCCESS) {
        return ze2cu(r);
    }
    *ptr = (CUdeviceptr)(uintptr_t)p;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    return ze2cu(zeVirtualMemFree(g_context, (const void *)(uintptr_t)ptr, size));
}

CUresult CUDAAPI ze_cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                                const CUmemAllocationProp *prop, unsigned long long flags) {
    ze_physical_mem_handle_t h = NULL;
    ze_physical_mem_desc_t desc = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC };
    size_t pagesize = 0;
    ze_result_t r;

    (void)prop;
    (void)flags;
    if (!handle) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    /* Physical allocation size must be aligned to the device page granularity. */
    if (zeVirtualMemQueryPageSize(g_context, g_device, size, &pagesize) == ZE_RESULT_SUCCESS && pagesize) {
        size = ALIGN_UP(size, pagesize);
    }
    desc.size = size;
    r = zePhysicalMemCreate(g_context, g_device, &desc, &h);
    if (r != ZE_RESULT_SUCCESS) {
        return ze2cu(r);
    }
    *handle = (CUmemGenericAllocationHandle)(uintptr_t)h;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuMemRelease(CUmemGenericAllocationHandle handle) {
    return ze2cu(zePhysicalMemDestroy(g_context, (ze_physical_mem_handle_t)(uintptr_t)handle));
}

CUresult CUDAAPI ze_cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                             CUmemGenericAllocationHandle handle, unsigned long long flags) {
    (void)flags;
    return ze2cu(zeVirtualMemMap(g_context, (const void *)(uintptr_t)ptr, size,
                                 (ze_physical_mem_handle_t)(uintptr_t)handle, offset,
                                 ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE));
}

CUresult CUDAAPI ze_cuMemUnmap(CUdeviceptr ptr, size_t size) {
    return ze2cu(zeVirtualMemUnmap(g_context, (const void *)(uintptr_t)ptr, size));
}

CUresult CUDAAPI ze_cuMemSetAccess(CUdeviceptr ptr, size_t size,
                                   const CUmemAccessDesc *desc, size_t count) {
    /* aimdo always grants a single READWRITE descriptor for the device. */
    (void)desc;
    (void)count;
    return ze2cu(zeVirtualMemSetAccessAttribute(g_context, (const void *)(uintptr_t)ptr, size,
                                                ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE));
}

/* ---- Misc memory ------------------------------------------------------- */

CUresult CUDAAPI ze_cuMemGetInfo(size_t *free, size_t *total) {
    size_t tot = 0;
    CUresult cr = ze_cuDeviceTotalMem(&tot, 0);

    if (cr != CUDA_SUCCESS) {
        return cr;
    }
    if (total) {
        *total = tot;
    }
    if (free) {
        /* Core Level Zero has no free-memory query (that lives in sysman).
         * Reporting free == total keeps the secondary cuMemGetInfo-based deficit
         * method inert; on Windows the WDDM budget poll in shmem-detect.c is the
         * authoritative pressure signal. */
        *free = tot;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuMemAllocHost(void **pp, size_t size) {
    ze_host_mem_alloc_desc_t hdesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };
    if (!pp) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return ze2cu(zeMemAllocHost(g_context, &hdesc, size, 4096, pp));
}

CUresult CUDAAPI ze_cuMemFreeHost(void *p) {
    return ze2cu(zeMemFree(g_context, p));
}

/* Level Zero has no equivalent of pinning arbitrary, already-allocated OS pages
 * (cuMemHostRegister). The hostbuf allocator hands us ordinary VirtualAlloc
 * memory; zeCommandListAppendMemoryCopy can read it directly (staging happens in
 * the driver if needed), so registration is a no-op success. */
CUresult CUDAAPI ze_cuMemHostRegister(void *p, size_t size, unsigned int flags) {
    (void)p;
    (void)size;
    (void)flags;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuMemHostUnregister(void *p) {
    (void)p;
    return CUDA_SUCCESS;
}

/* ---- Context ----------------------------------------------------------- */

CUresult CUDAAPI ze_cuCtxSynchronize(void) {
    if (!g_device) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    /* zeDeviceSynchronize waits for all work submitted to the device (across all
     * queues/command lists, including torch's), the semantic match for
     * cuCtxSynchronize. The Intel driver returns UNSUPPORTED_FEATURE for
     * zeContextSystemBarrier, so we use the device sync. */
    return ze2cu(zeDeviceSynchronize(g_device));
}

/* ---- Blocking H2D copy + events ---------------------------------------- */

/* Lazily create one synchronous immediate command list for blocking copies.
 * SYNCHRONOUS mode means zeCommandListAppendMemoryCopy does not return until the
 * copy has completed, which gives us cuMemcpyHtoDAsync-with-an-implicit-sync. */
static CUresult ensure_copy_cmdlist(void) {
    ze_command_queue_group_properties_t *groups = NULL;
    uint32_t group_count = 0;
    uint32_t ordinal = 0;
    ze_command_queue_desc_t qdesc = { ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC };
    ze_result_t r;

    if (g_copy_cmdlist) {
        return CUDA_SUCCESS;
    }
    if (!g_context || !g_device) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    /* Prefer a dedicated copy engine; fall back to a compute group (which can
     * also do memory copies) or ordinal 0. */
    if (zeDeviceGetCommandQueueGroupProperties(g_device, &group_count, NULL) == ZE_RESULT_SUCCESS
        && group_count) {
        groups = (ze_command_queue_group_properties_t *)calloc(group_count, sizeof(*groups));
        if (groups) {
            for (uint32_t i = 0; i < group_count; i++) {
                groups[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
            }
            if (zeDeviceGetCommandQueueGroupProperties(g_device, &group_count, groups)
                == ZE_RESULT_SUCCESS) {
                uint32_t compute_ord = UINT32_MAX, copy_ord = UINT32_MAX;
                for (uint32_t i = 0; i < group_count; i++) {
                    if ((groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)
                        && copy_ord == UINT32_MAX) {
                        copy_ord = i;
                    }
                    if ((groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)
                        && compute_ord == UINT32_MAX) {
                        compute_ord = i;
                    }
                }
                if (copy_ord != UINT32_MAX) {
                    ordinal = copy_ord;
                } else if (compute_ord != UINT32_MAX) {
                    ordinal = compute_ord;
                }
            }
            free(groups);
        }
    }

    qdesc.ordinal = ordinal;
    qdesc.index = 0;
    qdesc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    r = zeCommandListCreateImmediate(g_context, g_device, &qdesc, &g_copy_cmdlist);
    if (r != ZE_RESULT_SUCCESS) {
        log(ERROR, "%s: zeCommandListCreateImmediate failed (0x%x)\n", __func__, r);
        return ze2cu(r);
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, size_t size,
                                      CUstream hStream) {
    CUresult cr;
    ze_result_t r;

    /* The torch SYCL stream handle is not a Level Zero queue, so we ignore it
     * and run a blocking copy on our own synchronous immediate list. Correctness
     * comes from the copy completing before this returns. */
    (void)hStream;
    if (size == 0) {
        return CUDA_SUCCESS;
    }
    if (!src || !dst) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!g_copy_lock) {
        g_copy_lock = mutex_create();
    }
    mutex_lock(g_copy_lock);
    cr = ensure_copy_cmdlist();
    if (cr != CUDA_SUCCESS) {
        mutex_unlock(g_copy_lock);
        return cr;
    }
    r = zeCommandListAppendMemoryCopy(g_copy_cmdlist, (void *)(uintptr_t)dst, src, size,
                                      NULL, 0, NULL);
    mutex_unlock(g_copy_lock);
    return ze2cu(r);
}

/* Events: because every H2D copy above is blocking, an event is only ever
 * queried after the work it would track has already finished. A tiny non-null
 * heap object satisfies the hostbuf ring-slot state machine; record/synchronize
 * are no-ops. If copies ever become truly async, these must become real
 * zeEvent objects. */
CUresult CUDAAPI ze_cuEventCreate(CUevent *phEvent, unsigned int flags) {
    void *e;

    (void)flags;
    if (!phEvent) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    e = malloc(1);
    if (!e) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *phEvent = (CUevent)e;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuEventDestroy(CUevent hEvent) {
    free((void *)hEvent);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuEventRecord(CUevent hEvent, CUstream hStream) {
    (void)hEvent;
    (void)hStream;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI ze_cuEventSynchronize(CUevent hEvent) {
    (void)hEvent;
    return CUDA_SUCCESS;
}

/* ---- Fail-loud stubs (CUDA/HIP-detour-only entry points) --------------- */

CUresult CUDAAPI ze_cuMemAlloc_v2(CUdeviceptr *dptr, size_t size) {
    (void)dptr;
    (void)size;
    log(ERROR, "%s: not supported on the XPU backend (torch allocations are "
               "intercepted via the ze_loader detour)\n", __func__);
    return CUDA_ERROR_UNKNOWN;
}

CUresult CUDAAPI ze_cuMemFree_v2(CUdeviceptr dptr) {
    (void)dptr;
    log(ERROR, "%s: not supported on the XPU backend\n", __func__);
    return CUDA_ERROR_UNKNOWN;
}

CUresult CUDAAPI ze_cuMemAllocAsync(CUdeviceptr *dptr, size_t size, CUstream hStream) {
    (void)dptr;
    (void)size;
    (void)hStream;
    log(ERROR, "%s: not supported on the XPU backend\n", __func__);
    return CUDA_ERROR_UNKNOWN;
}

CUresult CUDAAPI ze_cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    (void)dptr;
    (void)hStream;
    log(ERROR, "%s: not supported on the XPU backend\n", __func__);
    return CUDA_ERROR_UNKNOWN;
}

/* ---- Detour helpers ---------------------------------------------------- */

int aimdo_ze_managed_ordinal(void) { return g_managed_ordinal; }
ze_context_handle_t aimdo_ze_context(void) { return g_context; }
ze_device_handle_t  aimdo_ze_device(void)  { return g_device; }
