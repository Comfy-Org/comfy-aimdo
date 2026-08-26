#include "plat.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static void *g_cuda_module;
#if defined(_WIN32) || defined(_WIN64)
static void *g_nvml_module;
#endif

AimdoCudaDispatch g_cuda;

#if defined(_WIN32) || defined(_WIN64)
typedef int nvmlReturn_t;
typedef struct nvmlDevice_st *nvmlDevice_t;
typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef nvmlReturn_t (*PFN_nvmlInit_v2)(void);
typedef nvmlReturn_t (*PFN_nvmlShutdown)(void);
typedef nvmlReturn_t (*PFN_nvmlDeviceGetHandleByUUID)(const char *uuid, nvmlDevice_t *device);
typedef nvmlReturn_t (*PFN_nvmlDeviceGetMemoryInfo)(nvmlDevice_t device, nvmlMemory_t *memory);

static PFN_nvmlInit_v2 p_nvmlInit_v2;
static PFN_nvmlShutdown p_nvmlShutdown;
static PFN_nvmlDeviceGetHandleByUUID p_nvmlDeviceGetHandleByUUID;
static PFN_nvmlDeviceGetMemoryInfo p_nvmlDeviceGetMemoryInfo;
static bool g_nvml_initialized;

#define NVML_SUCCESS 0
#endif

/* Keep the ABI target pinned to the minimum supported CUDA version so
 * cuGetProcAddress does not hand us newer function revisions with different
 * signatures.
 */
#define AIMDO_CUDA_ABI_VERSION 12060

typedef struct {
    void **slot;
    const char *symbol;
    cuuint64_t flags;
} DispatchSymbol;

static const DispatchSymbol dispatch_symbols[] = {
    { (void **)&g_cuda.p_cuInit, "cuInit", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuGetErrorString, "cuGetErrorString", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuCtxGetDevice, "cuCtxGetDevice", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuCtxSynchronize, "cuCtxSynchronize", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuDeviceGet, "cuDeviceGet", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuDeviceGetAttribute, "cuDeviceGetAttribute", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuDeviceTotalMem, "cuDeviceTotalMem", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuDeviceGetName, "cuDeviceGetName", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuDeviceGetUuid, "cuDeviceGetUuid", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemGetInfo, "cuMemGetInfo", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemAlloc_v2, "cuMemAlloc", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemFree_v2, "cuMemFree", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemAllocAsync, "cuMemAllocAsync", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemAllocAsync_ptsz, "cuMemAllocAsync", CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM },
    { (void **)&g_cuda.p_cuMemFreeAsync, "cuMemFreeAsync", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemFreeAsync_ptsz, "cuMemFreeAsync", CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM },
    { (void **)&g_cuda.p_cuMemAllocHost, "cuMemAllocHost", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemFreeHost, "cuMemFreeHost", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemHostRegister, "cuMemHostRegister", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemHostUnregister, "cuMemHostUnregister", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemAddressReserve, "cuMemAddressReserve", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemAddressFree, "cuMemAddressFree", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemCreate, "cuMemCreate", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemMap, "cuMemMap", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemSetAccess, "cuMemSetAccess", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemUnmap, "cuMemUnmap", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemRelease, "cuMemRelease", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuEventCreate, "cuEventCreate", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuEventDestroy, "cuEventDestroy", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuEventRecord, "cuEventRecord", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
    { (void **)&g_cuda.p_cuEventSynchronize, "cuEventSynchronize", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
#if defined(_WIN32) || defined(_WIN64)
    { (void **)&g_cuda.p_cuDeviceGetLuid, "cuDeviceGetLuid", CU_GET_PROC_ADDRESS_LEGACY_STREAM },
#endif
};

static const char *const cuda_library_names[] = {
#if defined(_WIN32) || defined(_WIN64)
    "nvcuda.dll",
    "nvcuda64.dll",
#else
    "libcuda.so.1",
    "libcuda.so",
#endif
};

#if defined(_WIN32) || defined(_WIN64)
static void *aimdo_nvml_resolve_symbol(const char *symbol) {
    return (void *)GetProcAddress((HMODULE)g_nvml_module, symbol);
}

static void aimdo_nvml_runtime_cleanup(void) {
    if (g_nvml_initialized) {
        p_nvmlShutdown();
        g_nvml_initialized = false;
    }
    p_nvmlInit_v2 = NULL;
    p_nvmlShutdown = NULL;
    p_nvmlDeviceGetHandleByUUID = NULL;
    p_nvmlDeviceGetMemoryInfo = NULL;
    if (g_nvml_module) {
        FreeLibrary((HMODULE)g_nvml_module);
        g_nvml_module = NULL;
    }
}

static bool aimdo_nvml_runtime_init(void) {
    static const WCHAR nvml_fallback[] = L"\\NVIDIA Corporation\\NVSMI\\nvml.dll";

    if (g_nvml_initialized) {
        return true;
    }

    WCHAR path[MAX_PATH];
    DWORD length;

    g_nvml_module = (void *)LoadLibraryExW(L"nvml.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!g_nvml_module &&
        (length = GetEnvironmentVariableW(L"ProgramW6432", path, ARRAY_SIZE(path))) > 0 &&
        length + ARRAY_SIZE(nvml_fallback) <= ARRAY_SIZE(path)) {
        memcpy(path + length, nvml_fallback, sizeof(nvml_fallback));
        g_nvml_module = (void *)LoadLibraryW(path);
    }
    if (!g_nvml_module) {
        log(AIMDO_LOG_ERROR, "%s: failed to load NVML\n", __func__);
        return false;
    }

    p_nvmlInit_v2 = (PFN_nvmlInit_v2)aimdo_nvml_resolve_symbol("nvmlInit_v2");
    p_nvmlShutdown = (PFN_nvmlShutdown)aimdo_nvml_resolve_symbol("nvmlShutdown");
    p_nvmlDeviceGetHandleByUUID = (PFN_nvmlDeviceGetHandleByUUID)aimdo_nvml_resolve_symbol("nvmlDeviceGetHandleByUUID");
    p_nvmlDeviceGetMemoryInfo = (PFN_nvmlDeviceGetMemoryInfo)aimdo_nvml_resolve_symbol("nvmlDeviceGetMemoryInfo");
    if (!p_nvmlInit_v2 || !p_nvmlShutdown || !p_nvmlDeviceGetHandleByUUID || !p_nvmlDeviceGetMemoryInfo) {
        log(AIMDO_LOG_ERROR, "%s: failed to resolve required NVML symbols\n", __func__);
        aimdo_nvml_runtime_cleanup();
        return false;
    }
    if (p_nvmlInit_v2() != NVML_SUCCESS) {
        log(AIMDO_LOG_ERROR, "%s: failed to initialize NVML\n", __func__);
        aimdo_nvml_runtime_cleanup();
        return false;
    }
    g_nvml_initialized = true;

    return true;
}

bool aimdo_nvml_device_init(CUdevice device, void **handle) {
    CUuuid cuda_uuid;
    char uuid[41];
    unsigned char *b = (unsigned char *)cuda_uuid.bytes;
    nvmlDevice_t nvml_handle;

    if (!handle || !aimdo_nvml_runtime_init() ||
        !CHECK_CU(g_cuda.p_cuDeviceGetUuid(&cuda_uuid, device))) {
        return false;
    }
    snprintf(uuid, sizeof(uuid),
             "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
             b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    if (p_nvmlDeviceGetHandleByUUID(uuid, &nvml_handle) != NVML_SUCCESS) {
        memcpy(uuid, "MIG-", 4);
        if (p_nvmlDeviceGetHandleByUUID(uuid, &nvml_handle) != NVML_SUCCESS) {
            log(AIMDO_LOG_ERROR, "%s: failed to resolve NVML device %s\n", __func__, uuid);
            return false;
        }
    }
    *handle = (void *)nvml_handle;
    return true;
}

bool aimdo_nvml_memory_info(void *handle, size_t *free_bytes, size_t *total_bytes) {
    nvmlMemory_t memory;

    if (!g_nvml_initialized || !handle || !free_bytes || !total_bytes ||
        p_nvmlDeviceGetMemoryInfo((nvmlDevice_t)handle, &memory) != NVML_SUCCESS) {
        return false;
    }
    *free_bytes = (size_t)memory.free;
    *total_bytes = (size_t)memory.total;
    return true;
}
#endif

bool aimdo_cuda_runtime_init(void) {
    if (g_cuda.p_cuInit) {
        return true;
    }

    if (!(g_cuda_module = aimdo_find_loaded_module(
              cuda_library_names, ARRAY_SIZE(cuda_library_names)))) {
        return false;
    }

#if defined(_WIN32) || defined(_WIN64)
    g_cuda.p_cuGetProcAddress = (PFN_cuGetProcAddress)GetProcAddress((HMODULE)g_cuda_module,
                                                                     "cuGetProcAddress");
#else
    g_cuda.p_cuGetProcAddress = (PFN_cuGetProcAddress)dlsym(g_cuda_module,
                                                            "cuGetProcAddress");
#endif
    if (!g_cuda.p_cuGetProcAddress) {
        log(AIMDO_LOG_ERROR, "%s: failed to resolve cuGetProcAddress\n", __func__);
        aimdo_cuda_runtime_cleanup();
        return false;
    }

    for (size_t i = 0; i < ARRAY_SIZE(dispatch_symbols); i++) {
        void *resolved = NULL;

        if (g_cuda.p_cuGetProcAddress(dispatch_symbols[i].symbol, &resolved,
                                      AIMDO_CUDA_ABI_VERSION,
                                      dispatch_symbols[i].flags, NULL) != CUDA_SUCCESS) {
            resolved = NULL;
        }
        if (!resolved) {
            log(AIMDO_LOG_ERROR, "%s: failed to resolve required CUDA symbol %s\n", __func__,
                dispatch_symbols[i].symbol);
            aimdo_cuda_runtime_cleanup();
            return false;
        }
        *dispatch_symbols[i].slot = resolved;
    }

    if (g_cuda.p_cuInit(0) != CUDA_SUCCESS) {
        log(AIMDO_LOG_ERROR, "%s: cuInit failed\n", __func__);
        aimdo_cuda_runtime_cleanup();
        return false;
    }

    return true;
}

void aimdo_cuda_runtime_cleanup(void) {
#if defined(_WIN32) || defined(_WIN64)
    aimdo_nvml_runtime_cleanup();
#endif
    memset(&g_cuda, 0, sizeof(g_cuda));

    if (!g_cuda_module) {
        return;
    }

#if !defined(_WIN32) && !defined(_WIN64)
    dlclose(g_cuda_module);
#endif
    g_cuda_module = NULL;
}
