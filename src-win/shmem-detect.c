#include "plat.h"
#include "aimdo-time.h"

#include <windows.h>
#include <dxgi1_4.h>

#if defined(__HIP_PLATFORM_AMD__)
typedef union {
    struct {
        char name[256];
        char uuid[16];
        char luid[8];
    };
    void *ptr;
    unsigned long long ull;
    unsigned char bytes[4096];
} AimdoHipDeviceProp;
#endif

bool aimdo_wddm_init(CUdevice dev)
{
    int fail_code = 1;
    LUID cuda_luid;
    char adapter_name[256];
    IDXGIFactory4 *factory;
    IDXGIAdapter1 *adapter;
    UINT i;

    factory = NULL;
    adapter = NULL;
    adapter_name[0] = '\0';
    if (g_wddm_adapter) {
        g_wddm_adapter->lpVtbl->Release(g_wddm_adapter);
        g_wddm_adapter = NULL;
    }

#if defined(__HIP_PLATFORM_AMD__)
    AimdoHipDeviceProp hip_props = {0};

    if (!g_device_get_properties ||
        !CHECK_CU(g_device_get_properties(hip_props.bytes, dev))) {
        goto fail;
    }
    memcpy(&cuda_luid, hip_props.luid, sizeof(cuda_luid));
#else
    unsigned int node_mask;
    if (!CHECK_CU(cuDeviceGetLuid((char *)&cuda_luid, &node_mask, dev))) {
        goto fail;
    }
#endif

    fail_code++;

    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void **)&factory))) {
        goto fail;
    }

    for (i = 0; factory->lpVtbl->EnumAdapters1(factory, i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->lpVtbl->GetDesc1(adapter, &desc);

        if (desc.AdapterLuid.LowPart == cuda_luid.LowPart &&
            desc.AdapterLuid.HighPart == cuda_luid.HighPart) {
            if (!WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, adapter_name,
                                     sizeof(adapter_name), NULL, NULL)) {
                strcpy(adapter_name, "<unknown>");
            }

            if (FAILED(adapter->lpVtbl->QueryInterface(adapter, &IID_IDXGIAdapter3, (void **)&g_wddm_adapter))) {
                adapter->lpVtbl->Release(adapter);
                break;
            }

            log(INFO,
                "comfy-aimdo WDDM adapter match: %s runtime_luid=%08lx:%08lx dxgi_luid=%08lx:%08lx\n",
                adapter_name,
                (unsigned long)(unsigned int)cuda_luid.HighPart,
                (unsigned long)cuda_luid.LowPart,
                (unsigned long)(unsigned int)desc.AdapterLuid.HighPart,
                (unsigned long)desc.AdapterLuid.LowPart);

            adapter->lpVtbl->Release(adapter);
            factory->lpVtbl->Release(factory);
            return true;
        }
        adapter->lpVtbl->Release(adapter);
    }

fail:
    g_wddm_adapter = NULL;
    if (factory) {
        factory->lpVtbl->Release(factory);
    }
    log(WARNING, "comfy-aimdo WDDM init failed (%d). aimdo is blind to the CUDA Sysmem Fallback Policy\n", fail_code);
    return false;
}

/* Apparently this is still too small for all common graphics VRAM spikes.
 * However we can't pad too much on the smaller cards, and its not the end
 * of the world if we page out a little bit because it will adapt and correct
 * quickly.
 */

/* FIXME: This should be 0 if sysmem fallback is disabled by the user */
#define WDDM_BUDGET_HEADROOM (512 * 1024 * 1024)
#define CUDA_BUDGET_HEADROOM (192 * 1024 * 1024)
#define NVML_BUDGET_HEADROOM (512 * 1024 * 1024)

bool poll_budget_deficit(const char **prevailing_deficit_method)
{
    DXGI_QUERY_VIDEO_MEMORY_INFO info;
    size_t free_vram = 0, total_vram = 0;
    bool used_nvml = false;

    uint64_t now = GET_TICK();

    if (now - wddm_timestamp_last_check < 2000) {
        return true;
    }
    wddm_timestamp_last_check = now;
    total_vram_last_check = total_vram_usage;

    if (g_wddm_adapter) {
        if (SUCCEEDED(g_wddm_adapter->lpVtbl->QueryVideoMemoryInfo(g_wddm_adapter, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            log(DEBUG,
                "%s: WDDM budget=%zu MB usage=%zu MB reservation=%zu MB available=%zu MB\n",
                __func__, (size_t)(info.Budget / M), (size_t)(info.CurrentUsage / M),
                (size_t)(info.CurrentReservation / M),
                (size_t)(info.AvailableForReservation / M));
        } else {
            log(WARNING, "comfy-aimdo WDDM VRAM query failed. Using physical capacity as fallback\n");
        }
    }

    deficit_sync = -(ssize_t)(1ULL << 60);
    *prevailing_deficit_method = "none";

#if defined(AIMDO_CUDA)
    used_nvml = nvml_device && aimdo_nvml_memory_info(nvml_device, &free_vram, &total_vram);
#endif
    if (used_nvml || CHECK_CU(cuMemGetInfo(&free_vram, &total_vram))) {
        ssize_t headroom = used_nvml ? NVML_BUDGET_HEADROOM : CUDA_BUDGET_HEADROOM / 2;
        ssize_t deficit_cuda = headroom - (ssize_t)free_vram;

        log(DEBUG,
            "%s: device memory free=%zu MB total=%zu MB deficit_cuda=%zd MB\n",
            __func__, free_vram / M, total_vram / M, deficit_cuda / (ssize_t)M);

        if (deficit_cuda > deficit_sync) {
            deficit_sync = deficit_cuda;
            *prevailing_deficit_method = used_nvml ? "NVML (Windows)" : "cuMemGetInfo (Windows)";
        }
    }

    log(DEBUG, "%s: prevailing method %s\n", __func__, *prevailing_deficit_method);
    return true;
}

void aimdo_wddm_cleanup()
{
    if (g_wddm_adapter) {
        g_wddm_adapter->lpVtbl->Release(g_wddm_adapter);
        g_wddm_adapter = NULL;
    }
}
