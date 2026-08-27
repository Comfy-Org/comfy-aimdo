#include "plat.h"
#include "aimdo-time.h"

#include <windows.h>
#include <dxgi1_4.h>
#include "dxcore_c.h"

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

IDXCoreAdapter *g_dxcore_adapter = NULL;

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
    if (g_dxcore_adapter) {
        g_dxcore_adapter->lpVtbl->Release(g_dxcore_adapter);
        g_dxcore_adapter = NULL;
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

    {
        IDXCoreAdapterFactory* dxcore_factory = NULL;
        IDXCoreAdapterList* dxcore_list = NULL;
        if (SUCCEEDED(DXCoreCreateAdapterFactory(&IID_IDXCoreAdapterFactory, (void**)&dxcore_factory))) {
            const GUID attributes[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
            if (SUCCEEDED(dxcore_factory->lpVtbl->CreateAdapterList(dxcore_factory, 1, attributes, &IID_IDXCoreAdapterList, (void**)&dxcore_list))) {
                uint32_t count = dxcore_list->lpVtbl->GetAdapterCount(dxcore_list);
                for (uint32_t j = 0; j < count; ++j) {
                    IDXCoreAdapter* d_adapter = NULL;
                    if (SUCCEEDED(dxcore_list->lpVtbl->GetAdapter(dxcore_list, j, &IID_IDXCoreAdapter, (void**)&d_adapter))) {
                        LUID luid;
                        if (SUCCEEDED(d_adapter->lpVtbl->GetProperty(d_adapter, DXCoreAdapterProperty_InstanceLuid, sizeof(luid), &luid))) {
                            if (luid.LowPart == cuda_luid.LowPart && luid.HighPart == cuda_luid.HighPart) {
                                DXCoreAdapterMemoryBudgetNodeSegmentGroup nodeSegmentGroup = {0};
                                DXCoreAdapterMemoryBudget test_budget = {0};
                                nodeSegmentGroup.segmentGroup = DXCoreSegmentGroup_Local;
                                
                                if (SUCCEEDED(d_adapter->lpVtbl->QueryState(d_adapter, DXCoreAdapterState_AdapterMemoryBudget, sizeof(nodeSegmentGroup), &nodeSegmentGroup, sizeof(test_budget), &test_budget))) {
                                    g_dxcore_adapter = d_adapter; 
                                    
                                    d_adapter->lpVtbl->GetProperty(d_adapter, DXCoreAdapterProperty_DriverDescription, sizeof(adapter_name), adapter_name);
                                    log(INFO,
                                        "comfy-aimdo DXCore adapter match: %s runtime_luid=%08lx:%08lx\n",
                                        adapter_name,
                                        (unsigned long)(unsigned int)cuda_luid.HighPart,
                                        (unsigned long)cuda_luid.LowPart);
                                    
                                    break;
                                }
                            }
                        }
                        if (g_dxcore_adapter != d_adapter) {
                            d_adapter->lpVtbl->Release(d_adapter);
                        }
                    }
                }
                dxcore_list->lpVtbl->Release(dxcore_list);
            }
            dxcore_factory->lpVtbl->Release(dxcore_factory);
        }
    }

    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void **)&factory))) {
        if (g_dxcore_adapter) return true;
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

    if (factory) {
        factory->lpVtbl->Release(factory);
    }
    
    if (g_dxcore_adapter) {
        return true;
    }

fail:
    g_wddm_adapter = NULL;
    if (g_dxcore_adapter) {
        g_dxcore_adapter->lpVtbl->Release(g_dxcore_adapter);
        g_dxcore_adapter = NULL;
    }
    log(WARNING, "comfy-aimdo WDDM init failed (%d). aimdo is blind to the CUDA Sysmem Fallback Policy\n", fail_code);
    return false;
}

#define WDDM_BUDGET_HEADROOM (512 * 1024 * 1024)
#define CUDA_BUDGET_HEADROOM (192 * 1024 * 1024)
#define NVML_BUDGET_HEADROOM (512 * 1024 * 1024)

bool poll_budget_deficit(const char **prevailing_deficit_method)
{
    DXGI_QUERY_VIDEO_MEMORY_INFO info;
    uint64_t effective_budget = vram_capacity;
    size_t free_vram = 0, total_vram = 0;
    bool budget_queried = false;
    bool used_nvml = false;

    uint64_t now = GET_TICK();

    if (now - wddm_timestamp_last_check < 2000) {
        return true;
    }
    wddm_timestamp_last_check = now;
    total_vram_last_check = total_vram_usage;

    *prevailing_deficit_method = "Physical Capacity (Fallback)";

    if (g_dxcore_adapter) {
        DXCoreAdapterMemoryBudgetNodeSegmentGroup nodeSegmentGroup = {0};
        DXCoreAdapterMemoryBudget budget = {0};
        nodeSegmentGroup.segmentGroup = DXCoreSegmentGroup_Local;

        if (SUCCEEDED(g_dxcore_adapter->lpVtbl->QueryState(g_dxcore_adapter, DXCoreAdapterState_AdapterMemoryBudget, sizeof(nodeSegmentGroup), &nodeSegmentGroup, sizeof(budget), &budget))) {
            effective_budget = budget.budget;
            *prevailing_deficit_method = "DXCore budget";
            budget_queried = true;
            log(DEBUG,
                "%s: DXCore budget=%zu MB usage=%zu MB reservation=%zu MB available=%zu MB\n",
                __func__, (size_t)(budget.budget / M), (size_t)(budget.currentUsage / M),
                (size_t)(budget.currentReservation / M),
                (size_t)(budget.availableForReservation / M));
        } else {
            log(WARNING, "comfy-aimdo DXCore VRAM query failed. Falling back to WDDM...\n");
        }
    } 
    
    if (!budget_queried && g_wddm_adapter) {
        if (SUCCEEDED(g_wddm_adapter->lpVtbl->QueryVideoMemoryInfo(g_wddm_adapter, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            effective_budget = info.Budget;
            *prevailing_deficit_method = "WDDM budget";
            budget_queried = true;
            log(DEBUG,
                "%s: WDDM budget=%zu MB usage=%zu MB reservation=%zu MB available=%zu MB\n",
                __func__, (size_t)(info.Budget / M), (size_t)(info.CurrentUsage / M),
                (size_t)(info.CurrentReservation / M),
                (size_t)(info.AvailableForReservation / M));
        } else {
            log(WARNING, "comfy-aimdo WDDM VRAM query failed. Using physical capacity as fallback\n");
        }
    }

    deficit_sync = (ssize_t)(total_vram_usage + WDDM_BUDGET_HEADROOM) - (ssize_t)effective_budget;

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

void aimdo_wddm_cleanup(void)
{
    if (g_wddm_adapter) {
        g_wddm_adapter->lpVtbl->Release(g_wddm_adapter);
        g_wddm_adapter = NULL;
    }
    if (g_dxcore_adapter) {
        g_dxcore_adapter->lpVtbl->Release(g_dxcore_adapter);
        g_dxcore_adapter = NULL;
    }
}