#include "plat.h"

#include <windows.h>
#include <dxgi1_4.h>

#include <cuda.h>

static struct {
    IDXGIFactory4 *factory;
    IDXGIAdapter3 *adapter;
} G_WDDM;

bool plat_init(CUdevice dev)
{
    int fail_code = 1;
    LUID cuda_luid;
    IDXGIAdapter1 *adapter;
    UINT i;
    unsigned int node_mask;

    adapter = NULL;

    if (!CHECK_CU(cuDeviceGetLuid((char *)&cuda_luid, &node_mask, dev))) {
        goto fail;
    }

    fail_code++;

    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void **)&G_WDDM.factory))) {
        goto fail;
    }

    for (i = 0; G_WDDM.factory->lpVtbl->EnumAdapters1(G_WDDM.factory, i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->lpVtbl->GetDesc1(adapter, &desc);

        if (desc.AdapterLuid.LowPart == cuda_luid.LowPart &&
            desc.AdapterLuid.HighPart == cuda_luid.HighPart) {

            if (FAILED(adapter->lpVtbl->QueryInterface(adapter, &IID_IDXGIAdapter3, (void **)&G_WDDM.adapter))) {
                adapter->lpVtbl->Release(adapter);
                break;
            }

            adapter->lpVtbl->Release(adapter);
            return true;
        }
        adapter->lpVtbl->Release(adapter);
    }

fail:
    G_WDDM.adapter = NULL;
    log(WARNING, "comfy-aimdo WDDM init failed (%d). aimdo is blind to the CUDA Sysmem Fallback Policy\n", fail_code)
    return false;
}

/* Apparently this is still too small for all common graphics VRAM spikes.
 * However we can't pad too much on the smaller cards, and its not the end
 * of the world if we page out a little bit because it will adapt and correct
 * quickly.
 */

/* FIXME: This should be 0 if sysmem fallback is disabled by the user */
#define WDDM_BUDGET_HEADROOM (512 * 1024 * 1024)

size_t wddm_budget_deficit(int device, size_t bytes)
{
    DXGI_QUERY_VIDEO_MEMORY_INFO info;
    uint64_t effective_budget = vram_capacity;
    ssize_t deficit;

    if (G_WDDM.adapter) {
        if (SUCCEEDED(G_WDDM.adapter->lpVtbl->QueryVideoMemoryInfo(G_WDDM.adapter, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            effective_budget = info.Budget;
        } else {
            log(WARNING, "comfy-aimdo WDDM VRAM query failed. Using physical capacity as fallback\n");
        }
    }

    deficit = (ssize_t)(total_vram_usage + bytes + WDDM_BUDGET_HEADROOM) - (ssize_t)effective_budget;

    if (deficit > 0) {
        log(DEBUG, "Imminent WDDM VRAM OOM detected. Budget: %llu MB, Request: %zu MB, Deficit: %zd MB\n",
            effective_budget / (1024 * 1024), bytes / (1024 * 1024), deficit / (1024 * 1024));
    }

    return (deficit > 0) ? (size_t)deficit : 0;
}

void plat_cleanup()
{
    if (G_WDDM.adapter) {
        G_WDDM.adapter->lpVtbl->Release(G_WDDM.adapter);
    }
    if (G_WDDM.factory) {
        G_WDDM.factory->lpVtbl->Release(G_WDDM.factory);
    }
}
