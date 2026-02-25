#include "plat.h"
#include "aimdo-time.h"

#include <windows.h>
#include <dxgi1_4.h>

#include <cuda.h>

static struct {
    IDXGIFactory4 *factory;
    IDXGIAdapter3 *adapter;
} G_WDDM;

bool aimdo_wddm_init(CUdevice dev)
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
#define WDDM_BUDGET_HEADROOM (512 * 1024 * 1024)

/* Pytorch will also use a non of NON_LOCAL VRAM for the sake its transfer
 * buffer that are off the books to Aimdo. That forms a natural head room
 * so keep this as a small addition to that.
 */
#define WDDM_NL_CHECK_HEADROOM (128 * 1024 * 1024)

/* Cuda 12 is very defensive in under-reporting the available VRAM under WDDM
 * so keep this lower than on Linux.
 */
#define CUDA_BUDGET_HEADROOM (128 * 1024 * 1024)

bool poll_budget_deficit()
{
    uint64_t now = GET_TICK();
    static uint64_t last_check = 0;

    ssize_t effective_budget = (ssize_t)vram_capacity - VRAM_HEADROOM;
    ssize_t total = (ssize_t)total_vram_usage;

    if (now - last_check < 2000) {
        return true;
    }
    last_check = now;
    total_vram_last_check = total_vram_usage;
    deficit_sync = -(ssize_t)(512LL * G);

    prevailing_deficit_method = "None";

    if (G_WDDM.adapter) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info;
        DXGI_QUERY_VIDEO_MEMORY_INFO info_nl;

        if (SUCCEEDED(G_WDDM.adapter->lpVtbl->QueryVideoMemoryInfo(G_WDDM.adapter, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            ssize_t adjusted_budget = (ssize_t)info.Budget - WDDM_BUDGET_HEADROOM;
            /* The most pessimistic number is the truth. */
            if (adjusted_budget < effective_budget) {
                effective_budget = adjusted_budget;
            }
            if (info.CurrentUsage > total) {
                total = info.CurrentUsage;
            }
            if (SUCCEEDED(G_WDDM.adapter->lpVtbl->QueryVideoMemoryInfo(G_WDDM.adapter, 0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info_nl))) {
                deficit_sync = (ssize_t)info.CurrentUsage + (ssize_t)info_nl.CurrentUsage - total_pin_usage +
                          WDDM_NL_CHECK_HEADROOM - (ssize_t)info.Budget;
                prevailing_deficit_method = "WDDM non-local memory inbalance";
            } else {
                log(WARNING, "comfy-aimdo WDDM VRAM query failed. Using physical capacity as fallback\n");
            }
        } else {
            log(WARNING, "comfy-aimdo WDDM VRAM query failed. Using physical capacity as fallback\n");
        }
    }

    {
        ssize_t deficit_pessimism = total - effective_budget;

        if (deficit_pessimism > deficit_sync) {
            deficit_sync = deficit_pessimism;
            prevailing_deficit_method = "WDDM pessimistic memory estimation";
        }
    } {
        size_t free_vram = 0, total_vram = 0;
        ssize_t deficit_cuda;

        if (!CHECK_CU(cuMemGetInfo(&free_vram, &total_vram))) {
            return false;
        }
        deficit_cuda = (ssize_t)CUDA_BUDGET_HEADROOM - free_vram;

        if (deficit_cuda > deficit_sync) {
            deficit_sync = deficit_cuda;
            prevailing_deficit_method = "cuMemGetInfo (Windows)";
        }
    }

    return true;
}

void aimdo_wddm_cleanup()
{
    if (G_WDDM.adapter) {
        G_WDDM.adapter->lpVtbl->Release(G_WDDM.adapter);
    }
    if (G_WDDM.factory) {
        G_WDDM.factory->lpVtbl->Release(G_WDDM.factory);
    }
}
