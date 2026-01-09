#include "plat.h"

#include <windows.h>
#include <dxgi1_4.h>

typedef SSIZE_T ssize_t;

static struct {
    IDXGIFactory4 *factory;
    IDXGIAdapter3 *adapter;
} G_WDDM;

static bool wddm_inited;

SHARED_EXPORT
bool wddm_init(int device_id)
{
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&G_WDDM.factory))) {
        log(WARNING, "aimdo WDDM init failed (1). aimdo is blind to the CUDA Sysmem Fallback Policy\n")
        return false;
    }

    if (FAILED(G_WDDM.factory->lpVtbl->EnumAdapters1(G_WDDM.factory, device_id, (IDXGIAdapter1**)&G_WDDM.adapter))) {
        G_WDDM.factory->lpVtbl->Release(G_WDDM.factory);
        log(WARNING, "aimdo WDDM init failed (2). aimdo is blind to the CUDA Sysmem Fallback Policy\n")
        return false;
    }

    wddm_inited = true;
    return true;
}

#define WDDM_BUDGET_HEADROOM (192 * 1024 * 1024)

size_t wddm_budget_deficit(size_t bytes)
{
    ssize_t deficit;
    DXGI_QUERY_VIDEO_MEMORY_INFO info;

    if (!wddm_inited) {
        return 0;
    }

    if (FAILED(G_WDDM.adapter->lpVtbl->QueryVideoMemoryInfo(G_WDDM.adapter, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        log_shot(WARNING, "aimdo WDDM VRAM query failed. aimdo is blind to the CUDA Sysmem Fallback Policy\n");
        return 0;
    }

    deficit = info.CurrentUsage + bytes + WDDM_BUDGET_HEADROOM - info.Budget;
    if (deficit > 0) {
        log(DEBUG, "Imminent WDDM VRAM OOM detected %lld/%lld MB VRAM used (+%lldMB)\n",
                    (ull)info.CurrentUsage, (ull)info.Budget, (ull)bytes);
    }
    return deficit < 0 ? 0 : (size_t)deficit;
}

SHARED_EXPORT
void wddm_cleanup()
{
    if (G_WDDM.adapter) {
        G_WDDM.adapter->lpVtbl->Release(G_WDDM.adapter);
    }
    if (G_WDDM.factory) {
        G_WDDM.factory->lpVtbl->Release(G_WDDM.factory);
    }
}
