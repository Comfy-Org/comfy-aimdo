#include "plat.h"

#define VBAR_PAGE_SIZE (32 << 20)

#define VBAR_GET_PAGE_NR(x) ((x) / VBAR_PAGE_SIZE)
#define VBAR_GET_PAGE_NR_UP(x) VBAR_GET_PAGE_NR((x) + VBAR_PAGE_SIZE - 1)

/* ---- global lock for the priority linked list and vbars_dirty ---- */
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
static CRITICAL_SECTION vbar_lock;
static volatile LONG vbar_lock_init;

static inline void vbar_list_lock(void) {
    if (!InterlockedCompareExchange(&vbar_lock_init, 1, 0)) {
        InitializeCriticalSection(&vbar_lock);
        InterlockedExchange(&vbar_lock_init, 2);
    }
    while (vbar_lock_init != 2) { /* spin until init done */ }
    EnterCriticalSection(&vbar_lock);
}
static inline void vbar_list_unlock(void) { LeaveCriticalSection(&vbar_lock); }
#else
#include <pthread.h>
static pthread_mutex_t vbar_lock = PTHREAD_MUTEX_INITIALIZER;
static inline void vbar_list_lock(void) { pthread_mutex_lock(&vbar_lock); }
static inline void vbar_list_unlock(void) { pthread_mutex_unlock(&vbar_lock); }
#endif

typedef struct ResidentPage {
    CUmemGenericAllocationHandle handle;
    bool pinned;
    size_t serial;
} ResidentPage;

typedef struct ModelVBAR {
    CUdeviceptr vbar;
    size_t nr_pages;
    size_t watermark;
    size_t watermark_limit;

    int device;

    void *higher;
    void *lower;

    size_t resident_count;

    ResidentPage residency_map[1]; /* Must be last! */
} ModelVBAR;

ModelVBAR highest_priority;
ModelVBAR lowest_priority;

static inline void one_time_setup() {
    if (!highest_priority.lower) {
        assert(!lowest_priority.higher);
        highest_priority.lower = &lowest_priority;
        lowest_priority.higher = &highest_priority;
    }
}

static bool vbars_dirty;

SHARED_EXPORT
uint64_t vbars_analyze(bool only_dirty) {
    size_t calculated_total_vram = 0;

    vbar_list_lock();
    one_time_setup();
    if (only_dirty && !vbars_dirty) {
        vbar_list_unlock();
        return 0;
    }
    vbars_dirty = false;
    log(DEBUG, "---------------- VBAR Usage ---------------\n")

    for (ModelVBAR *i = lowest_priority.higher; i && i != &highest_priority; i = i->higher) {
        size_t actual_resident_count = 0;

        for (size_t p = 0; p < i->nr_pages; p++) {
            ResidentPage *rp = &i->residency_map[p];

            if (rp->handle) {
                actual_resident_count++;

                if (p >= i->watermark) {
                    log(WARNING, "VBAR %p: Resident page %zu is ABOVE watermark %zu\n",
                        (void*)i, p, i->watermark);
                }

                if (rp->pinned) {
                    log(WARNING, "VBAR %p: Page %zu is PINNED\n", (void*)i, p);
                }
            }
        }

        if (actual_resident_count != i->resident_count) {
            log(WARNING, "VBAR %p: resident_count sync error! Struct: %zu, Actual: %zu\n",
                (void*)i, i->resident_count, actual_resident_count);
        }

        calculated_total_vram += (actual_resident_count * VBAR_PAGE_SIZE);

        log(DEBUG, "VBAR %p: Actual Resident VRAM = %zu MB\n",
            (void*)i, (actual_resident_count * VBAR_PAGE_SIZE) / M);
    }

    log(DEBUG, "Total VRAM for VBARs: %zu MB\n", calculated_total_vram / M);
    vbar_list_unlock();
    return (uint64_t)calculated_total_vram;
}

static inline bool mod1(ModelVBAR *mv, size_t page_nr, bool do_free, bool do_unpin) {
    ResidentPage *rp = &mv->residency_map[page_nr];
    CUdeviceptr vaddr = mv->vbar + page_nr * VBAR_PAGE_SIZE;

    do_free = do_free && rp->handle && (do_unpin || !rp->pinned);
    if (do_free) {
        CHECK_CU(cuMemUnmap(vaddr, VBAR_PAGE_SIZE));
        CHECK_CU(cuMemRelease(rp->handle));
        dev_vram_sub(mv->device, VBAR_PAGE_SIZE);
        rp->handle = 0;
        mv->resident_count--;
    }
    if (do_unpin) {
        rp->pinned = false;
    }
    return do_free;
}

/* Must be called with vbar_list_lock held.
 * device_filter: -1 = evict from any device, >= 0 = only evict from that device.
 */
static size_t vbars_free_locked_dev(size_t size, int device_filter) {
    size_t pages_needed = VBAR_GET_PAGE_NR_UP(size);
    bool synced[AIMDO_MAX_DEVICES] = {false};

    one_time_setup();
    vbars_dirty = true;

    if (!size) {
        return 0;
    }

    for (ModelVBAR *i = lowest_priority.higher; pages_needed && i != &highest_priority;
         i = i->higher) {
        if (device_filter >= 0 && i->device != device_filter)
            continue;
        for (;pages_needed && i->watermark > i->watermark_limit; i->watermark--) {
            int dev = i->device;
            if (dev >= 0 && dev < AIMDO_MAX_DEVICES && !synced[dev]) {
                CUcontext prev;
                with_device_ctx(dev, &prev);
                CHECK_CU(cuCtxSynchronize());
                restore_ctx(prev);
                synced[dev] = true;
            }
            if (mod1(i, i->watermark - 1, true, false)) {
                pages_needed--;
            }
        }
    }

    return pages_needed;
}

size_t vbars_free(size_t size) {
    size_t ret;

    vbar_list_lock();
    ret = vbars_free_locked_dev(size, -1);
    vbar_list_unlock();
    return ret;
}

size_t vbars_free_dev(size_t size, int device) {
    size_t ret;

    vbar_list_lock();
    ret = vbars_free_locked_dev(size, device);
    vbar_list_unlock();
    return ret;
}

static inline size_t move_cursor_to_absent(ModelVBAR *mv, size_t cursor) {
    while (cursor < mv->watermark && mv->residency_map[cursor].handle) {
        cursor++;
    }
    return cursor;
}

static void vbars_free_for_vbar(ModelVBAR *mv, size_t target) {
    size_t cursor = move_cursor_to_absent(mv, 0);
    int device_filter = mv->device;

    CUcontext prev;
    with_device_ctx(device_filter, &prev);
    CHECK_CU(cuCtxSynchronize());
    restore_ctx(prev);

    for (ModelVBAR *i = lowest_priority.higher;
         cursor < target && cursor < mv->watermark && i != &highest_priority;
         i = i->higher) {
        if (i->device != device_filter)
            continue;
        for (; cursor < target && cursor < mv->watermark && i->watermark > i->watermark_limit;
             i->watermark--) {
            if (mod1(i, i->watermark - 1, true, false)) {
                cursor = move_cursor_to_absent(mv, cursor + 1);
            }
        }
    }
}

static inline void remove_vbar(ModelVBAR *mv) {
    ((ModelVBAR *)mv->lower)->higher = mv->higher;
    ((ModelVBAR *)mv->higher)->lower = mv->lower;
}

static inline void insert_vbar(ModelVBAR *mv) {
    mv->lower = highest_priority.lower;
    ((ModelVBAR *)highest_priority.lower)->higher = mv;
    mv->higher = &highest_priority;
    highest_priority.lower = mv;
}

static inline void insert_vbar_last(ModelVBAR *mv) {
    mv->higher = lowest_priority.higher;
    ((ModelVBAR *)lowest_priority.higher)->lower = mv;
    mv->lower = &lowest_priority;
    lowest_priority.higher = mv;
}

SHARED_EXPORT
void *vbar_allocate(uint64_t size, int device) {
    ModelVBAR *mv;

    log_reset_shots();
    log(DEBUG, "%s (start): size=%zuM, device=%d\n", __func__, size / M, device);

    /* Phase 1: lazy init for this device */
    ensure_device_init(device);

    /* Use per-device capacity if available, else global */
    uint64_t cap = (device >= 0 && device < AIMDO_MAX_DEVICES && g_dev[device].inited)
                   ? g_dev[device].vram_capacity : vram_capacity;

    size_t nr_pages = VBAR_GET_PAGE_NR_UP(size);
    size_t nr_pages_max = VBAR_GET_PAGE_NR(cap);
    if (nr_pages_max < nr_pages) {
        nr_pages = nr_pages_max;
    }
    size = (uint64_t)nr_pages * VBAR_PAGE_SIZE;

    if (!(mv = calloc(1, sizeof(*mv) + nr_pages * sizeof(mv->residency_map[0])))) {
        log(CRITICAL, "Host OOM\n");
        return NULL;
    }

    /* FIXME: Do I care about alignment? Does Cuda just look after itself? */
    if (!CHECK_CU(cuMemAddressReserve(&mv->vbar, size, 0, 0, 0))) {
        log(ERROR, "Could not reseve Virtual Address space for VBAR\n");
        free(mv);
        return NULL;
    }

    mv->device = device;
    mv->nr_pages = mv->watermark = nr_pages;

    vbar_list_lock();
    one_time_setup();
    vbars_dirty = true;
    insert_vbar(mv);
    vbar_list_unlock();

    log(DEBUG, "%s (return): vbar=%p\n", __func__, (void *)mv);
    return mv;
}

SHARED_EXPORT
void vbar_set_watermark_limit(void *vbar, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    log(DEBUG, "%s: size=%zu\n", __func__, size);
    vbar_list_lock();
    mv->watermark_limit = VBAR_GET_PAGE_NR_UP(size);
    vbar_list_unlock();
}

SHARED_EXPORT
void vbars_reset_watermark_limits() {
    vbar_list_lock();
    one_time_setup();
    log(VERBOSE, "%s\n", __func__);

    for (ModelVBAR *i = lowest_priority.higher; i && i != &highest_priority; i = i->higher) {
        i->watermark_limit = 0;
    }
    vbar_list_unlock();
}

SHARED_EXPORT
void vbar_prioritize(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    log(DEBUG, "%s vbar=%p\n", __func__, vbar);

    log_reset_shots();

    vbar_list_lock();
    vbars_dirty = true;
    remove_vbar(mv);
    insert_vbar(mv);
    mv->watermark = mv->nr_pages;
    vbar_list_unlock();
}

SHARED_EXPORT
void vbar_deprioritize(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    log(DEBUG, "%s vbar=%p\n", __func__, vbar);

    log_reset_shots();

    vbar_list_lock();
    vbars_dirty = true;
    remove_vbar(mv);
    insert_vbar_last(mv);
    vbar_list_unlock();
}

SHARED_EXPORT
uint64_t vbar_get(void *vbar) {
    log(DEBUG, "%s vbar=%p\n", __func__, vbar);
    return (uint64_t)((ModelVBAR *)vbar)->vbar;
}

#define VBAR_FAULT_SUCCESS           0
#define VBAR_FAULT_OOM               1
#define VBAR_FAULT_ERROR             2

SHARED_EXPORT
int vbar_fault(void *vbar, uint64_t offset, uint64_t size, uint32_t *signature) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    int ret = VBAR_FAULT_SUCCESS;
    size_t signature_index = 0;

    size_t page_end = VBAR_GET_PAGE_NR_UP(offset + size);

    log(VVERBOSE, "%s (start): offset=%lldk, size=%lldk\n", __func__, (ull)(offset / K), (ull)(size / K));

    vbar_list_lock();
    vbars_dirty = true;

    /* Stopgap: use per-device budget to avoid phantom cross-GPU deficit.
     * Only evict pages from the same device to avoid cross-GPU thrashing.
     */
    vbars_free_locked_dev(budget_deficit_dev(0, mv->device), mv->device);

    if (page_end > mv->watermark) {
        log(VVERBOSE, "VBAR Allocation is above watermark\n");
        vbar_list_unlock();
        return VBAR_FAULT_OOM;
    }

    for (uint64_t page_nr = VBAR_GET_PAGE_NR(offset); page_nr < page_end; page_nr++) {
        CUresult err = CUDA_ERROR_OUT_OF_MEMORY;
        CUdeviceptr vaddr = mv->vbar + page_nr * VBAR_PAGE_SIZE;
        ResidentPage *rp = &mv->residency_map[page_nr];

        if (rp->handle) {
            signature[signature_index++] = rp->serial;
            continue;
        }

        log(VERBOSE, "VBAR needs to allocate VRAM for page %d\n", (int)page_nr);

        if (budget_deficit_dev(VBAR_PAGE_SIZE, mv->device) ||
            (err = three_stooges(vaddr, VBAR_PAGE_SIZE, mv->device, &rp->handle)) != CUDA_SUCCESS) {
            if (err != CUDA_ERROR_OUT_OF_MEMORY) {
                log(ERROR, "VRAM Allocation failed (non OOM)\n");
                vbar_list_unlock();
                return VBAR_FAULT_ERROR;
            }
            log(DEBUG, "VBAR allocator attempt exceeds available VRAM ...\n");
            vbars_free_for_vbar(mv, page_end);
            if (page_nr >= mv->watermark) {
                log(DEBUG, "VBAR allocation cancelled due to watermark reduction\n");
                vbar_list_unlock();
                return VBAR_FAULT_OOM;
            }
            if ((err = three_stooges(vaddr, VBAR_PAGE_SIZE, mv->device, &rp->handle)) != CUDA_SUCCESS) {
                log(ERROR, "VRAM Allocation failed\n");
                vbar_list_unlock();
                return VBAR_FAULT_ERROR;
            }
        }
        rp->serial++;
        signature[signature_index++] = rp->serial;
        mv->resident_count++;
    }

    /* We got our allocation */

    for (uint64_t page_nr = VBAR_GET_PAGE_NR(offset); page_nr < page_end; page_nr++) {
        ResidentPage *rp = &mv->residency_map[page_nr];
        rp->pinned = true;
    }

    vbar_list_unlock();
    log(VVERBOSE, "%s (return) %d\n", __func__, ret);
    return ret;
}

SHARED_EXPORT
void vbar_unpin(void *vbar, uint64_t offset, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    log(VVERBOSE, "%s (start): offset=%lldk, size=%lldk\n", __func__, (ull)(offset / K), (ull)(size / K));

    vbar_list_lock();
    vbars_dirty = true;
    size_t page_end = VBAR_GET_PAGE_NR_UP(offset + size);

    if (page_end > mv->watermark) {
        CUcontext prev;
        with_device_ctx(mv->device, &prev);
        CHECK_CU(cuCtxSynchronize());
        restore_ctx(prev);
    }

    for (uint64_t page_nr = VBAR_GET_PAGE_NR(offset); page_nr < page_end && page_nr < mv->nr_pages; page_nr++) {
        mod1(mv, page_nr, page_nr >= mv->watermark, true);
    }
    vbar_list_unlock();
}

SHARED_EXPORT
void vbar_free(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    log(DEBUG, "%s: vbar=%p\n", __func__, vbar);

    vbar_list_lock();
    vbars_dirty = true;

    {
        CUcontext prev;
        with_device_ctx(mv->device, &prev);
        CHECK_CU(cuCtxSynchronize());
        restore_ctx(prev);
    }

    for (uint64_t page_nr = 0; page_nr < mv->nr_pages; page_nr++) {
        mod1(mv, page_nr, true, true);
    }
    remove_vbar(mv);
    vbar_list_unlock();

    CHECK_CU(cuMemAddressFree(mv->vbar, (size_t)mv->nr_pages * VBAR_PAGE_SIZE));
    free(mv);
}

SHARED_EXPORT
size_t vbar_loaded_size(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    return mv->resident_count * VBAR_PAGE_SIZE;
}

SHARED_EXPORT
size_t vbar_get_nr_pages(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    return mv->nr_pages;
}

SHARED_EXPORT
size_t vbar_get_watermark(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    return mv->watermark;
}

SHARED_EXPORT
void vbar_get_residency(void *vbar, uint8_t *out, size_t max_pages) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t n = mv->nr_pages < max_pages ? mv->nr_pages : max_pages;
    for (size_t i = 0; i < n; i++) {
        ResidentPage *rp = &mv->residency_map[i];
        /* bit 0: resident, bit 1: pinned */
        out[i] = (rp->handle ? 1 : 0) | (rp->pinned ? 2 : 0);
    }
}

SHARED_EXPORT
uint64_t vbar_free_memory(void *vbar, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t pages_to_free = VBAR_GET_PAGE_NR_UP(size);
    size_t pages_freed = 0;

    log(DEBUG, "%s (start): size=%lldk\n", __func__, (ull)size);

    vbar_list_lock();
    vbars_dirty = true;

    {
        CUcontext prev;
        with_device_ctx(mv->device, &prev);
        CHECK_CU(cuCtxSynchronize());
        restore_ctx(prev);
    }

    for (;pages_to_free && mv->watermark > mv->watermark_limit; mv->watermark--) {
        /* In theory we should never have pins here, but
         * respect pins if it really comes up.
         */
        if (mod1(mv, mv->watermark - 1, true, false)) {
            pages_to_free--;
            pages_freed++;
        }
    }

    vbar_list_unlock();
    return (uint64_t)pages_freed * VBAR_PAGE_SIZE;
}
