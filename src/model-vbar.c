#include "plat.h"

#define VBAR_PAGE_SIZE (32 << 20)

#define VBAR_GET_PAGE_NR(x) ((x) / VBAR_PAGE_SIZE)
#define VBAR_GET_PAGE_NR_UP(x) VBAR_GET_PAGE_NR((x) + VBAR_PAGE_SIZE - 1)

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

void vbars_analyze() {
    size_t calculated_total_vram = 0;

    one_time_setup();
    if (!vbars_dirty) {
        return;
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
}

static inline bool mod1(ModelVBAR *mv, size_t page_nr, bool do_free, bool do_unpin) {
    ResidentPage *rp = &mv->residency_map[page_nr];
    CUdeviceptr vaddr = mv->vbar + page_nr * VBAR_PAGE_SIZE;

    do_free = do_free && rp->handle && (do_unpin || !rp->pinned);
    if (do_free) {
        CHECK_CU(cuMemUnmap(vaddr, VBAR_PAGE_SIZE));
        CHECK_CU(cuMemRelease(rp->handle));
        total_vram_usage -= VBAR_PAGE_SIZE;
        rp->handle = 0;
        mv->resident_count--;
    }
    if (do_unpin) {
        rp->pinned = false;
    }
    return do_free;
}

void vbars_free(size_t size) {
    size_t pages_needed = VBAR_GET_PAGE_NR_UP(size);

    one_time_setup();
    vbars_dirty = true;

    if (!size) {
        return;
    }

    CHECK_CU(cuCtxSynchronize());

    for (ModelVBAR *i = lowest_priority.higher; pages_needed && i != &highest_priority;
         i = i->higher) {
        for (;pages_needed && i->watermark > i->watermark_limit; i->watermark--) {
            if (mod1(i, i->watermark - 1, true, false)) {
                pages_needed--;
            }
        }
    }
}

static inline size_t move_cursor_to_absent(ModelVBAR *mv, size_t cursor) {
    while (cursor < mv->watermark && mv->residency_map[cursor].handle) {
        cursor++;
    }
    return cursor;
}

static void vbars_free_for_vbar(ModelVBAR *mv, size_t target) {
    size_t cursor = move_cursor_to_absent(mv, 0);

    CHECK_CU(cuCtxSynchronize());

    for (ModelVBAR *i = lowest_priority.higher;
         cursor < target && cursor < mv->watermark && i != &highest_priority;
         i = i->higher) {
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

    one_time_setup();
    log_reset_shots();
    log(DEBUG, "%s (start): size=%zuM, device=%d\n", __func__, size / M, device);
    vbars_dirty = true;

    size_t nr_pages = VBAR_GET_PAGE_NR_UP(size);
    size_t nr_pages_max = VBAR_GET_PAGE_NR(vram_capacity);
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
    
    insert_vbar(mv);

    log(DEBUG, "%s (return): vbar=%p\n", __func__, (void *)mv);
    return mv;
}

SHARED_EXPORT
void vbar_set_watermark_limit(void *vbar, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    log(DEBUG, "%s: size=%zu\n", __func__, size);
    mv->watermark_limit = VBAR_GET_PAGE_NR_UP(size);
}

SHARED_EXPORT
void vbars_reset_watermark_limits() {
    one_time_setup();
    log(VERBOSE, "%s\n", __func__);

    for (ModelVBAR *i = lowest_priority.higher; i && i != &highest_priority; i = i->higher) {
        i->watermark_limit = 0;
    }
}

SHARED_EXPORT
void vbar_prioritize(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    log(DEBUG, "%s vbar=%p\n", __func__, vbar);
    vbars_dirty = true;

    log_reset_shots();

    remove_vbar(mv);
    insert_vbar(mv);

    mv->watermark = mv->nr_pages;
}

SHARED_EXPORT
void vbar_deprioritize(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    log(DEBUG, "%s vbar=%p\n", __func__, vbar);
    vbars_dirty = true;

    log_reset_shots();

    remove_vbar(mv);
    insert_vbar_last(mv);
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
    vbars_dirty = true;

    if (page_end > mv->watermark) {
        log(VVERBOSE, "VBAR Allocation is above watermark\n");
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

        if (wddm_budget_deficit(mv->device, VBAR_PAGE_SIZE) ||
            (err = three_stooges(vaddr, VBAR_PAGE_SIZE, mv->device, &rp->handle)) != CUDA_SUCCESS) {
            if (err != CUDA_ERROR_OUT_OF_MEMORY) {
                log(ERROR, "VRAM Allocation failed (non OOM)\n");
                return VBAR_FAULT_ERROR;
            }
            log(DEBUG, "VBAR allocator attempt exceeds available VRAM ...\n");
            vbars_free_for_vbar(mv, page_end);
            if (page_nr >= mv->watermark) {
                log(DEBUG, "VBAR allocation cancelled due to watermark reduction\n");
                return VBAR_FAULT_OOM;
            }
            if ((err = three_stooges(vaddr, VBAR_PAGE_SIZE, mv->device, &rp->handle)) != CUDA_SUCCESS) {
                log(ERROR, "VRAM Allocation failed\n");
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

    log(VVERBOSE, "%s (return) %d\n", __func__, ret);
    return ret;
}

SHARED_EXPORT
void vbar_unpin(void *vbar, uint64_t offset, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    log(VVERBOSE, "%s (start): offset=%lldk, size=%lldk\n", __func__, (ull)(offset / K), (ull)(size / K));
    vbars_dirty = true;
    size_t page_end = VBAR_GET_PAGE_NR_UP(offset + size);

    if (page_end > mv->watermark) {
        CHECK_CU(cuCtxSynchronize());
    }

    for (uint64_t page_nr = VBAR_GET_PAGE_NR(offset); page_nr < page_end && page_nr < mv->nr_pages; page_nr++) {
        mod1(mv, page_nr, page_nr >= mv->watermark, true);
    }
}

SHARED_EXPORT
void vbar_free(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    log(DEBUG, "%s: vbar=%p\n", __func__, vbar);
    vbars_dirty = true;

    CHECK_CU(cuCtxSynchronize());

    for (uint64_t page_nr = 0; page_nr < mv->nr_pages; page_nr++) {
        mod1(mv, page_nr, true, true);
    }
    remove_vbar(mv);
    free(mv);
}

SHARED_EXPORT
size_t vbar_loaded_size(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    return mv->resident_count * VBAR_PAGE_SIZE;
}

SHARED_EXPORT
uint64_t vbar_free_memory(void *vbar, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t pages_to_free = VBAR_GET_PAGE_NR_UP(size);
    size_t pages_freed = 0;

    log(DEBUG, "%s (start): size=%lldk\n", __func__, (ull)size);
    vbars_dirty = true;

    CHECK_CU(cuCtxSynchronize());

    for (;pages_to_free && mv->watermark > mv->watermark_limit; mv->watermark--) {
        /* In theory we should never have pins here, but
         * respect pins if it really comes up.
         */
        if (mod1(mv, mv->watermark - 1, true, false)) {
            pages_to_free--;
            pages_freed++;
        }
    }

    return (uint64_t)pages_freed * VBAR_PAGE_SIZE;
}
