#include "plat.h"

#define VBAR_PAGE_SIZE (32 << 20)

#define VBAR_GET_PAGE_NR(x) ((x) / VBAR_PAGE_SIZE)
#define VBAR_GET_PAGE_NR_UP(x) VBAR_GET_PAGE_NR((x) + VBAR_PAGE_SIZE - 1)

typedef struct ResidentPage {
    CUmemGenericAllocationHandle handle;
    bool pinned;
} ResidentPage;

typedef struct ModelVBAR {
    CUdeviceptr vbar;
    size_t nr_pages;
    size_t watermark;

    int device;

    void *higher;
    void *lower;

    ResidentPage residency_map[1]; /* Must be last! */
} ModelVBAR;

ModelVBAR highest_priority;
ModelVBAR lowest_priority;

static inline bool mod1(ModelVBAR *mv, size_t page_nr, bool do_free, bool do_unpin) {
    ResidentPage *rp = &mv->residency_map[page_nr];
    CUdeviceptr vaddr = mv->vbar + page_nr * VBAR_PAGE_SIZE;

    do_free = do_free && rp->handle && (do_unpin || !rp->pinned);
    if (do_free) {
        CHECK_CU(cuMemUnmap(vaddr, VBAR_PAGE_SIZE));
        CHECK_CU(cuMemRelease(rp->handle));
        rp->handle = 0;
    }
    if (do_unpin) {
        rp->pinned = false;
    }
    return do_free;
}

void vbars_free(size_t size) {
    size_t pages_needed = VBAR_GET_PAGE_NR_UP(size);

    for (ModelVBAR *i = lowest_priority.higher; pages_needed && i != &highest_priority;
         i = i->higher) {
        for (;pages_needed && i->watermark; i->watermark--) {
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

static void vbars_free_for_vbar(ModelVBAR *mv) {
    size_t cursor = move_cursor_to_absent(mv, 0);

    for (ModelVBAR *i = lowest_priority.higher; cursor < mv->watermark && i != &highest_priority;
         i = i->higher) {
        for (; cursor < mv->watermark && i->watermark; i->watermark--) {
            if (mod1(i, i->watermark - 1, true, false)) {
                cursor = move_cursor_to_absent(mv, cursor + 1);
            }
        }
    }
}

static inline void one_time_setup() {
    if (!highest_priority.lower) {
        assert(!lowest_priority.higher);
        highest_priority.lower = &lowest_priority;
        lowest_priority.higher = &highest_priority;
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

SHARED_EXPORT
void *vbar_allocate(uint64_t size, int device) {
    ModelVBAR *mv;

    one_time_setup();
    log(DEBUG, "%s (start): size=%zu, device=%d\n", __func__, size, device);

    size_t nr_pages = VBAR_GET_PAGE_NR_UP(size);
    size = (uint64_t)nr_pages * VBAR_PAGE_SIZE;

    if (!(mv = calloc(1, sizeof(*mv) + nr_pages * sizeof(mv->residency_map[0])))) {
        log(CRITICAL, "Host OOM\n");
        return NULL;
    }

    /* FIXME: Do I care about alignment? Does Cuda just look after itself? */
    if (!CHECK_CU(cuMemAddressReserve(&mv->vbar, size, 0, 0, 0))) {
        log(ERROR, "Could not reseve Virtual Address space for VBAR");
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
void vbar_prioritize(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    log(DEBUG, "%s vbar=%p", __func__);

    remove_vbar(mv);
    insert_vbar(mv);

    mv->watermark = mv->nr_pages;
}

SHARED_EXPORT
uint64_t vbar_get(void *vbar) {
    log(DEBUG, "%s vbar=%p", __func__);
    return (uint64_t)((ModelVBAR *)vbar)->vbar;
}

#define VBAR_FAULT_SUCCESS  0
#define VBAR_FAULT_OOM      1
#define VBAR_FAULT_ERROR    2

SHARED_EXPORT
int vbar_fault(void *vbar, uint64_t offset, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    size_t page_end = VBAR_GET_PAGE_NR_UP(offset + size);
    if (page_end > mv->watermark) {
        return VBAR_FAULT_OOM;
    }

    for (uint64_t page_nr = VBAR_GET_PAGE_NR(offset); page_nr < page_end; page_nr++) {
        CUresult err;
        CUdeviceptr vaddr = mv->vbar + page_nr * VBAR_PAGE_SIZE;
        ResidentPage *rp = &mv->residency_map[page_nr];

        if (rp->handle) {
            continue;
        }

        if ((err = three_stooges(vaddr, VBAR_PAGE_SIZE, mv->device, &rp->handle)) != CUDA_SUCCESS) {
            if (err != CUDA_ERROR_OUT_OF_MEMORY) {
                return VBAR_FAULT_ERROR;
            }
            fprintf(stderr, "DEBUG: OOMED\n");
            vbars_free_for_vbar(mv);
            /* vbars free is allowed to come after us too */
            if (page_nr >= mv->watermark) {
                return VBAR_FAULT_OOM;
            }
            if (three_stooges(vaddr, VBAR_PAGE_SIZE, mv->device, &rp->handle) != CUDA_SUCCESS) {
                return VBAR_FAULT_ERROR;
            }
        }
    }

    /* We got our allocation */

    for (uint64_t page_nr = VBAR_GET_PAGE_NR(offset); page_nr < page_end; page_nr++) {
        ResidentPage *rp = &mv->residency_map[page_nr];
        rp->pinned = true;
    }

    return VBAR_FAULT_SUCCESS;
}

void vbar_unpin(void *vbar, uint64_t offset, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    size_t page_end = VBAR_GET_PAGE_NR_UP(offset + size);

    for (uint64_t page_nr = VBAR_GET_PAGE_NR(offset); page_nr < page_end; page_nr++) {
        mod1(mv, page_nr, page_end > mv->watermark, true);
    }
}

SHARED_EXPORT
void vbar_free(void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    for (uint64_t page_nr = 0; page_nr < mv->nr_pages; page_nr++) {
        mod1(mv, page_nr, true, true);
    }
}
