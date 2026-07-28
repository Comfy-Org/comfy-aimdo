#include "plat.h"

#include <stdarg.h>

enum RecordStatus {
    RECORD_OK = 0,
    RECORD_INVALID_STATE,
    RECORD_MISMATCH,
    RECORD_UNSUPPORTED,
    RECORD_CUDA_FAILURE,
    RECORD_OUT_OF_MEMORY,
};

enum RecordEventType {
    RECORD_ALLOC,
    RECORD_FREE,
    RECORD_CHILD,
};

typedef struct RecordPage {
    CUmemGenericAllocationHandle handle;
    struct RecordPage *next;
    struct RecordPage *next_free;
    bool released;
} RecordPage;

typedef struct RecordAllocation {
    CUdeviceptr ptr;
    size_t size;
    size_t mapped_size;
    size_t page_count;
    size_t mapped_count;
    size_t unmapped_count;
    RecordPage **pages;
    bool live;
    bool address_freed;
    struct RecordFrame *frame;
    struct RecordAllocation *next;
    struct RecordAllocation *hash_next;
} RecordAllocation;

typedef struct RecordEvent {
    enum RecordEventType type;
    union {
        RecordAllocation *allocation;
        struct RecordFrame *child;
    } item;
} RecordEvent;

typedef struct RecordRoot {
    AimdoContext *devctx;
    CUcontext context;
    CUstream stream;
    RecordPage *pages;
    RecordPage *free_pages;
    RecordAllocation *allocations;
    RecordAllocation *allocation_table[SIZE_HASH_SIZE];
    bool poisoned;
    bool draining;
    bool active;
    char error[256];
} RecordRoot;

typedef struct RecordFrame {
    RecordRoot *root;
    struct RecordFrame *parent;
    RecordEvent *events;
    size_t event_count;
    size_t event_capacity;
    size_t cursor;
    size_t iteration;
    size_t recorded_iterations;
    size_t live_count;
    bool compiled;
    bool active_iteration;
} RecordFrame;

static _Thread_local RecordFrame *g_record_frame;
static _Thread_local char g_record_error[256];

static int record_error(RecordRoot *root, int status, const char *format, ...) {
    va_list args;
    char *error = root ? root->error : g_record_error;

    if (root && root->poisoned) {
        return status;
    }

    va_start(args, format);
    vsnprintf(error, 256, format, args);
    va_end(args);
    if (root) {
        root->poisoned = true;
    }
    return status;
}

static size_t record_depth(RecordFrame *frame) {
    size_t depth = 0;

    while (frame->parent) {
        depth++;
        frame = frame->parent;
    }
    return depth;
}

static bool append_event(RecordFrame *frame, RecordEvent event) {
    if (frame->event_count == frame->event_capacity) {
        size_t capacity = frame->event_capacity ? frame->event_capacity * 2 : 32;
        RecordEvent *events = realloc(frame->events, capacity * sizeof(*events));

        if (!events) {
            record_error(frame->root, RECORD_OUT_OF_MEMORY,
                         "depth %zu iteration %zu: could not grow allocation trace",
                         record_depth(frame), frame->iteration);
            return false;
        }
        frame->events = events;
        frame->event_capacity = capacity;
    }

    frame->events[frame->event_count++] = event;
    return true;
}

static RecordEvent *next_event(RecordFrame *frame, enum RecordEventType type) {
    RecordEvent *event;

    if (frame->cursor == frame->event_count) {
        record_error(frame->root, RECORD_MISMATCH,
                     "depth %zu iteration %zu event %zu: unexpected operation",
                     record_depth(frame), frame->iteration, frame->cursor);
        return NULL;
    }

    event = &frame->events[frame->cursor];
    if (event->type != type) {
        record_error(frame->root, RECORD_MISMATCH,
                     "depth %zu iteration %zu event %zu: operation type changed",
                     record_depth(frame), frame->iteration, frame->cursor);
        return NULL;
    }
    frame->cursor++;
    return event;
}

static bool make_page(RecordRoot *root, RecordPage **page_out, CUresult *cuda_status) {
    CUmemAllocationProp prop = {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .location.type = CU_MEM_LOCATION_TYPE_DEVICE,
        .location.id = root->devctx->_device_id,
    };
    RecordPage *page = malloc(sizeof(*page));
    CUresult status;

    if (!page) {
        *cuda_status = CUDA_ERROR_OUT_OF_MEMORY;
        record_error(root, RECORD_OUT_OF_MEMORY, "could not allocate page metadata");
        return false;
    }
    memset(page, 0, sizeof(*page));

    status = cuMemCreate(&page->handle, CUDA_PAGE_SIZE, &prop, 0);
    if (status != CUDA_SUCCESS) {
        *cuda_status = status;
        free(page);
        record_error(root, status == CUDA_ERROR_OUT_OF_MEMORY ? RECORD_OUT_OF_MEMORY : RECORD_CUDA_FAILURE,
                     "cuMemCreate failed with status %d", status);
        return false;
    }

    page->next = root->pages;
    root->pages = page;
    total_vram_usage += CUDA_PAGE_SIZE;
    *page_out = page;
    return true;
}

static bool map_page(RecordRoot *root, CUdeviceptr ptr, RecordPage *page, bool *mapped,
                     CUresult *cuda_status) {
    CUmemAccessDesc access = {
        .location.type = CU_MEM_LOCATION_TYPE_DEVICE,
        .location.id = root->devctx->_device_id,
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    CUresult status;

    *mapped = false;
    status = cuMemMap(ptr, CUDA_PAGE_SIZE, 0, page->handle, 0);
    if (status == CUDA_SUCCESS) {
        *mapped = true;
        status = cuMemSetAccess(ptr, CUDA_PAGE_SIZE, &access, 1);
    }
    if (status != CUDA_SUCCESS) {
        *cuda_status = status;
        record_error(root, RECORD_CUDA_FAILURE, "mapping allocation page failed with status %d", status);
        return false;
    }
    return true;
}

static unsigned int allocation_hash(CUdeviceptr ptr) {
    return ((uintptr_t)ptr >> 10 ^ (uintptr_t)ptr >> 21) % SIZE_HASH_SIZE;
}

static RecordAllocation *find_allocation(RecordRoot *root, CUdeviceptr ptr) {
    RecordAllocation *allocation = root->allocation_table[allocation_hash(ptr)];

    while (allocation && allocation->ptr != ptr) {
        allocation = allocation->hash_next;
    }
    return allocation;
}

static bool record_context_matches(RecordRoot *root) {
    CUcontext context;

    return cuCtxGetCurrent(&context) == CUDA_SUCCESS && context == root->context;
}

static void release_frame(RecordFrame *frame) {
    for (size_t i = 0; i < frame->event_count; i++) {
        if (frame->events[i].type == RECORD_CHILD) {
            release_frame(frame->events[i].item.child);
        }
    }
    free(frame->events);
    free(frame);
}

static bool release_root(RecordRoot *root) {
    RecordAllocation *allocation = root->allocations;
    RecordPage *page = root->pages;
    AimdoContext *previous = g_devctx;
    CUresult status;

    set_devctx(root->devctx);
    while (allocation) {
        while (allocation->unmapped_count < allocation->mapped_count) {
            CUdeviceptr ptr = allocation->ptr + allocation->unmapped_count * CUDA_PAGE_SIZE;

            status = cuMemUnmap(ptr, CUDA_PAGE_SIZE);
            if (status != CUDA_SUCCESS) {
                snprintf(root->error, sizeof(root->error),
                         "cuMemUnmap failed with status %d; pop may be retried", status);
                set_devctx(previous);
                return false;
            }
            unmap_workaround(ptr, CUDA_PAGE_SIZE);
            allocation->unmapped_count++;
        }
        if (!allocation->address_freed) {
            status = cuMemAddressFree(allocation->ptr, allocation->mapped_size);
            if (status != CUDA_SUCCESS) {
                snprintf(root->error, sizeof(root->error),
                         "cuMemAddressFree failed with status %d; pop may be retried", status);
                set_devctx(previous);
                return false;
            }
            allocation->address_freed = true;
        }
        allocation = allocation->next;
    }
    while (page) {
        if (!page->released) {
            status = cuMemRelease(page->handle);
            if (status != CUDA_SUCCESS) {
                snprintf(root->error, sizeof(root->error),
                         "cuMemRelease failed with status %d; pop may be retried", status);
                set_devctx(previous);
                return false;
            }
            page->released = true;
            total_vram_usage -= CUDA_PAGE_SIZE;
        }
        page = page->next;
    }
    set_devctx(previous);

    allocation = root->allocations;
    while (allocation) {
        RecordAllocation *next = allocation->next;

        free(allocation->pages);
        free(allocation);
        allocation = next;
    }
    page = root->pages;
    while (page) {
        RecordPage *next = page->next;

        free(page);
        page = next;
    }
    free(root);
    return true;
}

static int finish_iteration(RecordFrame *frame) {
    if (!frame->active_iteration) {
        return record_error(frame->root, RECORD_INVALID_STATE,
                            "depth %zu: iterate was not called", record_depth(frame));
    }
    if (frame->live_count) {
        return record_error(frame->root, RECORD_MISMATCH,
                            "depth %zu iteration %zu: %zu allocations remain live",
                            record_depth(frame), frame->iteration, frame->live_count);
    }
    if (frame->compiled && frame->cursor != frame->event_count) {
        return record_error(frame->root, RECORD_MISMATCH,
                            "depth %zu iteration %zu: expected %zu more operations",
                            record_depth(frame), frame->iteration,
                            frame->event_count - frame->cursor);
    }
    if (!frame->compiled) {
        frame->compiled = true;
    }
    return RECORD_OK;
}

SHARED_EXPORT
int push_record(CUstream stream, void *graph) {
    RecordFrame *frame;
    RecordRoot *root;

    g_record_error[0] = '\0';
    if (!g_record_frame) {
        if (!set_devctx_for_current_cuda_device()) {
            return record_error(NULL, RECORD_INVALID_STATE, "no initialized CUDA device is current");
        }
        if (graph) {
            frame = graph;
            root = frame->root;
            if (frame->parent || root->active || root->draining) {
                return record_error(NULL, RECORD_INVALID_STATE, "allocation graph is not available");
            }
            if (root->stream != stream || root->devctx != g_devctx ||
                !record_context_matches(root)) {
                return record_error(NULL, RECORD_UNSUPPORTED,
                                    "allocation graph stream or CUDA context changed");
            }
            if (root->poisoned) {
                return record_error(NULL, RECORD_MISMATCH, "allocation graph is poisoned");
            }
        } else {
            root = calloc(1, sizeof(*root));
            frame = calloc(1, sizeof(*frame));
            if (!root || !frame) {
                free(root);
                free(frame);
                return record_error(NULL, RECORD_OUT_OF_MEMORY, "could not create allocation record");
            }
            root->devctx = g_devctx;
            if (cuCtxGetCurrent(&root->context) != CUDA_SUCCESS || !root->context) {
                free(root);
                free(frame);
                return record_error(NULL, RECORD_INVALID_STATE, "no CUDA context is current");
            }
            root->stream = stream;
            frame->root = root;
        }
        root->active = true;
        g_record_frame = frame;
        return RECORD_OK;
    }

    if (graph) {
        return record_error(g_record_frame->root, RECORD_INVALID_STATE,
                            "nested allocation record cannot select a graph");
    }

    if (g_record_frame->root->stream != stream) {
        return record_error(g_record_frame->root, RECORD_UNSUPPORTED,
                            "nested allocation record must use the outer stream");
    }
    if (!g_record_frame->active_iteration) {
        return record_error(g_record_frame->root, RECORD_INVALID_STATE,
                            "nested allocation record is outside an iteration");
    }

    if (!g_record_frame->compiled) {
        frame = calloc(1, sizeof(*frame));
        if (!frame) {
            return record_error(g_record_frame->root, RECORD_OUT_OF_MEMORY,
                                "could not create nested allocation record");
        }
        frame->root = g_record_frame->root;
        frame->parent = g_record_frame;
        if (!append_event(g_record_frame, (RecordEvent){ .type = RECORD_CHILD, .item.child = frame })) {
            free(frame);
            return RECORD_OUT_OF_MEMORY;
        }
    } else {
        RecordEvent *event = next_event(g_record_frame, RECORD_CHILD);

        if (!event) {
            return RECORD_MISMATCH;
        }
        frame = event->item.child;
        frame->parent = g_record_frame;
    }

    frame->cursor = 0;
    frame->iteration = 0;
    frame->active_iteration = false;
    g_record_frame = frame;
    return RECORD_OK;
}

SHARED_EXPORT
int iterate(void) {
    RecordFrame *frame = g_record_frame;
    int status;

    if (!frame) {
        return record_error(NULL, RECORD_INVALID_STATE, "no allocation record is active");
    }
    if (frame->root->poisoned) {
        return RECORD_MISMATCH;
    }
    if (!record_context_matches(frame->root)) {
        return record_error(frame->root, RECORD_UNSUPPORTED,
                            "the CUDA context changed during allocation recording");
    }
    if (frame->active_iteration && (status = finish_iteration(frame)) != RECORD_OK) {
        return status;
    }

    frame->iteration++;
    frame->cursor = 0;
    frame->active_iteration = true;
    return RECORD_OK;
}

SHARED_EXPORT
int pop(void **graph) {
    RecordFrame *frame = g_record_frame;
    RecordRoot *root;
    int status;

    if (!graph) {
        return record_error(frame ? frame->root : NULL, RECORD_INVALID_STATE,
                            "pop requires an allocation graph output");
    }
    *graph = NULL;
    if (!frame) {
        return record_error(NULL, RECORD_INVALID_STATE, "no allocation record is active");
    }
    root = frame->root;
    if (!record_context_matches(root)) {
        snprintf(root->error, sizeof(root->error),
                 "the CUDA context changed during allocation recording");
        return RECORD_UNSUPPORTED;
    }
    if (root->draining) {
        status = root->poisoned ? RECORD_MISMATCH : RECORD_OK;
    } else if (root->poisoned) {
        status = RECORD_MISMATCH;
    } else {
        status = finish_iteration(frame);
    }
    if (!root->draining && status == RECORD_OK && frame->parent) {
        if (!frame->recorded_iterations) {
            frame->recorded_iterations = frame->iteration;
        } else if (frame->recorded_iterations != frame->iteration) {
            status = record_error(root, RECORD_MISMATCH,
                                  "depth %zu: expected %zu iterations, got %zu",
                                  record_depth(frame), frame->recorded_iterations, frame->iteration);
        }
    }

    if (!root->draining) {
        frame->active_iteration = false;
    }
    if (frame->parent) {
        g_record_frame = frame->parent;
        return status;
    }

    g_record_frame = NULL;
    root->active = false;
    *graph = frame;
    if (root->error[0]) {
        snprintf(g_record_error, sizeof(g_record_error), "%s", root->error);
    }
    return status;
}

SHARED_EXPORT
int destroy_record(void *graph) {
    RecordFrame *frame = graph;
    RecordRoot *root;

    if (!frame || frame->parent || frame->root->active) {
        return record_error(NULL, RECORD_INVALID_STATE, "allocation graph is not available for destruction");
    }
    root = frame->root;
    if (!record_context_matches(root)) {
        return record_error(NULL, RECORD_UNSUPPORTED,
                            "allocation graph CUDA context changed");
    }
    if (cuCtxSynchronize() != CUDA_SUCCESS) {
        snprintf(g_record_error, sizeof(g_record_error),
                 "could not synchronize the allocation graph context; destruction may be retried");
        return RECORD_CUDA_FAILURE;
    }
    if (!release_root(root)) {
        root->draining = true;
        snprintf(g_record_error, sizeof(g_record_error), "%s", root->error);
        return RECORD_CUDA_FAILURE;
    }
    release_frame(frame);
    return RECORD_OK;
}

SHARED_EXPORT
const char *record_last_error(void) {
    if (g_record_frame && g_record_frame->root->error[0]) {
        return g_record_frame->root->error;
    }
    return g_record_error[0] ? g_record_error : NULL;
}

bool record_malloc_async(CUdeviceptr *dev_ptr, size_t size, CUstream stream,
                         CUresult *status) {
    RecordFrame *frame = g_record_frame;
    RecordRoot *root;
    RecordAllocation *allocation;
    RecordEvent *event;

    if (!frame || frame->root->stream != stream) {
        return false;
    }
    root = frame->root;
    *status = CUDA_ERROR_INVALID_VALUE;
    *dev_ptr = 0;
    if (root->devctx != g_devctx || !record_context_matches(root) || root->poisoned ||
        !frame->active_iteration || !size) {
        record_error(root, RECORD_INVALID_STATE, "allocation record is not ready for an allocation");
        return true;
    }

    if (frame->compiled) {
        event = next_event(frame, RECORD_ALLOC);
        if (!event) {
            return true;
        }
        allocation = event->item.allocation;
        if (size > allocation->mapped_size || allocation->live) {
            record_error(root, RECORD_MISMATCH,
                         "depth %zu iteration %zu event %zu: allocation size %zu exceeds recorded capacity %zu",
                         record_depth(frame), frame->iteration, frame->cursor - 1,
                         size, allocation->mapped_size);
            return true;
        }
        allocation->live = true;
        frame->live_count++;
        *dev_ptr = allocation->ptr;
        *status = CUDA_SUCCESS;
        return true;
    }

    allocation = calloc(1, sizeof(*allocation));
    if (!allocation) {
        *status = CUDA_ERROR_OUT_OF_MEMORY;
        record_error(root, RECORD_OUT_OF_MEMORY, "could not allocate allocation metadata");
        return true;
    }
    allocation->size = size;
    allocation->mapped_size = CUDA_ALIGN_UP(size);
    allocation->page_count = allocation->mapped_size / CUDA_PAGE_SIZE;
    allocation->frame = frame;
    allocation->pages = calloc(allocation->page_count, sizeof(*allocation->pages));
    if (!allocation->pages) {
        *status = CUDA_ERROR_OUT_OF_MEMORY;
        free(allocation);
        record_error(root, RECORD_OUT_OF_MEMORY, "could not allocate page metadata");
        return true;
    }
    *status = cuMemAddressReserve(&allocation->ptr, allocation->mapped_size,
                                  CUDA_PAGE_SIZE, 0, 0);
    if (*status != CUDA_SUCCESS) {
        free(allocation->pages);
        free(allocation);
        record_error(root, *status == CUDA_ERROR_OUT_OF_MEMORY ? RECORD_OUT_OF_MEMORY : RECORD_CUDA_FAILURE,
                     "could not reserve allocation address space (status %d)", *status);
        return true;
    }
    allocation->next = root->allocations;
    root->allocations = allocation;
    allocation->hash_next = root->allocation_table[allocation_hash(allocation->ptr)];
    root->allocation_table[allocation_hash(allocation->ptr)] = allocation;

    for (size_t i = 0; i < allocation->page_count; i++) {
        RecordPage *page = root->free_pages;
        bool mapped;

        if (page) {
            root->free_pages = page->next_free;
            page->next_free = NULL;
        } else if (!make_page(root, &page, status)) {
            return true;
        }
        allocation->pages[i] = page;
        if (!map_page(root, allocation->ptr + i * CUDA_PAGE_SIZE, page, &mapped, status)) {
            allocation->mapped_count += mapped;
            return true;
        }
        allocation->mapped_count++;
    }

    allocation->live = true;
    if (!append_event(frame, (RecordEvent){ .type = RECORD_ALLOC, .item.allocation = allocation })) {
        *status = CUDA_ERROR_OUT_OF_MEMORY;
        return true;
    }
    frame->live_count++;
    *dev_ptr = allocation->ptr;
    *status = CUDA_SUCCESS;
    return true;
}

bool record_free(CUdeviceptr dev_ptr, CUstream stream, bool is_async,
                 CUresult *status) {
    RecordFrame *frame = g_record_frame;
    RecordRoot *root;
    RecordAllocation *allocation;
    RecordEvent *event;

    if (!frame) {
        return false;
    }
    root = frame->root;
    allocation = find_allocation(root, dev_ptr);
    if (!allocation && (!is_async || root->stream != stream)) {
        return false;
    }
    if (!allocation) {
        return false;
    }

    *status = CUDA_ERROR_INVALID_VALUE;
    if (!is_async || root->stream != stream) {
        record_error(root, RECORD_UNSUPPORTED, "compiled allocation freed outside its record stream");
        *status = CUDA_SUCCESS;
        return true;
    }
    if (root->devctx != g_devctx || !record_context_matches(root) || root->poisoned ||
        !frame->active_iteration ||
        allocation->frame != frame || !allocation->live) {
        record_error(root, RECORD_MISMATCH, "allocation free does not match a live allocation in this frame");
        return true;
    }

    if (frame->compiled) {
        event = next_event(frame, RECORD_FREE);
        if (!event || event->item.allocation != allocation) {
            if (event) {
                record_error(root, RECORD_MISMATCH,
                             "depth %zu iteration %zu event %zu: freed allocation changed",
                             record_depth(frame), frame->iteration, frame->cursor - 1);
            }
            return true;
        }
    } else if (!append_event(frame, (RecordEvent){ .type = RECORD_FREE, .item.allocation = allocation })) {
        return true;
    }

    allocation->live = false;
    frame->live_count--;
    if (!frame->compiled) {
        for (size_t i = 0; i < allocation->page_count; i++) {
            allocation->pages[i]->next_free = root->free_pages;
            root->free_pages = allocation->pages[i];
        }
    }
    *status = CUDA_SUCCESS;
    return true;
}

void record_cleanup(void) {
    RecordFrame *frame = g_record_frame;

    if (!frame) {
        return;
    }
    while (frame->parent) {
        frame = frame->parent;
    }
    if (!record_context_matches(frame->root) || cuCtxSynchronize() != CUDA_SUCCESS) {
        g_record_frame = NULL;
        log(ERROR, "%s: leaking an active allocation record because its context could not be synchronized\n",
            __func__);
        return;
    }
    g_record_frame = NULL;
    if (!release_root(frame->root)) {
        log(ERROR, "%s: leaking an allocation record after VMM teardown failed\n", __func__);
        return;
    }
    release_frame(frame);
}
