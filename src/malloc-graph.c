#include "plat.h"

#define MG_PAGE (8ULL * M)
#define MG_PAGES 8192ULL
#define MG_SMALL_PAGES (MG_PAGES / 8)
#define MG_SMALL_LIMIT (4ULL * M)
#define MG_SMALL_ALIGN 4096ULL
#define RETURN_G_FAILED(cond, retval) \
    if (cond) { \
        if (g) { \
            g->failed = true; \
        } \
        return retval; \
    }

typedef enum {
    EV_SENTINEL = 0,
    EV_ALLOC,
    EV_FREE,
    EV_ALLOC_SMALL,
    EV_FREE_SMALL,
    EV_CALL,
} EventType;

typedef struct Event Event;
typedef struct State State;
typedef struct AllocationState AllocationState;

struct Event {
    EventType type;
    union {
        struct {
            size_t value;
            size_t bytes;
        };
        struct {
            Event *scope;
            char *name;
        };
    };

    Event *next;
    Event *previous;
    AllocationState *snapshot;
};

struct State {
    Event *scope;
    Event *cursor;
    size_t live;
    uint32_t depth;
    bool recording;

    State *next;
};

typedef struct {
    uint16_t va_span;
    uint32_t owner;
} VirtualPage;

typedef struct {
    size_t offset;
    size_t bytes;
    uint32_t owner;
} SmallRange;

struct AllocationState {
    VirtualPage virtual_pages[MG_PAGES];
    bool physical_live[MG_PAGES];
    size_t small_count;
    SmallRange small_ranges[];
};

typedef struct {
    CUdeviceptr base;
    CUdeviceptr small_base;
    CUstream stream;
    int device;

    Event root;
    State *state;

    AllocationState *allocations;
    size_t small_capacity;
    int va_phys[MG_PAGES];
    CUmemGenericAllocationHandle physical_handles[MG_PAGES];
    CUmemGenericAllocationHandle small_handles[MG_SMALL_PAGES];

    size_t va_count;
    size_t phys_count;
    size_t small_size;
    size_t small_pages;

    bool failed;
    bool complete;
    bool paused;
} MallocGraph;

static _Thread_local MallocGraph *active_graph;

static size_t allocation_state_size(size_t small_count) {
    return sizeof(AllocationState) + small_count * sizeof(SmallRange);
}

static bool reserve_small_ranges(MallocGraph *g, size_t count) {
    if (count <= g->small_capacity) {
        return true;
    }

    size_t capacity = MAX(count, g->small_capacity * 2);
    AllocationState *allocations = realloc(g->allocations, allocation_state_size(capacity));
    RETURN_G_FAILED(!allocations, false);
    g->allocations = allocations;
    g->small_capacity = capacity;
    return true;
}

static AllocationState *snapshot_allocations(MallocGraph *g) {
    size_t size = allocation_state_size(g->allocations->small_count);
    AllocationState *snapshot = malloc(size);
    RETURN_G_FAILED(!snapshot, NULL);
    memcpy(snapshot, g->allocations, size);
    return snapshot;
}

static bool restore_allocations(MallocGraph *g, AllocationState *snapshot) {
    RETURN_G_FAILED(!reserve_small_ranges(g, snapshot->small_count), false);
    memcpy(g->allocations, snapshot, allocation_state_size(snapshot->small_count));
    return true;
}

static Event *next_event(MallocGraph *g, const char *name) {
    Event *event = g->state->cursor;

    if (name && event->type == EV_CALL && !strcmp(event->name, name)) {
        return event;
    }
    for (event = event->next; event && event->type == EV_CALL; event = event->next) {
        if (name && !strcmp(event->name, name)) {
            return event;
        }
    }
    return name ? NULL : event;
}

static Event *event(MallocGraph *g, EventType type, size_t value, size_t bytes) {
    Event *e;

    if (g->state->recording) {
        e = calloc(1, sizeof(*e));
        RETURN_G_FAILED(!e, NULL);
        e->type = type;
        e->value = value;
        e->bytes = bytes;
        e->previous = g->state->cursor;
        g->state->cursor->next = e;
    } else {
        e = next_event(g, NULL);
        RETURN_G_FAILED(!e || e->type != type || e->value != value || e->bytes != bytes, NULL);
    }
    g->state->cursor = e;
    return e;
}

static bool push_stack(MallocGraph *g, Event *scope, bool recording) {
    State *state = malloc(sizeof(*state));
    RETURN_G_FAILED(!state, false);
    if (recording && !scope->snapshot) {
        scope->snapshot = snapshot_allocations(g);
        if (!scope->snapshot) {
            free(state);
            return false;
        }
    }
    *state = (State){.scope = scope, .cursor = scope, .depth = g->state ? g->state->depth + 1 : 1,
                     .recording = recording, .next = g->state};
    g->state = state;
    return true;
}

static CUresult create_page(MallocGraph *g, CUmemGenericAllocationHandle *handle) {
    CUmemAllocationProp prop = {.type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .location = {CU_MEM_LOCATION_TYPE_DEVICE, g->device}};
    CUresult r;

    vbars_free_stream(budget_deficit(MG_PAGE), g->stream);
    r = cuMemCreate(handle, MG_PAGE, &prop, 0);
    if (r == CUDA_ERROR_OUT_OF_MEMORY) {
        vbars_free_stream(MG_PAGE, g->stream);
        r = cuMemCreate(handle, MG_PAGE, &prop, 0);
    }
    if (!r) {
        total_vram_usage += MG_PAGE;
    }
    return r;
}

static int map_page(MallocGraph *g, size_t va, size_t phys) {
    CUmemAccessDesc access = {.location = {CU_MEM_LOCATION_TYPE_DEVICE, g->device},
                              .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    CUdeviceptr addr = g->base + va * MG_PAGE;
    CUresult r;

    if (phys == g->phys_count) {
        r = create_page(g, &g->physical_handles[phys]);
        if (r) {
            return r;
        }
        g->phys_count++;
    }

    if ((r = cuMemMap(addr, MG_PAGE, 0, g->physical_handles[phys], 0)) ||
        (r = cuMemSetAccess(addr, MG_PAGE, &access, 1))) {
        return r;
    }
    g->va_phys[va] = (int)phys;
    return 0;
}

static bool dry_apply(MallocGraph *g, Event *event) {
    AllocationState *allocations = g->allocations;
    if (event->type == EV_CALL) {
        return true;
    }
    size_t value = event->value;

    if (event->type == EV_ALLOC) {
        size_t pages = ALIGN_UP(event->bytes, MG_PAGE) / MG_PAGE;
        for (size_t i = 0; i < pages; i++) {
            allocations->physical_live[g->va_phys[value + i]] = true;
        }
        allocations->virtual_pages[value] = (VirtualPage){
            .va_span = pages, .owner = g->state->depth};
        g->state->live += pages;
    } else if (event->type == EV_FREE) {
        VirtualPage *first = &allocations->virtual_pages[value];
        for (size_t i = 0; i < first->va_span; i++) {
            allocations->physical_live[g->va_phys[value + i]] = false;
        }
        g->state->live -= first->va_span;
        *first = (VirtualPage){};
    } else if (event->type == EV_ALLOC_SMALL) {
        size_t entry = 0;
        while (entry < allocations->small_count && allocations->small_ranges[entry].offset < value) {
            entry++;
        }
        RETURN_G_FAILED(!reserve_small_ranges(g, allocations->small_count + 1), false);
        allocations = g->allocations;
        memmove(&allocations->small_ranges[entry + 1], &allocations->small_ranges[entry],
                (allocations->small_count - entry) * sizeof(SmallRange));
        allocations->small_ranges[entry] = (SmallRange){
            .offset = value, .bytes = ALIGN_UP(event->bytes, MG_SMALL_ALIGN),
            .owner = g->state->depth};
        allocations->small_count++;
        g->state->live++;
    } else if (event->type == EV_FREE_SMALL) {
        size_t entry = 0;
        while (allocations->small_ranges[entry].offset != value) {
            entry++;
        }
        memmove(&allocations->small_ranges[entry], &allocations->small_ranges[entry + 1],
                (allocations->small_count - entry - 1) * sizeof(SmallRange));
        allocations->small_count--;
        g->state->live--;
    }
    return true;
}

static bool materialize(MallocGraph *g) {
    State *state = g->state;
    size_t count = 0;

    for (Event *event = state->cursor; event != state->scope; event = event->previous) {
        count++;
    }

    Event **events = count ? malloc(count * sizeof(*events)) : NULL;
    RETURN_G_FAILED(count && !events, false);
    Event *event = state->cursor;
    for (size_t i = count; i > 0; i--) {
        events[i - 1] = event;
        event = event->previous;
    }

    if (!restore_allocations(g, state->scope->snapshot)) {
        free(events);
        return false;
    }
    state->live = 0;
    for (size_t i = 0; i < count; i++) {
        if (!dry_apply(g, events[i])) {
            free(events);
            return false;
        }
    }
    free(events);
    return true;
}

bool malloc_graph_alloc(CUdeviceptr *ptr, size_t size, CUstream stream) {
    MallocGraph *g = active_graph;

    if (!g || g->paused || stream != g->stream) {
        return false;
    }

    *ptr = 0;
    size_t va = 0;
    AllocationState *allocations = g->allocations;

    if (!g->state->recording) {
        EventType type = size < MG_SMALL_LIMIT ? EV_ALLOC_SMALL : EV_ALLOC;
        Event *e = next_event(g, NULL);
        RETURN_G_FAILED(!e || e->type != type || e->bytes != size, true);
        g->state->cursor = e;
        *ptr = type == EV_ALLOC_SMALL
            ? g->small_base + e->value
            : g->base + e->value * MG_PAGE;
        return true;
    }

    if (size < MG_SMALL_LIMIT) {
        size_t bytes = ALIGN_UP(size, MG_SMALL_ALIGN);
        size_t offset = g->small_size;
        size_t smallest_hole = SIZE_MAX;
        size_t previous_end = 0;
        size_t insert_at = allocations->small_count;

        for (size_t i = 0; i < allocations->small_count; i++) {
            SmallRange *range = &allocations->small_ranges[i];
            size_t hole = range->offset - previous_end;
            if (hole >= bytes && hole < smallest_hole) {
                offset = previous_end;
                smallest_hole = hole;
                insert_at = i;
            }
            previous_end = range->offset + range->bytes;
        }
        size_t hole = g->small_size - previous_end;
        if (hole >= bytes && hole < smallest_hole) {
            offset = previous_end;
            insert_at = allocations->small_count;
        }

        size_t end = offset + bytes;
        if (end > g->small_size) {
            size_t pages = ALIGN_UP(end, MG_PAGE) / MG_PAGE;
            RETURN_G_FAILED(pages > MG_SMALL_PAGES, true);

            CUmemAccessDesc access = {.location = {CU_MEM_LOCATION_TYPE_DEVICE, g->device},
                                      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
            if (g->small_pages < pages) {
                CUmemGenericAllocationHandle *handle = &g->small_handles[g->small_pages];
                CUdeviceptr addr = g->small_base + g->small_pages * MG_PAGE;
                CUresult r = create_page(g, handle);
                RETURN_G_FAILED(r, true);
                g->small_pages++;
                RETURN_G_FAILED(cuMemMap(addr, MG_PAGE, 0, *handle, 0) ||
                                cuMemSetAccess(addr, MG_PAGE, &access, 1), true);
            }
            g->small_size = end;
        }

        RETURN_G_FAILED(!reserve_small_ranges(g, allocations->small_count + 1), true);
        allocations = g->allocations;
        memmove(&allocations->small_ranges[insert_at + 1], &allocations->small_ranges[insert_at],
                (allocations->small_count - insert_at) * sizeof(SmallRange));
        allocations->small_ranges[insert_at] = (SmallRange){
            .offset = offset, .bytes = bytes, .owner = g->state->depth};
        allocations->small_count++;

        event(g, EV_ALLOC_SMALL, offset, size);
        g->state->live++;
        *ptr = g->small_base + offset;
        return true;
    }

    size_t pages = ALIGN_UP(size, MG_PAGE) / MG_PAGE;

    while (va + pages <= g->va_count) {
        size_t j;
        for (j = 0; j < pages; j++) {
            if (allocations->physical_live[g->va_phys[va + j]]) {
                break;
            }
        }
        if (j == pages) {
            break;
        }
        va += j + 1;
    }
    if (va + pages > g->va_count) {
        va = g->va_count;

        RETURN_G_FAILED(va + pages > MG_PAGES, true);

        g->va_count = va + pages;
    }

    for (size_t j = 0; j < pages; j++) {
        if (g->va_phys[va + j] < 0) {
            size_t p = 0;
            while (p < g->phys_count && allocations->physical_live[p]) {
                p++;
            }
            RETURN_G_FAILED(map_page(g, va + j, p), true);
        }
        allocations->physical_live[g->va_phys[va + j]] = true;
    }

    event(g, EV_ALLOC, va, size);
    allocations->virtual_pages[va].va_span = pages;
    allocations->virtual_pages[va].owner = g->state->depth;
    g->state->live += pages;
    *ptr = g->base + va * MG_PAGE;
    return true;
}

bool malloc_graph_free(CUdeviceptr ptr, size_t size, CUstream stream, int *result) {
    MallocGraph *g = active_graph;

    if (!g || g->paused || stream != g->stream) {
        return false;
    }

    AllocationState *allocations = g->allocations;
    bool small = ptr >= g->small_base && ptr < g->small_base + MG_SMALL_PAGES * MG_PAGE;
    if (!small && (ptr < g->base || ptr >= g->base + MG_PAGES * MG_PAGE)) {
        RETURN_G_FAILED(size, false);
        return false;
    }

    *result = 0;
    if (small) {
        size_t offset = ptr - g->small_base;

        if (g->state->recording) {
            size_t entry = 0;
            while (entry < allocations->small_count && allocations->small_ranges[entry].offset != offset) {
                entry++;
            }
            RETURN_G_FAILED(entry == allocations->small_count ||
                            allocations->small_ranges[entry].owner != g->state->depth ||
                            !event(g, EV_FREE_SMALL, offset, 0), true);

            memmove(&allocations->small_ranges[entry], &allocations->small_ranges[entry + 1],
                    (allocations->small_count - entry - 1) * sizeof(SmallRange));
            allocations->small_count--;
            g->state->live--;
        } else {
            event(g, EV_FREE_SMALL, offset, 0);
        }
        return true;
    }

    size_t va = (ptr - g->base) / MG_PAGE;

    if (g->state->recording) {
        VirtualPage *first = &allocations->virtual_pages[va];
        RETURN_G_FAILED(first->owner != g->state->depth || !first->va_span || !event(g, EV_FREE, va, 0), true);

        for (size_t j = 0; j < first->va_span; j++) {
            allocations->physical_live[g->va_phys[va + j]] = false;
        }
        g->state->live -= first->va_span;
        first->va_span = 0;
        first->owner = 0;
    } else {
        event(g, EV_FREE, va, 0);
    }
    return true;
}

SHARED_EXPORT void *malloc_graph_create(void *devctx, CUstream stream) {
    MallocGraph *g = calloc(1, sizeof(*g));

    if (!g || active_graph) {
        free(g);
        return NULL;
    }
    g->allocations = calloc(1, sizeof(*g->allocations));
    if (!g->allocations) {
        goto fail;
    }

    set_devctx(devctx);
    g->stream = stream;
    g->device = g_devctx->_device_id;

    if (cuMemAddressReserve(&g->base, MG_PAGES * MG_PAGE, MG_PAGE, 0, 0)) {
        goto fail;
    }
    if (cuMemAddressReserve(&g->small_base, MG_SMALL_PAGES * MG_PAGE, MG_PAGE, 0, 0)) {
        goto fail_address;
    }

    for (size_t i = 0; i < MG_PAGES; i++) {
        g->va_phys[i] = -1;
    }

    if (!push_stack(g, &g->root, true)) {
        goto fail_small_address;
    }
    active_graph = g;
    return g;

fail_small_address:
    cuMemAddressFree(g->small_base, MG_SMALL_PAGES * MG_PAGE);
fail_address:
    cuMemAddressFree(g->base, MG_PAGES * MG_PAGE);
fail:
    free(g->allocations);
    free(g);
    return NULL;
}

SHARED_EXPORT bool malloc_graph_pause(void *handle, bool paused) {
    MallocGraph *g = handle;

    if (!g || g != active_graph || g->failed) {
        return false;
    }
    g->paused = paused;
    return true;
}

SHARED_EXPORT int malloc_graph_push(void *handle, const char *name) {
    MallocGraph *g = handle;

    if (!g || g != active_graph || g->failed) {
        return 0;
    }

    Event *scope;
    bool recording = g->state->recording;

    if (recording) {
        recording = g->state->cursor->type != EV_CALL || strcmp(g->state->cursor->name, name);

        if (recording) {
            Event *call = calloc(1, sizeof(*call));
            scope = calloc(1, sizeof(*scope));

            if (!call || !scope) {
                free(call);
                free(scope);
                g->failed = true;
                return 0;
            }

            call->type = EV_CALL;
            call->scope = scope;
            call->name = strdup(name);
            call->previous = g->state->cursor;
            g->state->cursor->next = call;
            g->state->cursor = call;
        } else {
            scope = g->state->cursor->scope;
        }
    } else {
        Event *event = next_event(g, name);

        RETURN_G_FAILED(!event, 0);
        g->state->cursor = event;
        scope = event->scope;
    }

    if (!push_stack(g, scope, recording)) {
        return 0;
    }
    return recording ? 2 : 1;
}

SHARED_EXPORT bool malloc_graph_pop(void *handle) {
    MallocGraph *g = handle;

    if (!g || g != active_graph || g->failed) {
        return false;
    }

    RETURN_G_FAILED(g->state->live || (!g->state->recording && next_event(g, NULL)), false);

    State *state = g->state;
    g->state = state->next;
    free(state);

    if (!g->state) {
        g->complete = true;
        active_graph = NULL;
    }
    return true;
}

SHARED_EXPORT bool malloc_graph_replay(void *handle) {
    MallocGraph *g = handle;

    if (!g || g->failed || !g->complete || active_graph || !push_stack(g, &g->root, false)) {
        return false;
    }
    g->complete = false;
    active_graph = g;
    return true;
}

SHARED_EXPORT uint64_t malloc_graph_stat(void *handle, int which) {
    MallocGraph *g = handle;

    if (!g) {
        return 0;
    }

    return (which == 1 ? g->va_count + g->small_pages
                       : g->phys_count + g->small_pages) * MG_PAGE;
}

static void free_events(Event *event) {
    while (event) {
        Event *next = event->next;

        if (event->type == EV_CALL) {
            Event *scope = event->scope;
            free_events(scope->next);
            free(scope->snapshot);
            free(scope);
            free(event->name);
        }

        free(event);
        event = next;
    }
}

SHARED_EXPORT void malloc_graph_destroy(void *handle) {
    MallocGraph *g = handle;

    if (!g) {
        return;
    }

    if (active_graph == g) {
        active_graph = NULL;
    }

    while (g->state) {
        State *state = g->state;
        g->state = state->next;
        free(state);
    }

    for (size_t i = 0; i < g->va_count; i++) {
        if (g->va_phys[i] >= 0) {
            cuMemUnmap(g->base + i * MG_PAGE, MG_PAGE);
        }
    }

    for (size_t i = 0; i < g->phys_count; i++) {
        cuMemRelease(g->physical_handles[i]);
    }

    for (size_t i = 0; i < g->small_pages; i++) {
        cuMemUnmap(g->small_base + i * MG_PAGE, MG_PAGE);
        cuMemRelease(g->small_handles[i]);
    }

    cuMemAddressFree(g->base, MG_PAGES * MG_PAGE);
    cuMemAddressFree(g->small_base, MG_SMALL_PAGES * MG_PAGE);
    total_vram_usage -= (g->phys_count + g->small_pages) * MG_PAGE;

    free(g->root.snapshot);
    free(g->allocations);
    free_events(g->root.next);
    free(g);
}
