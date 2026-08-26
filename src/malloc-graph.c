#include "plat.h"

#define MG_PAGE (8ULL * M)
#define MG_PAGES 8192ULL
#define MG_SMALL_PAGES (MG_PAGES / 8)
#define MG_SMALL_LIMIT (4ULL * M)
#define MG_SMALL_ALIGN 4096ULL
#define RETURN_G_FAILED(cond, retval) \
    do { \
        if (cond) { \
            if (g) { \
                g->failed = true; \
            } \
            return retval; \
        } \
    } while (0)

typedef enum {
    EV_SENTINEL = 0,
    EV_ALLOC,
    EV_FREE,
    EV_ALLOC_SMALL,
    EV_FREE_SMALL,
    EV_CALL,
    EV_END,
} EventType;

typedef struct Event Event;
typedef struct State State;
typedef struct AllocationState AllocationState;
typedef struct SmallRange SmallRange;

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

    Event **next;
    size_t next_count;
    Event *previous;
    Event *allocation_previous;
    AllocationState *snapshot;
    SmallRange *small_snapshot;
};

struct State {
    Event *cursor;
    Event *allocations;
    uint32_t depth;
    bool recording;
    bool broken;

    State *next;
};

typedef struct {
    int va_span;
    uint32_t owner_depth;
    Event *allocation;
} VirtualPage;

struct SmallRange {
    size_t offset;
    size_t bytes;
    uint32_t owner_depth;
    Event *allocation;

    SmallRange *next;
};

struct AllocationState {
    VirtualPage virtual_pages[MG_PAGES];
    bool physical_live[MG_PAGES];
};

typedef struct {
    CUdeviceptr base;
    CUdeviceptr small_base;
    CUstream stream;
    int device;
    void *owner_thread;

    Event root;
    State *state;

    AllocationState allocations;
    SmallRange *small_ranges;
    int va_phys[MG_PAGES];
    CUmemGenericAllocationHandle physical_handles[MG_PAGES];
    CUmemGenericAllocationHandle small_handles[MG_SMALL_PAGES];

    size_t va_count;
    size_t phys_count;
    size_t small_size;
    size_t small_pages;
    size_t used;
    size_t peak_used;

    bool failed;
    bool complete;
    bool paused;
    bool sync_paused;
    bool assert_breaks;
} MallocGraph;

static _Thread_local MallocGraph *active_graph;

bool malloc_graph_sync_paused(void) {
    return active_graph && active_graph->sync_paused;
}

static void free_small_ranges(SmallRange *range) {
    while (range) {
        SmallRange *next = range->next;
        free(range);
        range = next;
    }
}

static bool copy_small_ranges(SmallRange **copy, SmallRange *range) {
    for (; range; range = range->next) {
        *copy = malloc(sizeof(**copy));
        if (!*copy) {
            return false;
        }
        **copy = *range;
        copy = &(*copy)->next;
    }
    return true;
}

static void add_used(MallocGraph *g, size_t bytes) {
    g->used += bytes;
    g->peak_used = MAX(g->peak_used, g->used);
}

static size_t allocation_used(MallocGraph *g) {
    size_t used = 0;

    for (size_t i = 0; i < g->phys_count; i++) {
        if (g->allocations.physical_live[i]) {
            used += MG_PAGE;
        }
    }
    for (SmallRange *range = g->small_ranges; range; range = range->next) {
        used += range->bytes;
    }
    return used;
}

static bool append_event(MallocGraph *g, Event *parent, Event *event) {
    Event **next = realloc(parent->next, (parent->next_count + 1) * sizeof(*next));
    RETURN_G_FAILED(!next, false);
    parent->next = next;
    parent->next[parent->next_count++] = event;
    event->previous = parent;
    return true;
}

static Event *find_event(MallocGraph *g, EventType type, size_t value, size_t bytes, const char *name) {
    Event *cursor = g->state->cursor;

    for (size_t i = 0; i < cursor->next_count; i++) {
        Event *event = cursor->next[i];
        if (event->type != type ||
            (type == EV_CALL && strcmp(event->name, name)) ||
            (type != EV_CALL && event->bytes != bytes) ||
            ((type == EV_FREE || type == EV_FREE_SMALL) && event->value != value)) {
            continue;
        }
        return event;
    }
    return NULL;
}

static Event *event(MallocGraph *g, EventType type, size_t value, size_t bytes) {
    Event *event = calloc(1, sizeof(*event));
    RETURN_G_FAILED(!event, NULL);
    event->type = type;
    event->value = value;
    event->bytes = bytes;
    if (!append_event(g, g->state->cursor, event)) {
        free(event);
        return NULL;
    }
    g->state->cursor = event;
    return event;
}

static void track_allocation(MallocGraph *g, Event *event) {
    event->allocation_previous = g->state->allocations;
    g->state->allocations = event;
}

static bool untrack_allocation(MallocGraph *g, Event *event) {
    Event **entry = &g->state->allocations;
    while (*entry && *entry != event) {
        entry = &(*entry)->allocation_previous;
    }
    if (!*entry) {
        return false;
    }
    *entry = event->allocation_previous;
    event->allocation_previous = NULL;
    return true;
}

static bool push_stack(MallocGraph *g, Event *scope, bool recording) {
    if (recording && !scope->snapshot) {
        RETURN_G_FAILED(!copy_small_ranges(&scope->small_snapshot, g->small_ranges) ||
                        !(scope->snapshot = malloc(sizeof(*scope->snapshot))), false);
        *scope->snapshot = g->allocations;
    }

    State *state = malloc(sizeof(*state));
    RETURN_G_FAILED(!state, false);
    *state = (State){.cursor = scope, .depth = g->state ? g->state->depth + 1 : 1,
                     .recording = recording, .next = g->state};
    g->state = state;
    return true;
}

static CUresult create_page(MallocGraph *g, CUmemGenericAllocationHandle *handle) {
    CUmemAllocationProp prop = {.type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .location = {CU_MEM_LOCATION_TYPE_DEVICE, g->device}};
    CUresult r;

    vbars_free(budget_deficit(MG_PAGE));
    r = cuMemCreate(handle, MG_PAGE, &prop, 0);
    if (r == CUDA_ERROR_OUT_OF_MEMORY) {
        vbars_free(MG_PAGE);
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
    AllocationState *allocations = &g->allocations;
    size_t value = event->value;

    if (event->type == EV_ALLOC || event->type == EV_FREE) {
        VirtualPage *first = &allocations->virtual_pages[value];

        if (event->type == EV_ALLOC) {
            *first = (VirtualPage){
                .va_span = ALIGN_UP(event->bytes, MG_PAGE) / MG_PAGE,
                .owner_depth = g->state->depth,
                .allocation = event};
            track_allocation(g, event);
            add_used(g, first->va_span * MG_PAGE);
        }

        for (size_t i = 0; i < first->va_span; i++) {
            allocations->physical_live[g->va_phys[value + i]] = event->type == EV_ALLOC;
        }

        if (event->type == EV_FREE) {
            RETURN_G_FAILED(!untrack_allocation(g, first->allocation), false);
            g->used -= first->va_span * MG_PAGE;
            *first = (VirtualPage){};
        }
    } else if (event->type == EV_ALLOC_SMALL || event->type == EV_FREE_SMALL) {
        SmallRange **entry = &g->small_ranges;

        while (*entry && (*entry)->offset < value) {
            entry = &(*entry)->next;
        }

        if (event->type == EV_ALLOC_SMALL) {
            SmallRange *range = malloc(sizeof(*range));
            RETURN_G_FAILED(!range, false);
            *range = (SmallRange){
                .offset = value, .bytes = ALIGN_UP(event->bytes, MG_SMALL_ALIGN),
                .owner_depth = g->state->depth, .allocation = event, .next = *entry};
            *entry = range;
            track_allocation(g, event);
            add_used(g, range->bytes);
        } else {
            SmallRange *range = *entry;
            RETURN_G_FAILED(!untrack_allocation(g, range->allocation), false);
            *entry = range->next;
            g->used -= range->bytes;
            free(range);
        }
    }
    return true;
}

static bool materialize(MallocGraph *g, Event *event) {
    if (event->snapshot) {
        SmallRange *small_ranges = NULL;
        if (!copy_small_ranges(&small_ranges, event->small_snapshot)) {
            g->failed = true;
            free_small_ranges(small_ranges);
            return false;
        }
        free_small_ranges(g->small_ranges);
        g->allocations = *event->snapshot;
        g->small_ranges = small_ranges;
        g->state->allocations = NULL;
        g->used = allocation_used(g);
        g->peak_used = MAX(g->peak_used, g->used);
        return true;
    }
    return materialize(g, event->previous) && dry_apply(g, event);
}

static bool start_recording(MallocGraph *g) {
    RETURN_G_FAILED(g->assert_breaks, false);
    if (!materialize(g, g->state->cursor)) {
        return false;
    }
    g->state->recording = true;
    g->state->broken = true;
    return true;
}

bool malloc_graph_alloc(CUdeviceptr *ptr, size_t size, CUstream stream) {
    MallocGraph *g = active_graph;

    if (!g || g->failed || g->paused || stream != g->stream) {
        return false;
    }

    *ptr = 0;
    size_t va = 0;

    if (!g->state->recording) {
        EventType type = size < MG_SMALL_LIMIT ? EV_ALLOC_SMALL : EV_ALLOC;
        Event *match = find_event(g, type, 0, size, NULL);
        if (match) {
            g->state->cursor = match;
            *ptr = type == EV_ALLOC_SMALL
                ? g->small_base + match->value
                : g->base + match->value * MG_PAGE;
            return true;
        }
        RETURN_G_FAILED(!start_recording(g), true);
    }

    if (size < MG_SMALL_LIMIT) {
        size_t bytes = ALIGN_UP(size, MG_SMALL_ALIGN);
        size_t offset = g->small_size;
        size_t smallest_hole = SIZE_MAX;
        size_t previous_end = 0;
        SmallRange **insert_at = NULL;
        SmallRange **previous_next = &g->small_ranges;

        for (SmallRange *range = g->small_ranges; range; range = range->next) {
            size_t hole = range->offset - previous_end;
            if (hole >= bytes && hole < smallest_hole) {
                offset = previous_end;
                smallest_hole = hole;
                insert_at = previous_next;
            }
            previous_end = range->offset + range->bytes;
            previous_next = &range->next;
        }
        size_t hole = g->small_size - previous_end;
        if (hole >= bytes && hole < smallest_hole) {
            offset = previous_end;
            insert_at = previous_next;
        }
        if (!insert_at) {
            insert_at = previous_next;
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

        SmallRange *range = malloc(sizeof(*range));
        RETURN_G_FAILED(!range, true);
        *range = (SmallRange){.offset = offset, .bytes = bytes,
                              .owner_depth = g->state->depth, .next = *insert_at};
        *insert_at = range;

        Event *allocation = event(g, EV_ALLOC_SMALL, offset, size);
        RETURN_G_FAILED(!allocation, true);
        range->allocation = allocation;
        track_allocation(g, allocation);
        add_used(g, bytes);
        *ptr = g->small_base + offset;
        return true;
    }

    size_t pages = ALIGN_UP(size, MG_PAGE) / MG_PAGE;

    while (va + pages <= g->va_count) {
        size_t j;
        for (j = 0; j < pages; j++) {
            int phys = g->va_phys[va + j];
            if (g->allocations.physical_live[phys]) {
                break;
            }
            g->allocations.physical_live[phys] = true;
        }
        if (j == pages) {
            break;
        }
        for (size_t k = 0; k < j; k++) {
            g->allocations.physical_live[g->va_phys[va + k]] = false;
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
            while (p < g->phys_count && g->allocations.physical_live[p]) {
                p++;
            }
            RETURN_G_FAILED(map_page(g, va + j, p), true);
        }
        g->allocations.physical_live[g->va_phys[va + j]] = true;
    }

    Event *allocation = event(g, EV_ALLOC, va, size);
    RETURN_G_FAILED(!allocation, true);
    g->allocations.virtual_pages[va] = (VirtualPage){
        .va_span = pages, .owner_depth = g->state->depth, .allocation = allocation};
    track_allocation(g, allocation);
    add_used(g, pages * MG_PAGE);
    *ptr = g->base + va * MG_PAGE;
    return true;
}

bool malloc_graph_free(CUdeviceptr ptr, size_t size, CUstream stream, int *result) {
    MallocGraph *g = active_graph;

    if (!g || g->failed || g->paused || stream != g->stream) {
        return false;
    }

    bool small = ptr >= g->small_base && ptr < g->small_base + MG_SMALL_PAGES * MG_PAGE;
    if (!small && (ptr < g->base || ptr >= g->base + MG_PAGES * MG_PAGE)) {
        RETURN_G_FAILED(size, false);
        return false;
    }

    size_t value = small ? ptr - g->small_base : (ptr - g->base) / MG_PAGE;

    *result = 0;
    if (!g->state->recording) {
        EventType type = small ? EV_FREE_SMALL : EV_FREE;
        Event *match = find_event(g, type, value, 0, NULL);
        if (match) {
            g->state->cursor = match;
            return true;
        }
        RETURN_G_FAILED(!start_recording(g), true);
    }

    if (small) {
        SmallRange **entry = &g->small_ranges;
        while (*entry && (*entry)->offset != value) {
            entry = &(*entry)->next;
        }
        RETURN_G_FAILED(!*entry || (*entry)->owner_depth != g->state->depth ||
                        !event(g, EV_FREE_SMALL, value, 0) ||
                        !untrack_allocation(g, (*entry)->allocation), true);

        SmallRange *range = *entry;
        *entry = range->next;
        g->used -= range->bytes;
        free(range);
        return true;
    }

    VirtualPage *first = &g->allocations.virtual_pages[value];
    RETURN_G_FAILED(first->owner_depth != g->state->depth || !first->va_span ||
                    !event(g, EV_FREE, value, 0) ||
                    !untrack_allocation(g, first->allocation), true);

    for (size_t j = 0; j < first->va_span; j++) {
        g->allocations.physical_live[g->va_phys[value + j]] = false;
    }
    g->used -= first->va_span * MG_PAGE;
    *first = (VirtualPage){};
    return true;
}

SHARED_EXPORT void *malloc_graph_create(void *devctx, CUstream stream, bool assert_breaks) {
    MallocGraph *g = calloc(1, sizeof(*g));

    if (!g || active_graph) {
        free(g);
        return NULL;
    }

    set_devctx(devctx);
    g->stream = stream;
    g->device = g_devctx->_device_id;
    g->owner_thread = &active_graph;
    g->assert_breaks = assert_breaks;

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
    free_small_ranges(g->root.small_snapshot);
    free(g->root.snapshot);
    free(g);
    return NULL;
}

SHARED_EXPORT bool malloc_graph_pause(void *handle, bool paused, bool sync) {
    MallocGraph *g = handle;

    if (!g || g != active_graph || g->failed) {
        return false;
    }
    g->paused = paused;
    if (sync) {
        g->sync_paused = paused;
    }
    return true;
}

SHARED_EXPORT bool malloc_graph_set_stream(void *handle, CUstream stream) {
    MallocGraph *g = handle;

    if (!g || g != active_graph || g->failed) {
        return false;
    }
    g->stream = stream;
    return true;
}

SHARED_EXPORT bool malloc_graph_push(void *handle, const char *name) {
    MallocGraph *g = handle;

    if (!g || g->failed) {
        return false;
    }
    if (!name) {
        if (!g->complete || active_graph || !push_stack(g, &g->root, false)) {
            return false;
        }
        g->complete = false;
        active_graph = g;
        return true;
    }
    if (g != active_graph) {
        return false;
    }

    Event *call = find_event(g, EV_CALL, 0, 0, name);

    bool recording = !call;
    Event *scope;
    if (recording) {
        if (!g->state->recording && !start_recording(g)) {
            return false;
        }

        char *call_name = NULL;
        scope = NULL;
        if (!(call = calloc(1, sizeof(*call))) ||
            !(scope = calloc(1, sizeof(*scope))) ||
            !(call_name = malloc(strlen(name) + 1)) ||
            !append_event(g, g->state->cursor, call)) {
            free(call_name);
            free(call);
            free(scope);
            g->failed = true;
            return false;
        }

        strcpy(call_name, name);
        call->type = EV_CALL;
        call->scope = scope;
        call->name = call_name;
    } else {
        scope = call->scope;
    }

    return push_stack(g, scope, recording);
}

SHARED_EXPORT int malloc_graph_pop(void *handle) {
    MallocGraph *g = handle;

    if (!g || g != active_graph || g->failed) {
        return false;
    }

    if (!g->state->recording) {
        Event *end = find_event(g, EV_END, 0, 0, NULL);
        if (end) {
            g->state->cursor = end;
        } else if (!start_recording(g)) {
            return false;
        }
    }
    RETURN_G_FAILED(g->state->allocations, false);
    RETURN_G_FAILED(g->state->recording && !event(g, EV_END, 0, 0), false);

    State *state = g->state;
    g->state = state->next;
    int result = state->broken ? 2 : 1;
    free(state);

    if (!g->state) {
        g->complete = true;
        active_graph = NULL;
    }
    return result;
}

SHARED_EXPORT uint64_t malloc_graph_stat(void *handle, int which) {
    MallocGraph *g = handle;

    if (!g) {
        return 0;
    }

    switch (which) {
    case 0:
        return g->peak_used;
    case 1:
        return (g->va_count + g->small_pages) * MG_PAGE;
    case 2:
        return (g->phys_count + g->small_pages) * MG_PAGE;
    default:
        return 0;
    }
}

static void free_events(Event **events, size_t count) {
    for (size_t i = 0; i < count; i++) {
        Event *event = events[i];
        free_events(event->next, event->next_count);
        if (event->type == EV_CALL) {
            Event *scope = event->scope;
            free_events(scope->next, scope->next_count);
            free_small_ranges(scope->small_snapshot);
            free(scope->snapshot);
            free(scope);
            free(event->name);
        }

        free(event);
    }
    free(events);
}

SHARED_EXPORT void malloc_graph_destroy(void *handle) {
    MallocGraph *g = handle;

    if (!g || g->owner_thread != &active_graph) {
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

    free_small_ranges(g->root.small_snapshot);
    free_small_ranges(g->small_ranges);
    free(g->root.snapshot);
    free_events(g->root.next, g->root.next_count);
    free(g);
}
