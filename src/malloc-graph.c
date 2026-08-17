#include "plat.h"

#define MG_PAGE (8ULL * M)
#define MG_PAGES 8192ULL
#define RETURN_G_FAILED(cond, retval) \
    if (cond) { \
        if (g) { \
            g->failed = true; \
        } \
        return retval; \
    }

typedef enum { EV_SENTINEL = 0, EV_ALLOC, EV_FREE, EV_CALL } EventType;

typedef struct Event Event;
typedef struct State State;

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
};

struct State {
    Event *cursor;
    size_t live;
    bool recording;

    State *next;
};

typedef struct {
    int phys;
    int va_span;
    State *owner;
} VirtualPage;

typedef struct {
    bool live;
    CUmemGenericAllocationHandle handle;
} PhysicalPage;

typedef struct {
    CUdeviceptr base;
    CUstream stream;
    int device;

    Event root;
    State *state;

    VirtualPage virtual_pages[MG_PAGES];
    PhysicalPage physical_pages[MG_PAGES];

    size_t va_count;
    size_t phys_count;

    bool failed;
    bool complete;
} MallocGraph;

static _Thread_local MallocGraph *active_graph;

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
    *state = (State){.cursor = scope, .recording = recording, .next = g->state};
    g->state = state;
    return true;
}

static int map_page(MallocGraph *g, size_t va, size_t phys) {
    CUmemAllocationProp prop = {.type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .location = {CU_MEM_LOCATION_TYPE_DEVICE, g->device}};
    CUmemAccessDesc access = {.location = {CU_MEM_LOCATION_TYPE_DEVICE, g->device},
                              .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    CUdeviceptr addr = g->base + va * MG_PAGE;
    CUresult r;

    if (phys == g->phys_count) {
        r = cuMemCreate(&g->physical_pages[phys].handle, MG_PAGE, &prop, 0);
        if (r) {
            return r;
        }
        g->phys_count++;
        total_vram_usage += MG_PAGE;
    }

    if ((r = cuMemMap(addr, MG_PAGE, 0, g->physical_pages[phys].handle, 0)) ||
        (r = cuMemSetAccess(addr, MG_PAGE, &access, 1))) {
        return r;
    }
    g->virtual_pages[va].phys = (int)phys;
    return 0;
}

bool malloc_graph_alloc(CUdeviceptr *ptr, size_t size, CUstream stream) {
    MallocGraph *g = active_graph;

    if (!g || stream != g->stream || size < MG_PAGE) {
        return false;
    }

    *ptr = 0;
    size_t va = 0;

    if (!g->state->recording) {
        Event *e = next_event(g, NULL);
        RETURN_G_FAILED(!e || e->type != EV_ALLOC || e->bytes != size, true);
        va = (uint32_t)e->value;
        g->state->cursor = e;
        *ptr = g->base + va * MG_PAGE;
        return true;
    }

    size_t pages = ALIGN_UP(size, MG_PAGE) / MG_PAGE;

    while (va + pages <= g->va_count) {
        size_t j;
        for (j = 0; j < pages; j++) {
            VirtualPage *vpage = &g->virtual_pages[va + j];
            if (g->physical_pages[vpage->phys].live) {
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
        VirtualPage *vpage = &g->virtual_pages[va + j];
        if (vpage->phys < 0) {
            size_t p = 0;
            while (p < g->phys_count && g->physical_pages[p].live) {
                p++;
            }
            RETURN_G_FAILED(map_page(g, va + j, p), true);
        }
        g->physical_pages[vpage->phys].live = true;
        vpage->owner = g->state;
    }

    event(g, EV_ALLOC, va, size);
    g->virtual_pages[va].va_span = pages;
    g->state->live += pages;
    *ptr = g->base + va * MG_PAGE;
    return true;
}

bool malloc_graph_free(CUdeviceptr ptr, size_t size, CUstream stream, int *result) {
    MallocGraph *g = active_graph;

    if (!g || stream != g->stream) {
        return false;
    }

    if (ptr < g->base || ptr >= g->base + MG_PAGES * MG_PAGE) {
        RETURN_G_FAILED(size >= MG_PAGE, false);
        return false;
    }

    *result = 0;
    size_t va = (ptr - g->base) / MG_PAGE;

    if (g->state->recording) {
        VirtualPage *first = &g->virtual_pages[va];
        RETURN_G_FAILED(first->owner != g->state || !first->va_span || !event(g, EV_FREE, va, 0), true);

        for (size_t j = 0; j < first->va_span; j++) {
            VirtualPage *vpage = &g->virtual_pages[va + j];
            g->physical_pages[vpage->phys].live = false;
            vpage->owner = NULL;
        }
        g->state->live -= first->va_span;
        first->va_span = 0;
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

    set_devctx(devctx);
    g->stream = stream;
    g->device = g_devctx->_device_id;

    if (cuMemAddressReserve(&g->base, MG_PAGES * MG_PAGE, MG_PAGE, 0, 0)) {
        goto fail;
    }

    for (size_t i = 0; i < MG_PAGES; i++) {
        g->virtual_pages[i].phys = -1;
    }

    if (!push_stack(g, &g->root, true)) {
        goto fail_address;
    }
    active_graph = g;
    return g;

fail_address:
    cuMemAddressFree(g->base, MG_PAGES * MG_PAGE);
fail:
    free(g);
    return NULL;
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

    return (which == 1 ? g->va_count : g->phys_count) * MG_PAGE;
}

static void free_events(Event *event) {
    while (event) {
        Event *next = event->next;

        if (event->type == EV_CALL) {
            Event *scope = event->scope;
            free_events(scope->next);
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
        if (g->virtual_pages[i].phys >= 0) {
            cuMemUnmap(g->base + i * MG_PAGE, MG_PAGE);
        }
    }

    for (size_t i = 0; i < g->phys_count; i++) {
        cuMemRelease(g->physical_pages[i].handle);
    }

    cuMemAddressFree(g->base, MG_PAGES * MG_PAGE);
    total_vram_usage -= g->phys_count * MG_PAGE;

    free_events(g->root.next);
    free(g);
}
