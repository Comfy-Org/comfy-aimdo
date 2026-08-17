#include "plat.h"

#define MG_PAGE (8ULL * M)
#define MG_PAGES 8192ULL

typedef enum { EV_ALLOC, EV_FREE, EV_CALL } EventType;
typedef struct Event Event;
struct Event {
    EventType type;
    Event *next;
    union {
        struct {
            size_t value, bytes;
        };
        Event *scope;
        struct {
            char *name;
            Event *next_scope;
        };
    };
};

typedef struct State {
    Event *scope, *cursor;
    size_t live;
    bool recording;
    struct State *next;
} State;

typedef struct {
    CUdeviceptr base;
    CUstream stream;
    int device;
    Event root, *scope, *cursor, *scopes;
    State *stack, *free_states;
    size_t live;
    int *va_phys;
    uint32_t *va_span;
    Event **owners;
    uint8_t *va_live, *phys_live;
    CUmemGenericAllocationHandle *handles;
    size_t va_high, phys_count, used, peak;
    bool recording, failed, complete;
} MallocGraph;

static _Thread_local MallocGraph *active_graph;

static bool event(MallocGraph *g, EventType type, size_t value, size_t bytes, Event *scope) {
    Event *e;
    if (g->recording) {
        e = calloc(1, sizeof(*e));
        if (!e) {
            g->failed = true;
            return false;
        }
        e->type = type;
        if (type == EV_CALL) e->scope = scope;
        else { e->value = value; e->bytes = bytes; }
        g->cursor->next = e;
    } else {
        e = g->cursor->next;
        if (!e || e->type != type ||
            (type == EV_CALL ? e->scope != scope : e->value != value || e->bytes != bytes)) {
            g->failed = true;
            return false;
        }
    }
    g->cursor = e;
    return true;
}

static Event *find_scope(MallocGraph *g, const char *name) {
    for (Event *scope = g->scopes; scope; scope = scope->next_scope)
        if (!strcmp(scope->name, name)) return scope;
    return NULL;
}

static bool push_stack(MallocGraph *g) {
    State *state = g->free_states;
    if (state) g->free_states = state->next;
    else if (!(state = malloc(sizeof(*state)))) return false;
    *state = (State){g->scope, g->cursor, g->live, g->recording, g->stack};
    g->stack = state;
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
        r = cuMemCreate(&g->handles[phys], MG_PAGE, &prop, 0);
        if (r) return r;
        g->phys_count++;
        total_vram_usage += MG_PAGE;
    }
    if ((r = cuMemMap(addr, MG_PAGE, 0, g->handles[phys], 0)) ||
        (r = cuMemSetAccess(addr, MG_PAGE, &access, 1))) return r;
    g->va_phys[va] = (int)phys;
    return 0;
}

static int graph_alloc(MallocGraph *g, CUdeviceptr *ptr, size_t size) {
    size_t pages = ALIGN_UP(size, MG_PAGE) / MG_PAGE, va = 0;
    if (!g->recording) {
        Event *e = g->cursor->next;
        if (!e || e->type != EV_ALLOC || e->bytes != size) {
            g->failed = true;
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        va = (uint32_t)e->value;
        g->cursor = e;
    } else {
        bool found = false;
        for (; va + pages <= g->va_high; va++) {
            size_t j;
            for (j = 0; j < pages; j++) {
                int p = g->va_phys[va + j];
                if (g->va_live[va + j] || (p >= 0 && g->phys_live[p])) break;
            }
            if (j == pages) { found = true; break; }
        }
        if (!found) va = g->va_high;
        if (va + pages > MG_PAGES) {
            g->failed = true;
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        if (va + pages > g->va_high) g->va_high = va + pages;
        for (size_t j = 0; j < pages; j++) {
            if (g->va_phys[va + j] < 0) {
                size_t p = 0;
                while (p < g->phys_count && g->phys_live[p]) p++;
                if (map_page(g, va + j, p)) {
                    g->failed = true;
                    return CUDA_ERROR_OUT_OF_MEMORY;
                }
            }
            g->phys_live[g->va_phys[va + j]] = 1;
        }
        event(g, EV_ALLOC, va, size, NULL);
    }
    for (size_t j = 0; j < pages; j++) {
        int p = g->va_phys[va + j];
        g->va_live[va + j] = g->phys_live[p] = 1;
        g->owners[va + j] = g->scope;
    }
    g->va_span[va] = pages;
    g->live += pages;
    g->used += pages;
    if (g->used > g->peak) g->peak = g->used;
    *ptr = g->base + va * MG_PAGE;
    return 0;
}

static int graph_free(MallocGraph *g, CUdeviceptr ptr) {
    if (ptr < g->base || ptr >= g->base + MG_PAGES * MG_PAGE) {
        g->failed = true;
        return 0;
    }
    size_t va = (ptr - g->base) / MG_PAGE;
    if (!g->va_live[va] || g->owners[va] != g->scope) {
        g->failed = true;
        return 0;
    }
    size_t pages = g->va_span[va];
    if (!pages) { g->failed = true; return 0; }
    if (!event(g, EV_FREE, va, 0, NULL)) return 0;
    for (size_t j = 0; j < pages; j++) {
        int p = g->va_phys[va + j];
        g->va_live[va + j] = 0;
        g->phys_live[p] = 0;
        g->owners[va + j] = NULL;
    }
    g->va_span[va] = 0;
    g->live -= pages;
    g->used -= pages;
    return 0;
}

bool malloc_graph_alloc(CUdeviceptr *ptr, size_t size, CUstream stream) {
    MallocGraph *g = active_graph;
    if (!g || stream != g->stream || size < MG_PAGE) return false;
    *ptr = 0;
    graph_alloc(g, ptr, size);
    return true;
}

bool malloc_graph_free(CUdeviceptr ptr, size_t size, CUstream stream, int *result) {
    MallocGraph *g = active_graph;
    if (!g || stream != g->stream) return false;
    if (ptr < g->base || ptr >= g->base + MG_PAGES * MG_PAGE) {
        if (size >= MG_PAGE) g->failed = true;
        return false;
    }
    *result = graph_free(g, ptr);
    return true;
}

SHARED_EXPORT void *malloc_graph_create(void *devctx, CUstream stream) {
    MallocGraph *g = calloc(1, sizeof(*g));
    if (!g || active_graph) { free(g); return NULL; }
    set_devctx(devctx);
    g->stream = stream;
    g->device = g_devctx->_device_id;
    g->va_phys = malloc(MG_PAGES * sizeof(int));
    g->va_span = calloc(MG_PAGES, sizeof(uint32_t));
    g->owners = calloc(MG_PAGES, sizeof(Event *));
    g->va_live = calloc(MG_PAGES, 1);
    g->phys_live = calloc(MG_PAGES, 1);
    g->handles = calloc(MG_PAGES, sizeof(*g->handles));
    if (!g->va_phys || !g->va_span || !g->owners || !g->va_live || !g->phys_live || !g->handles ||
        cuMemAddressReserve(&g->base, MG_PAGES * MG_PAGE, MG_PAGE, 0, 0)) {
        free(g->va_phys); free(g->va_span); free(g->owners); free(g->va_live); free(g->phys_live); free(g->handles); free(g);
        return NULL;
    }
    for (size_t i = 0; i < MG_PAGES; i++) g->va_phys[i] = -1;
    g->scope = g->cursor = &g->root;
    g->recording = true;
    active_graph = g;
    return g;
}

SHARED_EXPORT bool malloc_graph_push(void *handle, const char *name) {
    MallocGraph *g = handle;
    if (!g || g != active_graph || g->failed) return false;
    Event *scope = find_scope(g, name);
    if (scope && g->recording &&
        (g->cursor->type != EV_CALL || g->cursor->scope != scope)) {
        g->failed = true;
        return false;
    }
    bool recording = !scope;
    if (recording) {
        scope = calloc(1, sizeof(*scope));
        if (!scope) { g->failed = true; return false; }
        scope->name = strdup(name);
        scope->next_scope = g->scopes;
        g->scopes = scope;
    }
    if (!event(g, EV_CALL, 0, 0, scope) || !push_stack(g)) {
        g->failed = true; return false;
    }
    g->scope = g->cursor = scope;
    g->live = 0;
    g->recording = recording;
    return true;
}

SHARED_EXPORT bool malloc_graph_pop(void *handle) {
    MallocGraph *g = handle;
    if (!g || g != active_graph || g->failed) return false;
    if (g->live) { g->failed = true; return false; }
    if (!g->recording && g->cursor->next) { g->failed = true; return false; }
    if (g->stack) {
        State *state = g->stack;
        g->stack = state->next;
        g->scope = state->scope;
        g->cursor = state->cursor;
        g->live = state->live;
        g->recording = state->recording;
        state->next = g->free_states;
        g->free_states = state;
    } else {
        g->complete = true;
        active_graph = NULL;
    }
    return true;
}

SHARED_EXPORT bool malloc_graph_replay(void *handle) {
    MallocGraph *g = handle;
    if (!g || g->failed || !g->complete || active_graph) return false;
    g->scope = g->cursor = &g->root;
    g->live = 0; g->recording = false;
    g->complete = false; active_graph = g;
    return true;
}

SHARED_EXPORT uint64_t malloc_graph_stat(void *handle, int which) {
    MallocGraph *g = handle;
    if (!g) return 0;
    return (which == 0 ? g->peak : which == 1 ? g->va_high : g->phys_count) * MG_PAGE;
}

static void free_events(Event *scope) {
    Event *event = scope->next;
    while (event) {
        Event *next = event->next;
        free(event);
        event = next;
    }
}

static void free_states(State *state) {
    while (state) {
        State *next = state->next;
        free(state);
        state = next;
    }
}

SHARED_EXPORT void malloc_graph_destroy(void *handle) {
    MallocGraph *g = handle;
    if (!g) return;
    if (active_graph == g) active_graph = NULL;
    for (size_t i = 0; i < g->va_high; i++) if (g->va_phys[i] >= 0) cuMemUnmap(g->base + i * MG_PAGE, MG_PAGE);
    for (size_t i = 0; i < g->phys_count; i++) cuMemRelease(g->handles[i]);
    cuMemAddressFree(g->base, MG_PAGES * MG_PAGE);
    total_vram_usage -= g->phys_count * MG_PAGE;
    free_events(&g->root);
    Event *scope = g->scopes;
    while (scope) {
        Event *next = scope->next_scope;
        free_events(scope); free(scope->name); free(scope);
        scope = next;
    }
    free_states(g->stack); free_states(g->free_states);
    free(g->va_phys); free(g->va_span); free(g->owners); free(g->va_live);
    free(g->phys_live); free(g->handles); free(g);
}
