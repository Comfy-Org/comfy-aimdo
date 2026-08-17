#include "plat.h"

#define MG_PAGE (8ULL * M)
#define MG_PAGES 8192ULL

typedef enum { EV_ALLOC, EV_FREE, EV_CALL } EventType;
typedef struct Scope Scope;
typedef struct {
    EventType type;
    size_t value, bytes;
    Scope *scope;
} Event;

struct Scope {
    char *name;
    Event *events;
    size_t count, capacity, cursor, live;
    bool recording, active;
};

typedef struct {
    CUdeviceptr base;
    CUstream stream;
    int device;
    Scope root, *current;
    Scope **stack, **scopes;
    size_t depth, stack_capacity, scope_count, scope_capacity;
    int *va_phys;
    uint32_t *va_span;
    Scope **owners;
    uint8_t *va_live, *phys_live;
    CUmemGenericAllocationHandle *handles;
    size_t va_high, phys_count, used, peak;
    bool failed, complete;
} MallocGraph;

static _Thread_local MallocGraph *active_graph;

static void fail(MallocGraph *g) { g->failed = true; }

static bool grow(void **p, size_t *capacity, size_t count, size_t size) {
    if (count < *capacity) return true;
    size_t n = *capacity ? *capacity * 2 : 8;
    void *q = realloc(*p, n * size);
    if (!q) return false;
    *p = q;
    *capacity = n;
    return true;
}

static bool event(MallocGraph *g, Scope *s, EventType type, size_t value, size_t bytes, Scope *scope) {
    if (s->recording) {
        if (!grow((void **)&s->events, &s->capacity, s->count, sizeof(Event))) {
            fail(g);
            return false;
        }
        s->events[s->count++] = (Event){type, value, bytes, scope};
        return true;
    }
    if (s->cursor == s->count || s->events[s->cursor].type != type ||
        s->events[s->cursor].value != value || s->events[s->cursor].bytes != bytes ||
        s->events[s->cursor].scope != scope) {
        fail(g);
        return false;
    }
    s->cursor++;
    return true;
}

static Scope *find_scope(MallocGraph *g, const char *name) {
    for (size_t i = 0; i < g->scope_count; i++)
        if (!strcmp(g->scopes[i]->name, name)) return g->scopes[i];
    return NULL;
}

static bool push_stack(MallocGraph *g, Scope *scope) {
    if (!grow((void **)&g->stack, &g->stack_capacity, g->depth, sizeof(Scope *))) return false;
    g->stack[g->depth++] = scope;
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
    Scope *s = g->current;
    size_t pages = ALIGN_UP(size, MG_PAGE) / MG_PAGE, va = 0;
    if (!s->recording) {
        if (s->cursor == s->count || s->events[s->cursor].type != EV_ALLOC ||
            s->events[s->cursor].bytes != size) {
            fail(g);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        va = (uint32_t)s->events[s->cursor].value;
        s->cursor++;
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
            fail(g);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        if (va + pages > g->va_high) g->va_high = va + pages;
        for (size_t j = 0; j < pages; j++) {
            if (g->va_phys[va + j] < 0) {
                size_t p = 0;
                while (p < g->phys_count && g->phys_live[p]) p++;
                if (map_page(g, va + j, p)) {
                    fail(g);
                    return CUDA_ERROR_OUT_OF_MEMORY;
                }
            }
            g->phys_live[g->va_phys[va + j]] = 1;
        }
        event(g, s, EV_ALLOC, va, size, NULL);
    }
    for (size_t j = 0; j < pages; j++) {
        int p = g->va_phys[va + j];
        g->va_live[va + j] = g->phys_live[p] = 1;
        g->owners[va + j] = s;
    }
    g->va_span[va] = pages;
    s->live += pages;
    g->used += pages;
    if (g->used > g->peak) g->peak = g->used;
    *ptr = g->base + va * MG_PAGE;
    return 0;
}

static int graph_free(MallocGraph *g, CUdeviceptr ptr) {
    if (ptr < g->base || ptr >= g->base + MG_PAGES * MG_PAGE) {
        fail(g);
        return 0;
    }
    size_t va = (ptr - g->base) / MG_PAGE;
    if (!g->va_live[va] || g->owners[va] != g->current) {
        fail(g);
        return 0;
    }
    size_t pages = g->va_span[va];
    if (!pages) { fail(g); return 0; }
    if (!event(g, g->current, EV_FREE, va, 0, NULL)) return 0;
    for (size_t j = 0; j < pages; j++) {
        int p = g->va_phys[va + j];
        g->va_live[va + j] = 0;
        g->phys_live[p] = 0;
        g->owners[va + j] = NULL;
    }
    g->va_span[va] = 0;
    g->current->live -= pages;
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

bool malloc_graph_free(CUdeviceptr ptr, CUstream stream, int *result) {
    MallocGraph *g = active_graph;
    if (!g || stream != g->stream) return false;
    if (ptr < g->base || ptr >= g->base + MG_PAGES * MG_PAGE) return false;
    *result = graph_free(g, ptr);
    return true;
}

bool malloc_graph_reject_external(CUstream stream) {
    if (!active_graph || active_graph->stream != stream) return false;
    fail(active_graph);
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
    g->owners = calloc(MG_PAGES, sizeof(Scope *));
    g->va_live = calloc(MG_PAGES, 1);
    g->phys_live = calloc(MG_PAGES, 1);
    g->handles = calloc(MG_PAGES, sizeof(*g->handles));
    if (!g->va_phys || !g->va_span || !g->owners || !g->va_live || !g->phys_live || !g->handles ||
        cuMemAddressReserve(&g->base, MG_PAGES * MG_PAGE, MG_PAGE, 0, 0)) {
        free(g->va_phys); free(g->va_span); free(g->owners); free(g->va_live); free(g->phys_live); free(g->handles); free(g);
        return NULL;
    }
    for (size_t i = 0; i < MG_PAGES; i++) g->va_phys[i] = -1;
    g->root.recording = true;
    g->current = &g->root;
    active_graph = g;
    return g;
}

SHARED_EXPORT bool malloc_graph_push(void *handle, const char *name) {
    MallocGraph *g = handle;
    if (!g || g != active_graph || g->failed) return false;
    Scope *s = find_scope(g, name);
    if (s && s->active) { fail(g); return false; }
    if (s && g->current->recording &&
        (!g->current->count || g->current->events[g->current->count - 1].type != EV_CALL ||
         g->current->events[g->current->count - 1].scope != s)) {
        fail(g);
        return false;
    }
    if (!s) {
        s = calloc(1, sizeof(*s));
        if (!s || !grow((void **)&g->scopes, &g->scope_capacity, g->scope_count, sizeof(Scope *))) {
            free(s); fail(g); return false;
        }
        s->name = strdup(name);
        s->recording = true;
        g->scopes[g->scope_count++] = s;
    }
    if (!event(g, g->current, EV_CALL, 0, 0, s) || !push_stack(g, g->current)) {
        fail(g); return false;
    }
    s->cursor = 0; s->active = true;
    g->current = s;
    return true;
}

SHARED_EXPORT bool malloc_graph_pop(void *handle) {
    MallocGraph *g = handle;
    if (!g || g != active_graph || g->failed) return false;
    Scope *s = g->current;
    if (s->live) { fail(g); return false; }
    if (!s->recording && s->cursor != s->count) { fail(g); return false; }
    s->recording = false;
    if (g->depth) {
        s->active = false;
        g->current = g->stack[--g->depth];
    } else {
        g->complete = true;
        active_graph = NULL;
    }
    return true;
}

SHARED_EXPORT bool malloc_graph_replay(void *handle) {
    MallocGraph *g = handle;
    if (!g || g->failed || !g->complete || active_graph) return false;
    g->root.cursor = 0; g->root.recording = false; g->root.active = true;
    g->current = &g->root; g->complete = false; active_graph = g;
    return true;
}

SHARED_EXPORT uint64_t malloc_graph_stat(void *handle, int which) {
    MallocGraph *g = handle;
    if (!g) return 0;
    return (which == 0 ? g->peak : which == 1 ? g->va_high : g->phys_count) * MG_PAGE;
}

SHARED_EXPORT void malloc_graph_destroy(void *handle) {
    MallocGraph *g = handle;
    if (!g) return;
    if (active_graph == g) active_graph = NULL;
    for (size_t i = 0; i < g->va_high; i++) if (g->va_phys[i] >= 0) cuMemUnmap(g->base + i * MG_PAGE, MG_PAGE);
    for (size_t i = 0; i < g->phys_count; i++) cuMemRelease(g->handles[i]);
    cuMemAddressFree(g->base, MG_PAGES * MG_PAGE);
    total_vram_usage -= g->phys_count * MG_PAGE;
    free(g->root.events);
    for (size_t i = 0; i < g->scope_count; i++) { free(g->scopes[i]->name); free(g->scopes[i]->events); free(g->scopes[i]); }
    free(g->scopes); free(g->stack); free(g->va_phys); free(g->va_span); free(g->owners); free(g->va_live);
    free(g->phys_live); free(g->handles); free(g);
}
