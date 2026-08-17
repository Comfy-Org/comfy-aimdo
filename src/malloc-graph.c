#include "plat.h"
#include "malloc-graph.h"

#define PAGE (8ULL * M)

enum { NODE_ALLOC, NODE_FREE, NODE_CHILD };
typedef struct Scope Scope;
typedef struct { int kind; size_t size, allocation; CUdeviceptr ptr; Scope *child; } Node;
typedef struct { CUdeviceptr ptr; size_t size, pages, owner; bool live; } Allocation;
typedef struct { CUdeviceptr ptr; size_t pages; } Extent;
typedef struct { CUdeviceptr ptr; size_t page; } Mapping;

struct Scope {
    char *name;
    Node *nodes;
    size_t count, cap, cursor, recording_runs;
    Scope *parent;
    Scope *open_child;
};

typedef struct {
    CUstream stream;
    CUdeviceptr base;
    Scope root, *active;
    Scope **scopes;
    size_t scope_count, scope_cap, va_high;
    Allocation *allocs;
    size_t alloc_count, alloc_cap;
    Extent *free_va;
    size_t free_count, free_cap;
    CUmemGenericAllocationHandle *pages;
    bool *page_free;
    size_t page_count, page_cap, used, peak, virtual_bytes;
    Mapping *maps;
    size_t map_count, map_cap;
    bool recording, failed;
} Graph;

static _Thread_local Graph *current;

static void fail(Graph *g) { g->failed = true; }
static void *grow(void *p, size_t *cap, size_t n, size_t size) {
    if (n < *cap) return p;
    *cap = *cap ? *cap * 2 : 16;
    return realloc(p, *cap * size);
}
static Node *node_add(Scope *s) {
    s->nodes = grow(s->nodes, &s->cap, s->count, sizeof(*s->nodes));
    return &s->nodes[s->count++];
}
static Scope *find_child(Scope *s, const char *name) {
    Graph *g = current;
    for (size_t i = 0; i < g->scope_count; i++)
        if (g->scopes[i] != s && !strcmp(g->scopes[i]->name, name)) return g->scopes[i];
    return NULL;
}
static bool event(Graph *g, int kind, size_t size, CUdeviceptr ptr) {
    Scope *s = g->active;
    if (g->recording) {
        if (kind != NODE_CHILD) s->open_child = NULL;
        Node *n = node_add(s); n->kind = kind; n->size = size; n->ptr = ptr; n->child = NULL;
        return true;
    }
    if (s->cursor >= s->count) { fail(g); return false; }
    Node *n = &s->nodes[s->cursor++];
    if (n->kind != kind || n->size != size || n->ptr != ptr) { fail(g); return false; }
    return true;
}
static Allocation *allocation(Graph *g, CUdeviceptr ptr) {
    for (size_t i = g->alloc_count; i--;) if (g->allocs[i].ptr == ptr && g->allocs[i].live) return &g->allocs[i];
    for (size_t i = g->alloc_count; i--;) if (g->allocs[i].ptr == ptr) return &g->allocs[i];
    return NULL;
}
static bool add_page(Graph *g) {
    CUmemAllocationProp prop = {0}; CUdevice dev;
    if (!CHECK_CU(g_cuda.p_cuCtxGetDevice(&dev))) return false;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = dev;
    g->pages = grow(g->pages, &g->page_cap, g->page_count, sizeof(*g->pages));
    g->page_free = realloc(g->page_free, g->page_cap * sizeof(*g->page_free));
    if (!CHECK_CU(g_cuda.p_cuMemCreate(&g->pages[g->page_count], PAGE, &prop, 0))) return false;
    g->page_free[g->page_count++] = true;
    total_vram_usage += PAGE;
    return true;
}
static bool mapped_pages_free(Graph *g, CUdeviceptr ptr, size_t pages) {
    for (size_t j = 0; j < pages; j++)
        for (size_t i = 0; i < g->map_count; i++)
            if (g->maps[i].ptr == ptr + j * PAGE && !g->page_free[g->maps[i].page]) return false;
    return true;
}
static CUdeviceptr acquire_va(Graph *g, size_t pages, bool *mapped) {
    size_t best = g->free_count;
    CUdeviceptr best_ptr = 0;
    for (size_t i = 0; i < g->free_count; i++) for (size_t j = 0; j + pages <= g->free_va[i].pages; j++) {
        CUdeviceptr ptr = g->free_va[i].ptr + j * PAGE;
        if (mapped_pages_free(g, ptr, pages) && (best == g->free_count || ptr < best_ptr)) { best = i; best_ptr = ptr; }
    }
    if (best != g->free_count) {
        CUdeviceptr p = best_ptr;
        *mapped = true;
        size_t before = (p - g->free_va[best].ptr) / PAGE;
        size_t after = g->free_va[best].pages - before - pages;
        if (before && after) {
            g->free_va = grow(g->free_va, &g->free_cap, g->free_count, sizeof(*g->free_va));
            g->free_va[g->free_count++] = (Extent){p + pages * PAGE, after};
            g->free_va[best].pages = before;
        } else if (before) g->free_va[best].pages = before;
        else if (after) { g->free_va[best].ptr = p + pages * PAGE; g->free_va[best].pages = after; }
        else g->free_va[best] = g->free_va[--g->free_count];
        return p;
    }
    CUdeviceptr p = g->base + g->va_high * PAGE; *mapped = false;
    g->va_high += pages; g->virtual_bytes = g->va_high * PAGE;
    return p;
}
static bool map_pages(Graph *g, CUdeviceptr va, size_t pages) {
    CUdevice dev; g_cuda.p_cuCtxGetDevice(&dev);
    CUmemAccessDesc access = {{CU_MEM_LOCATION_TYPE_DEVICE, dev}, CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    for (size_t j = 0; j < pages; j++) {
        size_t i;
        for (i = 0; i < g->page_count && !g->page_free[i]; i++);
        if (i == g->page_count && !add_page(g)) return false;
        g->page_free[i] = false;
        if (!CHECK_CU(g_cuda.p_cuMemMap(va + j * PAGE, PAGE, 0, g->pages[i], 0)) ||
            !CHECK_CU(g_cuda.p_cuMemSetAccess(va + j * PAGE, PAGE, &access, 1))) return false;
        g->maps = grow(g->maps, &g->map_cap, g->map_count, sizeof(*g->maps));
        g->maps[g->map_count++] = (Mapping){va + j * PAGE, i};
    }
    return true;
}

void *malloc_graph_record(CUstream stream) {
    if (current) return NULL;
    Graph *g = calloc(1, sizeof(*g));
    if (!g || !CHECK_CU(g_cuda.p_cuMemAddressReserve(&g->base, 1ULL << 40, PAGE, 0, 0))) { free(g); return NULL; }
    g->stream = stream; g->active = &g->root; g->recording = true; current = g;
    return g;
}
bool malloc_graph_push(void *opaque, const char *name) {
    Graph *g = opaque; if (g != current || g->failed) return false;
    Scope *s = g->active;
    if (g->recording) {
        Scope *child = find_child(s, name);
        if (child) {
            if (s->open_child != child) { fail(g); return false; }
            for (Scope *p = s; p; p = p->parent) if (p == child) { fail(g); return false; }
            Node *n = node_add(s); n->kind = NODE_CHILD; n->child = child; n->size = n->ptr = 0;
            child->cursor = 0; child->recording_runs++; g->recording = false; g->active = child; return true;
        }
        child = calloc(1, sizeof(*child)); child->name = strdup(name); child->parent = s;
        g->scopes = grow(g->scopes, &g->scope_cap, g->scope_count, sizeof(*g->scopes));
        g->scopes[g->scope_count++] = child;
        s->open_child = NULL;
        Node *n = node_add(s); n->kind = NODE_CHILD; n->child = child; n->size = n->ptr = 0; g->active = child; return true;
    }
    if (s->cursor >= s->count || s->nodes[s->cursor].kind != NODE_CHILD || strcmp(s->nodes[s->cursor].child->name, name)) { fail(g); return false; }
    Scope *child = s->nodes[s->cursor++].child;
    for (Scope *p = s; p; p = p->parent) if (p == child) { fail(g); return false; }
    child->cursor = 0; child->parent = s; g->active = child; return true;
}
bool malloc_graph_pop(void *opaque) {
    Graph *g = opaque; if (g != current) return false;
    Scope *s = g->active;
    for (size_t i = 0; i < g->alloc_count; i++) if (g->allocs[i].live && g->allocs[i].owner == (size_t)s) fail(g);
    if (!g->recording && s->cursor != s->count) fail(g);
    if (s != &g->root) {
        g->active = s->parent;
        if (s->recording_runs) { s->recording_runs--; g->recording = true; }
        if (g->recording) g->active->open_child = s;
    } else current = NULL;
    return !g->failed;
}
bool malloc_graph_replay(void *opaque, CUstream stream) {
    Graph *g = opaque; if (current || g->failed || stream != g->stream) { fail(g); return false; }
    g->recording = false; g->root.cursor = 0; g->active = &g->root; current = g; return true;
}
bool malloc_graph_failed(void *opaque) { return ((Graph *)opaque)->failed; }
bool malloc_graph_reject_external(CUstream stream) {
    if (!current || current->stream != stream) return false;
    fail(current);
    return true;
}
size_t malloc_graph_stat(void *opaque, int which) { Graph *g = opaque; return which == 0 ? g->peak : which == 1 ? g->virtual_bytes : g->page_count * PAGE; }
void malloc_graph_destroy(void *opaque) {
    Graph *g = opaque; if (!g) return; if (current == g) current = NULL;
    for (size_t i = 0; i < g->map_count; i++) g_cuda.p_cuMemUnmap(g->maps[i].ptr, PAGE);
    g_cuda.p_cuMemAddressFree(g->base, 1ULL << 40);
    for (size_t i = 0; i < g->page_count; i++) g_cuda.p_cuMemRelease(g->pages[i]);
    total_vram_usage -= g->page_count * PAGE;
    free(g->root.nodes);
    for (size_t i = 0; i < g->scope_count; i++) { free(g->scopes[i]->name); free(g->scopes[i]->nodes); free(g->scopes[i]); }
    free(g->scopes); free(g->allocs); free(g->free_va); free(g->pages); free(g->page_free); free(g->maps); free(g);
}
CUresult malloc_graph_alloc(CUdeviceptr *ptr, size_t size, CUstream stream) {
    Graph *g = current; if (!g || stream != g->stream || size < PAGE) return -1;
    size_t pages = (size + PAGE - 1) / PAGE;
    if (!g->recording) {
        Scope *s = g->active;
        if (s->cursor >= s->count || s->nodes[s->cursor].kind != NODE_ALLOC || s->nodes[s->cursor].size != size) { fail(g); return CUDA_ERROR_OUT_OF_MEMORY; }
        *ptr = s->nodes[s->cursor].ptr;
        Allocation *a = &g->allocs[s->nodes[s->cursor].allocation]; if (a->live) { fail(g); return CUDA_ERROR_OUT_OF_MEMORY; }
        a->live = true; a->owner = (size_t)s; g->used += pages * PAGE; s->cursor++; return CUDA_SUCCESS;
    }
    bool mapped; CUdeviceptr va = acquire_va(g, pages, &mapped); if (!va || (!mapped && !map_pages(g, va, pages))) return CUDA_ERROR_OUT_OF_MEMORY;
    if (mapped) for (size_t j = 0; j < pages; j++) for (size_t i = 0; i < g->map_count; i++)
        if (g->maps[i].ptr == va + j * PAGE) g->page_free[g->maps[i].page] = false;
    g->allocs = grow(g->allocs, &g->alloc_cap, g->alloc_count, sizeof(*g->allocs));
    size_t allocation_index = g->alloc_count;
    g->allocs[g->alloc_count++] = (Allocation){va, size, pages, (size_t)g->active, true};
    g->used += pages * PAGE; if (g->used > g->peak) g->peak = g->used; *ptr = va;
    event(g, NODE_ALLOC, size, va); g->active->nodes[g->active->count - 1].allocation = allocation_index; return CUDA_SUCCESS;
}
CUresult malloc_graph_free(CUdeviceptr ptr, CUstream stream) {
    Graph *g = current; Allocation *a = g ? allocation(g, ptr) : NULL;
    if (!g || stream != g->stream || !a) return -1;
    if (!a->live || a->owner != (size_t)g->active || !event(g, NODE_FREE, 0, ptr)) { fail(g); return CUDA_ERROR_OUT_OF_MEMORY; }
    a->live = false; g->used -= a->pages * PAGE;
    if (g->recording) {
        g->free_va = grow(g->free_va, &g->free_cap, g->free_count, sizeof(*g->free_va));
        g->free_va[g->free_count++] = (Extent){ptr, a->pages};
        for (size_t j = 0; j < a->pages; j++) for (size_t i = 0; i < g->map_count; i++)
            if (g->maps[i].ptr == ptr + j * PAGE) g->page_free[g->maps[i].page] = true;
        for (size_t i = 0; i < g->free_count; i++) for (size_t j = i + 1; j < g->free_count; j++) {
            Extent *x = &g->free_va[i], *y = &g->free_va[j];
            if (x->ptr + x->pages * PAGE == y->ptr) { x->pages += y->pages; *y = g->free_va[--g->free_count]; j--; }
            else if (y->ptr + y->pages * PAGE == x->ptr) { x->ptr = y->ptr; x->pages += y->pages; *y = g->free_va[--g->free_count]; j--; }
        }
    }
    return CUDA_SUCCESS;
}
