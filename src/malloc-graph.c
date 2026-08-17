#include "plat.h"

#ifdef AIMDO_CUDA

#define MG_PAGE (8 * M)
#define MG_ERROR CUDA_ERROR_OUT_OF_MEMORY
#define OP_ALLOC 1
#define OP_FREE 2
#define OP_CHILD 3
#define PACK(op,id) ((op)<<30|(id))
#define OPCODE(op) ((op)>>30)
#define OPID(op) ((op)&0x3fffffff)

typedef struct { uint32_t *ops, n, cap, cursor; char *name; bool built; } MgFrame;
typedef struct { CUdeviceptr ptr; size_t size; uint32_t owner, page, pages; bool live; } MgAlloc;
typedef struct MallocGraph {
    CUstream stream;
    MgFrame *frames; uint32_t nf, cf, depth, cstack, *stack;
    MgAlloc *allocs; uint32_t na, ca;
    CUdeviceptr base; uint32_t slots, high;
    int32_t *slot_phys; uint8_t *slot_live, *phys_live;
    CUmemGenericAllocationHandle *phys; uint32_t np;
    CUdeviceptr *small; uint32_t nsmall, csmall;
    size_t peak, used, virtual_bytes;
    bool recording, failed;
    int device;
} MallocGraph;

static _Thread_local MallocGraph *active;
SHARED_EXPORT void malloc_graph_destroy(void *ptr);

static bool grow(void **p, uint32_t *cap, size_t item, uint32_t need) {
    void *q;
    if (need <= *cap) return true;
    *cap = MAX(*cap * 2, 8);
    while (*cap < need) *cap *= 2;
    q = realloc(*p, item * *cap);
    if (!q) return false;
    *p = q;
    return true;
}

static int fail(MallocGraph *g) { g->failed = true; return MG_ERROR; }
static MgFrame *top(MallocGraph *g) { return &g->frames[g->stack[g->depth - 1]]; }

static bool emit(MgFrame *f, uint8_t op, uint32_t id) {
    if (!grow((void **)&f->ops, &f->cap, sizeof(*f->ops), f->n + 1)) return false;
    f->ops[f->n++] = PACK(op,id);
    return true;
}

static bool expect(MallocGraph *g, uint8_t op, uint32_t id) {
    MgFrame *f = top(g);
    if (f->cursor == f->n || f->ops[f->cursor] != PACK(op,id))
        return false;
    f->cursor++;
    return true;
}

static int graph_alloc(MallocGraph *g, CUdeviceptr *out, size_t size) {
    MgFrame *f = top(g);
    size_t rounded = ALIGN_UP(size, MG_PAGE);
    uint32_t id, pages = (uint32_t)(rounded / MG_PAGE), start = 0;
    CUmemAccessDesc access = {{CU_MEM_LOCATION_TYPE_DEVICE, g->device}, CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    CUmemAllocationProp prop = {.type=CU_MEM_ALLOCATION_TYPE_PINNED,
        .location={CU_MEM_LOCATION_TYPE_DEVICE, g->device}};

    if (!g->recording || f->built) {
        if (f->cursor == f->n || OPCODE(f->ops[f->cursor]) != OP_ALLOC) return fail(g);
        id = OPID(f->ops[f->cursor++]);
        if (g->allocs[id].size != size || g->allocs[id].live) return fail(g);
        for (uint32_t i = 0; i < g->allocs[id].pages; i++) {
            uint32_t slot = g->allocs[id].page + i;
            if (g->slot_live[slot] || g->phys_live[g->slot_phys[slot]]) return fail(g);
            g->slot_live[slot] = g->phys_live[g->slot_phys[slot]] = 1;
        }
        g->allocs[id].live = true; g->used += rounded; g->peak = MAX(g->peak, g->used);
        *out = g->allocs[id].ptr;
        return CUDA_SUCCESS;
    }
    id = g->na;
    if (!grow((void **)&g->allocs, &g->ca, sizeof(*g->allocs), id + 1) || !emit(f, OP_ALLOC, id)) return fail(g);
    for (; start + pages <= g->high; start++) {
        uint32_t i;
        for (i = 0; i < pages; i++) {
            int32_t p = g->slot_phys[start + i];
            if (g->slot_live[start + i] || (p >= 0 && g->phys_live[p])) break;
        }
        if (i == pages) break;
    }
    if (start + pages > g->high) start = g->high;
    if (start + pages > g->slots) return fail(g);
    g->high = MAX(g->high, start + pages);
    for (uint32_t i = 0; i < pages; i++) {
        uint32_t slot = start + i, p = 0;
        if (g->slot_phys[slot] < 0) {
            while (p < g->np && g->phys_live[p]) p++;
            if (p == g->np) {
                g->np++;
                if (!CHECK_CU_ERROR(cuMemCreate(&g->phys[p], MG_PAGE, &prop, 0))) return fail(g);
                total_vram_usage += MG_PAGE;
            }
            if (!CHECK_CU_ERROR(cuMemMap(g->base + slot * MG_PAGE, MG_PAGE, 0, g->phys[p], 0)) ||
                !CHECK_CU_ERROR(cuMemSetAccess(g->base + slot * MG_PAGE, MG_PAGE, &access, 1))) return fail(g);
            g->slot_phys[slot] = p;
        } else p = g->slot_phys[slot];
        g->slot_live[slot] = g->phys_live[p] = 1;
    }
    g->allocs[id] = (MgAlloc){g->base + start * MG_PAGE, size, g->stack[g->depth-1], start, pages, true};
    g->na++; g->virtual_bytes = (size_t)g->high * MG_PAGE; g->used += rounded; g->peak = MAX(g->peak, g->used);
    *out = g->allocs[id].ptr;
    return CUDA_SUCCESS;
}

static int graph_free(MallocGraph *g, CUdeviceptr ptr) {
    MgFrame *f = top(g);
    uint32_t id;
    for (id = 0; id < g->na && (g->allocs[id].ptr != ptr || !g->allocs[id].live); id++);
    if (id == g->na || !g->allocs[id].live || g->allocs[id].owner != g->stack[g->depth-1]) {
        g->failed = true;
        return CUDA_SUCCESS;
    }
    if ((!g->recording || f->built) ? !expect(g, OP_FREE, id) : !emit(f, OP_FREE, id)) {
        g->failed = true;
        return CUDA_SUCCESS;
    }
    g->allocs[id].live = false; g->used -= (size_t)g->allocs[id].pages * MG_PAGE;
    for (uint32_t i = 0; i < g->allocs[id].pages; i++) {
        uint32_t slot = g->allocs[id].page + i;
        g->slot_live[slot] = g->phys_live[g->slot_phys[slot]] = 0;
    }
    return CUDA_SUCCESS;
}

int malloc_graph_alloc(CUdeviceptr *p, size_t n, CUstream s) {
    if (!active || s != active->stream || n < MG_PAGE) return -1;
    return graph_alloc(active, p, n);
}
int malloc_graph_free(CUdeviceptr p, CUstream s) {
    uint32_t i;
    if (!active || s != active->stream) return -1;
    for (i=0; i<active->na && active->allocs[i].ptr != p; i++);
    if (i == active->na) {
        for (i=0; i<active->nsmall && active->small[i] != p; i++);
        if (i < active->nsmall) { active->small[i]=active->small[--active->nsmall]; return -1; }
        active->failed = true;
        return -1;
    }
    return graph_free(active, p);
}
void malloc_graph_note(CUdeviceptr p, size_t n, CUstream s) {
    if (active && s==active->stream && n<MG_PAGE &&
        grow((void **)&active->small,&active->csmall,sizeof(*active->small),active->nsmall+1))
        active->small[active->nsmall++]=p;
}

SHARED_EXPORT void *malloc_graph_record(CUstream stream) {
    MallocGraph *g;
    size_t free_bytes, total_bytes;
    if (active || !(g = calloc(1, sizeof(*g)))) return NULL;
    if (!set_devctx_for_current_cuda_device()) { free(g); return NULL; }
    g->stream=stream; g->recording=true; g->device=g_devctx->_device_id;
    if (!CHECK_CU_ERROR(cuMemGetInfo(&free_bytes, &total_bytes))) { free(g); return NULL; }
    g->slots = total_bytes / MG_PAGE;
    g->slot_phys=malloc((size_t)g->slots*sizeof(*g->slot_phys)); g->slot_live=calloc(g->slots,1);
    g->phys_live=calloc(g->slots,1); g->phys=calloc(g->slots,sizeof(*g->phys));
    if (!g->slot_phys || !g->slot_live || !g->phys_live || !g->phys ||
        !CHECK_CU_ERROR(cuMemAddressReserve(&g->base,(size_t)g->slots*MG_PAGE,MG_PAGE,0,0)) ||
        !grow((void **)&g->frames, &g->cf, sizeof(*g->frames), 1) ||
        !grow((void **)&g->stack, &g->cstack, sizeof(*g->stack), 1)) { malloc_graph_destroy(g); return NULL; }
    for(uint32_t i=0;i<g->slots;i++) g->slot_phys[i]=-1;
    g->nf=1; g->stack[0]=0; g->depth=1;
    active=g; return g;
}

SHARED_EXPORT int malloc_graph_push(void *ptr, const char *name) {
    MallocGraph *g=ptr; MgFrame *parent; uint32_t id;
    if (active != g || g->failed) return fail(g);
    parent=top(g);
    for (id=1; id<g->nf && strcmp(g->frames[id].name,name); id++);
    if (id < g->nf) {
        for (uint32_t i=0;i<g->depth;i++) if (g->stack[i]==id) return fail(g);
        if (g->recording && !parent->built &&
            (!parent->n || parent->ops[parent->n-1] != PACK(OP_CHILD,id))) return fail(g);
    } else {
        if (!g->recording || parent->built || !grow((void **)&g->frames,&g->cf,sizeof(*g->frames),g->nf+1)) return fail(g);
        id=g->nf++; g->frames[id].name=strdup(name);
    }
    if ((!g->recording || parent->built) ? !expect(g,OP_CHILD,id) : !emit(parent,OP_CHILD,id)) return fail(g);
    if (!grow((void **)&g->stack,&g->cstack,sizeof(*g->stack),g->depth+1)) return fail(g);
    g->stack[g->depth++]=id; g->frames[id].cursor=0; return 0;
}

SHARED_EXPORT int malloc_graph_pop(void *ptr) {
    MallocGraph *g=ptr; MgFrame *f;
    if (active != g || g->failed) return fail(g);
    f=top(g);
    if ((!g->recording || f->built) && f->cursor != f->n) return fail(g);
    for (uint32_t i=0;i<g->na;i++) if (g->allocs[i].live && g->allocs[i].owner==g->stack[g->depth-1]) return fail(g);
    f->built=true;
    if (--g->depth) return 0;
    g->recording=false; active=NULL; return 0;
}

SHARED_EXPORT int malloc_graph_replay(void *ptr, CUstream stream) {
    MallocGraph *g=ptr;
    if (active || g->failed || g->recording || stream != g->stream) return fail(g);
    g->frames[0].cursor=0; g->stack[0]=0; g->depth=1; active=g; return 0;
}
SHARED_EXPORT size_t malloc_graph_stat(void *ptr, int which) {
    MallocGraph *g=ptr; return which==0 ? g->peak : which==1 ? g->virtual_bytes : (size_t)g->np*MG_PAGE;
}
SHARED_EXPORT void malloc_graph_destroy(void *ptr) {
    MallocGraph *g=ptr;
    if (!g) return;
    if (active==g) active=NULL;
    for(uint32_t i=0;i<g->high;i++) if(g->slot_phys && g->slot_phys[i]>=0) cuMemUnmap(g->base+(size_t)i*MG_PAGE,MG_PAGE);
    for(uint32_t i=0;i<g->np;i++) cuMemRelease(g->phys[i]);
    if(g->base) cuMemAddressFree(g->base,(size_t)g->slots*MG_PAGE);
    if (g_devctx) total_vram_usage -= (size_t)g->np*MG_PAGE;
    for(uint32_t i=0;i<g->nf;i++){free(g->frames[i].ops);free(g->frames[i].name);} free(g->frames);free(g->allocs);free(g->phys);free(g->slot_phys);free(g->slot_live);free(g->phys_live);free(g->small);free(g->stack);free(g);
}
#else
int malloc_graph_alloc(CUdeviceptr *p, size_t n, CUstream s) { return -1; }
int malloc_graph_free(CUdeviceptr p, CUstream s) { return -1; }
void malloc_graph_note(CUdeviceptr p, size_t n, CUstream s) {}
#endif
