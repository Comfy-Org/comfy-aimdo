#include "plat.h"
#include "malloc-rogue.h"
#include "vmm-ref.h"

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
    bool rogue_owned;
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
    CUdeviceptr rogue_ptr;
    uint32_t owner_depth;
    Event *allocation;

    SmallRange *next;
};

struct AllocationState {
    VirtualPage virtual_pages[MG_PAGES];
    bool physical_live[MG_PAGES];
};

typedef struct {
    VirtualRange *base;
    VirtualRange *small_base;
    CUstream stream;
    int device;
    void *owner_thread;

    Event root;
    State *state;
    Event *rogue_candidates;

    AllocationState allocations;
    SmallRange *small_ranges;
    SmallRange *small_unusable;
    CUdeviceptr black_holes[MG_PAGES];
    int va_phys[MG_PAGES];
    PhysicalPage *physical_pages[MG_PAGES];
    PhysicalPage *mapped_pages[MG_PAGES];
    PhysicalPage *small_physical_pages[MG_SMALL_PAGES];
    PhysicalPage *small_mapped_pages[MG_SMALL_PAGES];

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
    bool handoff_attempted;
} MallocGraph;

static _Thread_local MallocGraph *active_graph;

static bool rogue_va(MallocGraph *g, size_t va);
static bool rogue_phys(MallocGraph *g, size_t phys);
static bool small_unavailable(MallocGraph *g, size_t offset, size_t bytes);
static CUresult map_reference(MallocGraph *g, CUdeviceptr address,
                              PhysicalPage *page, PhysicalPage **mapping);

static CUdeviceptr candidate_ptr(MallocGraph *g, Event *candidate) {
    VirtualRange *range = candidate->type == EV_ALLOC_SMALL ? g->small_base : g->base;
    return virtual_range_get(range) + candidate->value *
           (candidate->type == EV_ALLOC_SMALL ? 1 : MG_PAGE);
}

static bool graph_failed(MallocGraph *g) {
    for (Event *candidate = g->rogue_candidates; candidate;
         candidate = candidate->allocation_previous) {
        if (rogue_candidate_freed(candidate_ptr(g, candidate))) {
            g->failed = true;
            break;
        }
    }
    return g->failed;
}

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
        if (type == EV_ALLOC) {
            size_t pages = ALIGN_UP(event->bytes, MG_PAGE) / MG_PAGE;
            size_t i;
            for (i = 0; i < pages; i++) {
                size_t va = event->value + i;
                if (rogue_va(g, va) || rogue_phys(g, g->va_phys[va])) {
                    break;
                }
            }
            if (i < pages) {
                continue;
            }
        } else if (type == EV_ALLOC_SMALL &&
                   small_unavailable(g, event->value,
                                     ALIGN_UP(event->bytes, MG_SMALL_ALIGN))) {
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

static bool sever_event(MallocGraph *g, Event *event) {
    Event *parent = event->previous;
    size_t count = parent->next_count + event->next_count - 1;
    Event **next = count ? malloc(count * sizeof(*next)) : NULL;
    RETURN_G_FAILED(count && !next, false);

    size_t j = 0;
    for (size_t i = 0; i < parent->next_count; i++) {
        if (parent->next[i] == event) {
            for (size_t k = 0; k < event->next_count; k++) {
                next[j++] = event->next[k];
                event->next[k]->previous = parent;
            }
        } else {
            next[j++] = parent->next[i];
        }
    }

    if (g->state->cursor == event) {
        g->state->cursor = parent;
    }
    free(parent->next);
    free(event->next);
    parent->next = next;
    parent->next_count = count;
    event->next = NULL;
    event->next_count = 0;
    event->previous = NULL;
    return true;
}

static bool collect_rogue_candidates(MallocGraph *g) {
    Event *allocation = g->state->allocations;

    while (allocation) {
        Event *previous = allocation->allocation_previous;
        if (!register_rogue_candidate(candidate_ptr(g, allocation))) {
            g->failed = true;
            return false;
        }
        if (!sever_event(g, allocation)) {
            unregister_rogue_candidate(candidate_ptr(g, allocation));
            return false;
        }
        allocation->allocation_previous = g->rogue_candidates;
        g->rogue_candidates = allocation;
        allocation = previous;
    }
    g->state->allocations = NULL;
    return true;
}

static bool is_rogue_candidate(MallocGraph *g, Event *event) {
    for (Event *candidate = g->rogue_candidates; candidate;
         candidate = candidate->allocation_previous) {
        if (candidate == event) {
            return true;
        }
    }
    return false;
}

static bool rogue_va(MallocGraph *g, size_t va) {
    CUdeviceptr ptr = g->black_holes[va];
    if (ptr && !rogue_exists(ptr)) {
        if (map_reference(g, virtual_range_get(g->base) + va * MG_PAGE,
                          g->physical_pages[g->va_phys[va]], &g->mapped_pages[va])) {
            g->failed = true;
            return true;
        }
        g->black_holes[va] = 0;
    } else if (ptr) {
        return true;
    }
    for (Event *candidate = g->rogue_candidates; candidate;
         candidate = candidate->allocation_previous) {
        if (candidate->type == EV_ALLOC) {
            size_t pages = ALIGN_UP(candidate->bytes, MG_PAGE) / MG_PAGE;
            if (va >= candidate->value && va < candidate->value + pages) {
                return true;
            }
        }
    }
    return false;
}

static bool rogue_phys(MallocGraph *g, size_t phys) {
    for (Event *candidate = g->rogue_candidates; candidate;
         candidate = candidate->allocation_previous) {
        if (candidate->type == EV_ALLOC) {
            size_t pages = ALIGN_UP(candidate->bytes, MG_PAGE) / MG_PAGE;
            for (size_t i = 0; i < pages; i++) {
                if (g->va_phys[candidate->value + i] == (int)phys) {
                    return true;
                }
            }
        }
    }
    return false;
}

static void prune_small_unusable(MallocGraph *g) {
    SmallRange **entry = &g->small_unusable;
    while (*entry) {
        SmallRange *range = *entry;
        if (!rogue_exists(range->rogue_ptr)) {
            *entry = range->next;
            free(range);
            continue;
        }
        entry = &range->next;
    }
}

static bool small_unavailable(MallocGraph *g, size_t offset, size_t bytes) {
    size_t end = offset + bytes;

    prune_small_unusable(g);
    for (SmallRange *range = g->small_unusable; range; range = range->next) {
        if (offset < range->offset + range->bytes && range->offset < end) {
            return true;
        }
    }
    for (Event *candidate = g->rogue_candidates; candidate;
         candidate = candidate->allocation_previous) {
        if (rogue_candidate_freed(candidate_ptr(g, candidate))) {
            continue;
        }
        if (candidate->type == EV_ALLOC_SMALL) {
            size_t first = candidate->value / MG_PAGE * MG_PAGE;
            size_t candidate_end = ALIGN_UP(candidate->value + candidate->bytes, MG_PAGE);
            if (offset < candidate_end && first < end) {
                return true;
            }
        }
    }
    return false;
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

static CUresult create_page(MallocGraph *g, PhysicalPage **page) {
    CUresult r;

    vbars_free(budget_deficit(MG_PAGE));
    r = physical_page_alloc(page, MG_PAGE, g->device);
    if (r == CUDA_ERROR_OUT_OF_MEMORY) {
        vbars_free(MG_PAGE);
        r = physical_page_alloc(page, MG_PAGE, g->device);
    }
    return r;
}

static CUresult map_reference(MallocGraph *g, CUdeviceptr address,
                              PhysicalPage *page, PhysicalPage **mapping) {
    CUmemAccessDesc access = {.location = {CU_MEM_LOCATION_TYPE_DEVICE, g->device},
                              .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    PhysicalPage *reference = physical_page_ref(page, 0);
    if (!reference) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult result = cuMemMap(address, MG_PAGE, 0, physical_page_get(page), 0);
    if (!result) {
        reference->address = address;
        result = cuMemSetAccess(address, MG_PAGE, &access, 1);
    }
    if (result) {
        physical_page_unref(reference);
    } else {
        *mapping = reference;
    }
    return result;
}

static int map_page(MallocGraph *g, size_t va, size_t phys) {
    CUdeviceptr addr = virtual_range_get(g->base) + va * MG_PAGE;
    CUresult r;

    if (phys == g->phys_count) {
        r = create_page(g, &g->physical_pages[phys]);
        if (r) {
            return r;
        }
        g->phys_count++;
    }

    if ((r = map_reference(g, addr, g->physical_pages[phys], &g->mapped_pages[va]))) {
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

static void scrub_rogue_candidates(MallocGraph *g, AllocationState *allocations,
                                   SmallRange **small_ranges) {
    if (allocations) {
        for (size_t va = 0; va < g->va_count; va++) {
            VirtualPage *first = &allocations->virtual_pages[va];
            if (!first->allocation || !is_rogue_candidate(g, first->allocation)) {
                continue;
            }
            for (size_t i = 0; i < first->va_span; i++) {
                allocations->physical_live[g->va_phys[va + i]] = false;
            }
            *first = (VirtualPage){};
        }
    }

    if (small_ranges) {
        SmallRange **entry = small_ranges;
        while (*entry) {
            SmallRange *range = *entry;
            if (range->allocation && is_rogue_candidate(g, range->allocation)) {
                *entry = range->next;
                free(range);
            } else {
                entry = &range->next;
            }
        }
    }
}

static void scrub_rogue_snapshots(MallocGraph *g, Event *event) {
    scrub_rogue_candidates(g, event->snapshot, &event->small_snapshot);
    for (size_t i = 0; i < event->next_count; i++) {
        Event *next = event->next[i];
        if (next->type == EV_CALL) {
            scrub_rogue_snapshots(g, next->scope);
        }
        scrub_rogue_snapshots(g, next);
    }
}

static bool handoff_rogues(MallocGraph *g) {
    g->handoff_attempted = true;
    if (cuCtxSynchronize()) {
        g->failed = true;
        return false;
    }

    PhysicalPage **pages = NULL;
    bool freed = false;
    for (Event *candidate = g->rogue_candidates; candidate;
         candidate = candidate->allocation_previous) {
        if (!pages && !(pages = malloc(MG_PAGES * sizeof(*pages)))) {
            g->failed = true;
            return false;
        }

        VirtualRange *range;
        CUdeviceptr ptr;
        size_t count;

        if (candidate->type == EV_ALLOC_SMALL) {
            range = g->small_base;
            ptr = virtual_range_get(range) + candidate->value;
            size_t first = candidate->value / MG_PAGE;
            size_t last = (candidate->value + ALIGN_UP(candidate->bytes, MG_SMALL_ALIGN) - 1) / MG_PAGE;
            count = last - first + 1;
            for (size_t i = 0; i < count; i++) {
                pages[i] = g->small_physical_pages[first + i];
            }
        } else {
            range = g->base;
            ptr = virtual_range_get(range) + candidate->value * MG_PAGE;
            count = ALIGN_UP(candidate->bytes, MG_PAGE) / MG_PAGE;
            for (size_t i = 0; i < count; i++) {
                pages[i] = g->physical_pages[g->va_phys[candidate->value + i]];
            }
        }

        RogueHandoff handoff = handoff_rogue(range, ptr, pages, count);
        if (!handoff) {
            free(pages);
            g->failed = true;
            return false;
        }
        if (handoff == ROGUE_HANDOFF_FREED) {
            freed = true;
        } else {
            candidate->rogue_owned = true;
        }
        if (candidate->rogue_owned && candidate->type == EV_ALLOC) {
            for (size_t i = 0; i < count; i++) {
                g->black_holes[candidate->value + i] = ptr;
            }
        }
    }
    free(pages);

    if (freed && cuCtxSynchronize()) {
        g->failed = true;
        return false;
    }
    return true;
}

static void free_rogue_candidates(MallocGraph *g) {
    while (g->rogue_candidates) {
        Event *candidate = g->rogue_candidates;
        g->rogue_candidates = candidate->allocation_previous;
        unregister_rogue_candidate(candidate_ptr(g, candidate));
        free(candidate);
    }
}

static bool finalize_rogues(MallocGraph *g) {
    if (!g->rogue_candidates) {
        return true;
    }
    if (!handoff_rogues(g)) {
        return false;
    }

    PhysicalPage **replacements = calloc(g->phys_count, sizeof(*replacements));
    SmallRange *unusable = NULL;
    if (g->phys_count && !replacements) {
        goto fail;
    }

    for (Event *candidate = g->rogue_candidates; candidate;
         candidate = candidate->allocation_previous) {
        if (!candidate->rogue_owned) {
            continue;
        }
        if (candidate->type == EV_ALLOC_SMALL) {
            SmallRange *range = malloc(sizeof(*range));
            if (!range) {
                goto fail;
            }
            *range = (SmallRange){
                .offset = candidate->value / MG_PAGE * MG_PAGE,
                .bytes = ALIGN_UP(candidate->value + candidate->bytes, MG_PAGE) -
                         candidate->value / MG_PAGE * MG_PAGE,
                .rogue_ptr = virtual_range_get(g->small_base) + candidate->value,
                .next = unusable};
            unusable = range;
            continue;
        }

        size_t count = ALIGN_UP(candidate->bytes, MG_PAGE) / MG_PAGE;
        for (size_t i = 0; i < count; i++) {
            size_t phys = g->va_phys[candidate->value + i];
            if (!replacements[phys] && create_page(g, &replacements[phys])) {
                goto fail;
            }
        }
    }

    bool remap_failed = false;
    for (size_t va = 0; va < g->va_count; va++) {
        int phys = g->va_phys[va];
        if (phys < 0 || !replacements[phys]) {
            continue;
        }

        if (physical_page_unref(g->mapped_pages[va])) {
            remap_failed = true;
            continue;
        }
        g->mapped_pages[va] = NULL;
        if (!g->black_holes[va] &&
            map_reference(g, virtual_range_get(g->base) + va * MG_PAGE,
                          replacements[phys], &g->mapped_pages[va])) {
            remap_failed = true;
        }
    }
    for (size_t phys = 0; phys < g->phys_count; phys++) {
        if (replacements[phys]) {
            PhysicalPage *page = g->physical_pages[phys];
            g->physical_pages[phys] = replacements[phys];
            physical_page_unref(page);
        }
    }
    free(replacements);
    if (remap_failed) {
        free_small_ranges(unusable);
        g->failed = true;
        return false;
    }

    while (unusable) {
        SmallRange *next = unusable->next;
        unusable->next = g->small_unusable;
        g->small_unusable = unusable;
        unusable = next;
    }
    scrub_rogue_candidates(g, &g->allocations, &g->small_ranges);
    scrub_rogue_snapshots(g, &g->root);
    g->used = allocation_used(g);

    free_rogue_candidates(g);
    return true;

fail:
    if (replacements) {
        for (size_t phys = 0; phys < g->phys_count; phys++) {
            if (replacements[phys]) {
                physical_page_unref(replacements[phys]);
            }
        }
    }
    free(replacements);
    free_small_ranges(unusable);
    g->failed = true;
    return false;
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

static void consider_small_interval(size_t position, size_t offset, size_t bytes,
                                    size_t *next_offset, size_t *next_end) {
    size_t end = offset + bytes;
    if (end <= position) {
        return;
    }
    if (offset <= position) {
        *next_offset = position;
        *next_end = MAX(*next_end, end);
    } else if (offset < *next_offset) {
        *next_offset = offset;
        *next_end = end;
    } else if (offset == *next_offset) {
        *next_end = MAX(*next_end, end);
    }
}

static size_t small_allocation_offset(MallocGraph *g, size_t bytes) {
    size_t position = 0;
    size_t offset = g->small_size;
    size_t smallest_hole = SIZE_MAX;

    prune_small_unusable(g);
    while (position < g->small_size) {
        size_t next_offset = SIZE_MAX;
        size_t next_end = 0;

        for (SmallRange *range = g->small_ranges; range; range = range->next) {
            consider_small_interval(position, range->offset, range->bytes,
                                    &next_offset, &next_end);
        }
        for (SmallRange *range = g->small_unusable; range; range = range->next) {
            consider_small_interval(position, range->offset, range->bytes,
                                    &next_offset, &next_end);
        }
        for (Event *candidate = g->rogue_candidates; candidate;
             candidate = candidate->allocation_previous) {
            if (candidate->type == EV_ALLOC_SMALL) {
                size_t first = candidate->value / MG_PAGE * MG_PAGE;
                consider_small_interval(position, first,
                                        ALIGN_UP(candidate->value + candidate->bytes, MG_PAGE) - first,
                                        &next_offset, &next_end);
            }
        }

        if (next_offset == SIZE_MAX) {
            next_offset = g->small_size;
        }
        size_t hole = next_offset - position;
        if (hole >= bytes && hole < smallest_hole) {
            offset = position;
            smallest_hole = hole;
        }
        if (next_offset == g->small_size) {
            break;
        }
        position = next_end;
    }
    if (smallest_hole == SIZE_MAX) {
        offset = MAX(offset, position);
    }
    return offset;
}

bool malloc_graph_alloc(CUdeviceptr *ptr, size_t size, CUstream stream) {
    MallocGraph *g = active_graph;

    if (!g || graph_failed(g) || g->paused || stream != g->stream) {
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
                ? virtual_range_get(g->small_base) + match->value
                : virtual_range_get(g->base) + match->value * MG_PAGE;
            return true;
        }
        RETURN_G_FAILED(g->failed, true);
        RETURN_G_FAILED(!start_recording(g), true);
    }

    if (size < MG_SMALL_LIMIT) {
        size_t bytes = ALIGN_UP(size, MG_SMALL_ALIGN);
        size_t offset = small_allocation_offset(g, bytes);
        SmallRange **insert_at = &g->small_ranges;
        while (*insert_at && (*insert_at)->offset < offset) {
            insert_at = &(*insert_at)->next;
        }

        size_t end = offset + bytes;
        if (end > g->small_size) {
            size_t pages = ALIGN_UP(end, MG_PAGE) / MG_PAGE;
            RETURN_G_FAILED(pages > MG_SMALL_PAGES, true);

            if (g->small_pages < pages) {
                PhysicalPage **page = &g->small_physical_pages[g->small_pages];
                CUdeviceptr addr = virtual_range_get(g->small_base) + g->small_pages * MG_PAGE;
                CUresult r = create_page(g, page);
                RETURN_G_FAILED(r, true);
                g->small_pages++;
                RETURN_G_FAILED(map_reference(g, addr, *page,
                                &g->small_mapped_pages[g->small_pages - 1]), true);
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
        *ptr = virtual_range_get(g->small_base) + offset;
        return true;
    }

    size_t pages = ALIGN_UP(size, MG_PAGE) / MG_PAGE;

    while (va + pages <= g->va_count) {
        size_t j;
        for (j = 0; j < pages; j++) {
            int phys = g->va_phys[va + j];
            if (rogue_va(g, va + j) || g->allocations.physical_live[phys] ||
                rogue_phys(g, phys)) {
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
    RETURN_G_FAILED(g->failed, true);
    if (va + pages > g->va_count) {
        va = g->va_count;

        RETURN_G_FAILED(va + pages > MG_PAGES, true);

        g->va_count = va + pages;
    }

    for (size_t j = 0; j < pages; j++) {
        if (g->va_phys[va + j] < 0) {
            size_t p = 0;
            while (p < g->phys_count &&
                   (g->allocations.physical_live[p] || rogue_phys(g, p))) {
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
    *ptr = virtual_range_get(g->base) + va * MG_PAGE;
    return true;
}

bool malloc_graph_free(CUdeviceptr ptr, size_t size, CUstream stream, int *result) {
    MallocGraph *g = active_graph;

    if (!g || graph_failed(g) || g->paused || stream != g->stream) {
        return false;
    }

    CUdeviceptr base = virtual_range_get(g->base);
    CUdeviceptr small_base = virtual_range_get(g->small_base);
    bool small = ptr >= small_base && ptr < small_base + MG_SMALL_PAGES * MG_PAGE;
    if (!small && (ptr < base || ptr >= base + MG_PAGES * MG_PAGE)) {
        RETURN_G_FAILED(size, false);
        return false;
    }

    size_t value = small ? ptr - small_base : (ptr - base) / MG_PAGE;

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

    if (!(g->base = virtual_range_alloc(MG_PAGES * MG_PAGE, MG_PAGE))) {
        goto fail;
    }
    if (!(g->small_base = virtual_range_alloc(MG_SMALL_PAGES * MG_PAGE, MG_PAGE))) {
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
    virtual_range_unref(g->small_base);
fail_address:
    virtual_range_unref(g->base);
fail:
    free_small_ranges(g->root.small_snapshot);
    free(g->root.snapshot);
    free(g);
    return NULL;
}

SHARED_EXPORT bool malloc_graph_pause(void *handle, bool paused, bool sync) {
    MallocGraph *g = handle;

    if (!g || g != active_graph || graph_failed(g)) {
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

    if (!g || g != active_graph || graph_failed(g)) {
        return false;
    }
    g->stream = stream;
    return true;
}

SHARED_EXPORT bool malloc_graph_push(void *handle, const char *name) {
    MallocGraph *g = handle;

    if (!g || graph_failed(g)) {
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

    if (!g || g != active_graph || graph_failed(g)) {
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
    RETURN_G_FAILED(!collect_rogue_candidates(g), false);
    RETURN_G_FAILED(g->state->recording && !event(g, EV_END, 0, 0), false);

    if (!g->state->next && !finalize_rogues(g)) {
        return false;
    }

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

    if (g->rogue_candidates) {
        if (!g->handoff_attempted) {
            handoff_rogues(g);
        }
        free_rogue_candidates(g);
    }

    while (g->state) {
        State *state = g->state;
        g->state = state->next;
        free(state);
    }

    for (size_t i = 0; i < g->va_count; i++) {
        if (g->mapped_pages[i]) {
            physical_page_unref(g->mapped_pages[i]);
        }
    }

    for (size_t i = 0; i < g->phys_count; i++) {
        physical_page_unref(g->physical_pages[i]);
    }

    for (size_t i = 0; i < g->small_pages; i++) {
        physical_page_unref(g->small_mapped_pages[i]);
        physical_page_unref(g->small_physical_pages[i]);
    }

    virtual_range_unref(g->base);
    virtual_range_unref(g->small_base);
    free_small_ranges(g->root.small_snapshot);
    free_small_ranges(g->small_ranges);
    free_small_ranges(g->small_unusable);
    free(g->root.snapshot);
    free_events(g->root.next, g->root.next_count);
    free(g);
}
