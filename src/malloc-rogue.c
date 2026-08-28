#include "plat.h"
#include "malloc-rogue.h"

typedef struct Rogue {
    VirtualRange *range;
    CUdeviceptr ptr;
    PhysicalPage **pages;
    size_t page_count;
    bool freeing;

    struct Rogue *next;
} Rogue;

typedef struct RogueCandidate {
    CUdeviceptr ptr;
    bool freed;

    struct RogueCandidate *next;
} RogueCandidate;

bool register_rogue_candidate(CUdeviceptr ptr) {
    RogueCandidate *candidate = malloc(sizeof(*candidate));
    if (!candidate) {
        return false;
    }
    *candidate = (RogueCandidate){.ptr = ptr};

    allocations_lock();
    candidate->next = global_rogue_candidates;
    global_rogue_candidates = candidate;
    allocations_unlock();
    return true;
}

bool rogue_candidate_freed(CUdeviceptr ptr) {
    allocations_lock();
    RogueCandidate *candidate = global_rogue_candidates;
    while (candidate && candidate->ptr != ptr) {
        candidate = candidate->next;
    }
    bool freed = candidate && candidate->freed;
    allocations_unlock();
    return freed;
}

RogueHandoff handoff_rogue(VirtualRange *range, CUdeviceptr ptr,
                           PhysicalPage **pages, size_t page_count) {
    Rogue *rogue = malloc(sizeof(*rogue));
    PhysicalPage **references = calloc(page_count, sizeof(*references));

    if (!rogue || !references) {
        free(rogue);
        free(references);
        return ROGUE_HANDOFF_ERROR;
    }

    size_t page_bytes = pages[0]->allocation->bytes;
    CUdeviceptr address = ptr - (ptr - virtual_range_get(range)) % page_bytes;
    for (size_t i = 0; i < page_count; i++) {
        references[i] = physical_page_ref(pages[i], 0);
        if (!references[i]) {
            for (size_t j = 0; j < i; j++) {
                physical_page_unref(references[j]);
            }
            free(references);
            free(rogue);
            return ROGUE_HANDOFF_ERROR;
        }
    }
    for (size_t i = 0; i < page_count; i++) {
        references[i]->address = address + i * page_bytes;
    }

    *rogue = (Rogue){
        .range = virtual_range_ref(range), .ptr = ptr,
        .pages = references, .page_count = page_count};

    allocations_lock();
    RogueCandidate **entry = (RogueCandidate **)&global_rogue_candidates;
    while (*entry && (*entry)->ptr != ptr) {
        entry = &(*entry)->next;
    }
    RogueCandidate *candidate = *entry;
    if (candidate) {
        *entry = candidate->next;
    }
    RogueHandoff result = !candidate ? ROGUE_HANDOFF_ERROR :
        candidate->freed ? ROGUE_HANDOFF_FREED : ROGUE_HANDOFF_OWNED;
    if (result == ROGUE_HANDOFF_OWNED) {
        rogue->next = rogues;
        rogues = rogue;
    }
    allocations_unlock();

    free(candidate);
    if (result == ROGUE_HANDOFF_OWNED) {
        return result;
    }
    for (size_t i = 0; i < page_count; i++) {
        physical_page_unref(references[i]);
    }
    virtual_range_unref(rogue->range);
    free(references);
    free(rogue);
    return result;
}

bool rogue_exists(CUdeviceptr ptr) {
    allocations_lock();
    Rogue *rogue = rogues;
    while (rogue && rogue->ptr != ptr) {
        rogue = rogue->next;
    }
    allocations_unlock();
    return rogue != NULL;
}

bool free_rogue(CUdeviceptr ptr, int *result) {
    allocations_lock();
    Rogue **entry = (Rogue **)&rogues;
    while (*entry && (*entry)->ptr != ptr) {
        entry = &(*entry)->next;
    }
    if (!*entry) {
        RogueCandidate *candidate = global_rogue_candidates;
        while (candidate && candidate->ptr != ptr) {
            candidate = candidate->next;
        }
        if (!candidate) {
            allocations_unlock();
            return false;
        }
        candidate->freed = true;
        *result = CUDA_SUCCESS;
        allocations_unlock();
        return true;
    }

    Rogue *rogue = *entry;
    if (rogue->freeing) {
        *result = CUDA_SUCCESS;
        allocations_unlock();
        return true;
    }
    rogue->freeing = true;
    allocations_unlock();

    *result = cuCtxSynchronize();
    for (size_t i = 0; !*result && i < rogue->page_count; i++) {
        if (rogue->pages[i]) {
            *result = physical_page_unref(rogue->pages[i]);
            if (!*result) {
                rogue->pages[i] = NULL;
            }
        }
    }
    if (!*result) {
        *result = virtual_range_unref(rogue->range);
    }

    allocations_lock();
    if (!*result) {
        entry = (Rogue **)&rogues;
        while (*entry != rogue) {
            entry = &(*entry)->next;
        }
        *entry = rogue->next;
        free(rogue->pages);
        free(rogue);
    } else {
        rogue->freeing = false;
    }
    allocations_unlock();
    return true;
}
