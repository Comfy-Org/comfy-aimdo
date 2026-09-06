#pragma once

#include "vmm-ref.h"

#include <stdbool.h>

typedef enum {
    ROGUE_HANDOFF_ERROR,
    ROGUE_HANDOFF_FREED,
    ROGUE_HANDOFF_OWNED,
} RogueHandoff;

bool register_rogue_candidate(CUdeviceptr ptr);
void unregister_rogue_candidate(CUdeviceptr ptr);
bool rogue_candidate_freed(CUdeviceptr ptr);
RogueHandoff handoff_rogue(VirtualRange *range, CUdeviceptr ptr,
                           PhysicalPage **pages, size_t page_count);
bool rogue_exists(CUdeviceptr ptr);
bool free_rogue(CUdeviceptr ptr, int *result);
