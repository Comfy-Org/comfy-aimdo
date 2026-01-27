#include "plat.h"

uint64_t total_vram_usage;

SHARED_EXPORT
uint64_t get_total_vram_usage() {
    return total_vram_usage;
}
