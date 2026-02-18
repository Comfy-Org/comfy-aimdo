#include "plat.h"

void (*empty_cache)(void);

SHARED_EXPORT
void set_empty_cache(void (*ec)(void)) {
    empty_cache = ec;
}
