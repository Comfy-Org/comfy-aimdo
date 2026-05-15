#pragma once

#include <stdbool.h>
#include <stddef.h>

bool hostbuf_prewarm_join(void);
bool hostbuf_prewarm_start(void *ptr, size_t size);
