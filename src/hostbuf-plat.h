#pragma once

#include <stdbool.h>
#include <stddef.h>

size_t hostbuf_page_size(void);

void *hostbuf_reserve_address_space(size_t size);

bool hostbuf_commit_address_space(void *ptr, size_t size);
bool hostbuf_decommit_address_space(void *ptr, size_t size);
void hostbuf_release_address_space(void *ptr, size_t size);
