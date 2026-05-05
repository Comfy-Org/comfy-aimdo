#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef uint64_t XferFileHandle;

bool xfer_file_init(void);
void xfer_file_cleanup(void);
bool xfer_file_read(XferFileHandle file_handle, uint64_t offset, void *destination, size_t size);
bool xfer_file_read_at(XferFileHandle file_handle, uint64_t offset, void *destination, size_t size);
