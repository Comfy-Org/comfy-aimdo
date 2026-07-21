#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef uint64_t XferFileHandle;

bool xfer_file_init(void);
void xfer_file_cleanup(void);
XferFileHandle xfer_file_open_direct(XferFileHandle file_handle);
void xfer_file_close_direct(XferFileHandle file_handle);
bool xfer_file_direct_matches(XferFileHandle direct_handle,
                              XferFileHandle file_handle);
bool xfer_file_read(XferFileHandle file_handle, uint64_t offset, void *destination,
                    size_t size, bool mark_cold);
bool xfer_file_read_direct(XferFileHandle file_handle, uint64_t offset, void *destination,
                           size_t size);
bool xfer_file_read_at(XferFileHandle file_handle, uint64_t offset, void *destination,
                       size_t size, bool mark_cold);
bool xfer_file_read_at_direct(XferFileHandle file_handle, uint64_t offset,
                              void *destination, size_t size);
