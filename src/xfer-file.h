#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef uint64_t XferFileHandle;

bool xfer_file_init(void);
void xfer_file_cleanup(void);
bool xfer_file_read(XferFileHandle file_handle, uint64_t offset, void *destination,
                    size_t size, bool mark_cold);
bool xfer_file_read_at(XferFileHandle file_handle, uint64_t offset, void *destination,
                       size_t size, bool mark_cold);
void *xfer_copy_group_create(void);
bool xfer_copy_group_add(void *group, const void *source, void *destination,
                         size_t size);
bool xfer_copy_group_add_parallel(void *group, const void *source,
                                  void *destination, size_t size);
void xfer_copy_group_sync(void *group);
void xfer_copy_group_wait(void **group);
