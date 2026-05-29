#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef uint64_t XferFileHandle;

typedef enum {
    XFER_FILE_SOURCE_HANDLE,
    XFER_FILE_SOURCE_MMAP,
} XferFileSourceMode;

typedef struct {
    XferFileSourceMode mode;
    union {
        XferFileHandle file_handle;
        const uint8_t *mmap;
    } as;
    bool prefetch;
} XferFileSource;

bool xfer_file_init(void);
void xfer_file_cleanup(void);
bool xfer_file_read(XferFileSource source, uint64_t offset, void *destination,
                    size_t size, bool mark_cold);
bool xfer_file_read_at(XferFileHandle file_handle, uint64_t offset, void *destination,
                       size_t size, bool mark_cold);
void xfer_file_prefetch(const void *ptr, size_t size);
