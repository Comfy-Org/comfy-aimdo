#ifndef DYNAMIC_BUFFER_H
#define DYNAMIC_BUFFER_H

#include <stddef.h>
#include <stdbool.h>

#define MAX_BUFFER_RESERVATION (8ULL * 1024 * 1024 * 1024)

typedef struct {
    void *base_ptr;
    size_t committed;
} DynamicBuffer;

bool dynbuf_create(DynamicBuffer *buf);
bool dynbuf_grow(DynamicBuffer *buf, size_t required_size);
void dynbuf_destroy(DynamicBuffer *buf);

#endif /* DYNAMIC_BUFFER_H */
