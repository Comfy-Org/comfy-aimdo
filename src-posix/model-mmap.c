#include "plat.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    int fd;
    void *base_address;
    uint64_t size;
} ModelMMAP;

SHARED_EXPORT
void *model_mmap_allocate(char *file_path) {
    ModelMMAP *mm = calloc(1, sizeof(*mm));
    struct stat st;

    log(DEBUG, "%s: creating ModelMMAP for %s\n", __func__, file_path);

    if (!mm) {
        log(ERROR, "%s: allocation failed\n", __func__);
        return NULL;
    }

    mm->fd = open(file_path, O_RDONLY);
    if (mm->fd < 0) {
        log(ERROR, "%s: could not open file: %s (errno=%d)\n", __func__, file_path, errno);
        goto fail;
    }

    if (fstat(mm->fd, &st) != 0) {
        log(ERROR, "%s: fstat failed for %s (errno=%d)\n", __func__, file_path, errno);
        goto fail;
    }

    if (st.st_size <= 0) {
        log(ERROR, "%s: invalid file size for %s: %lld\n", __func__, file_path, (long long)st.st_size);
        goto fail;
    }

    mm->size = (uint64_t)st.st_size;
    mm->base_address = mmap(NULL, mm->size, PROT_READ, MAP_PRIVATE, mm->fd, 0);
    if (mm->base_address == MAP_FAILED) {
        log(ERROR, "%s: mmap failed for %s (errno=%d)\n", __func__, file_path, errno);
        goto fail;
    }

    log(DEBUG, "%s: returning %p (base=%p size=%llu)\n", __func__, (void *)mm, mm->base_address, mm->size);

    return mm;

fail:
    if (mm->fd >= 0) {
        close(mm->fd);
    }

    free(mm);
    return NULL;
}

SHARED_EXPORT
void *model_mmap_get(void *model_mmap_ptr) {
    ModelMMAP *mmap = (ModelMMAP *)model_mmap_ptr;

    return mmap ? mmap->base_address : NULL;
}

SHARED_EXPORT
bool model_mmap_bounce(void *model_mmap_ptr) {
    ModelMMAP *mmap = (ModelMMAP *)model_mmap_ptr;

    log(DEBUG, "%s: %p: SUCCESS in model_mmap bounce (nop)\n", __func__, (void *)mmap);
    return true;
}

SHARED_EXPORT
void model_mmap_deallocate(void *model_mmap_ptr) {
    ModelMMAP *mmap = (ModelMMAP *)model_mmap_ptr;

    if (!mmap) {
        return;
    }

    if (mmap->base_address && munmap(mmap->base_address, mmap->size) != 0) {
        log(ERROR, "%s: munmap failed for %p (errno=%d)\n", __func__, mmap->base_address, errno);
    }

    if (mmap->fd >= 0) {
        close(mmap->fd);
    }

    free(mmap);
}
