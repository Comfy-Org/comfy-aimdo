#include "plat.h"

#include <windows.h>

typedef struct {
    HANDLE hFile;
    HANDLE hMapping;
    void* base_address;
    uint64_t size;
} ModelMMAP;

static wchar_t *utf8_to_wide(const char *str) {
    int wide_len;
    wchar_t *wide;

    if (!str) {
        return NULL;
    }

    wide_len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str, -1, NULL, 0);
    if (wide_len <= 0) {
        log(ERROR, "%s: failed to convert utf-8 path to utf-16. OS Error: %lu\n", __func__, GetLastError());
        return NULL;
    }

    wide = calloc((size_t)wide_len, sizeof(wchar_t));
    if (!wide) {
        log(ERROR, "%s: allocation failed for wide path\n", __func__);
        return NULL;
    }

    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str, -1, wide, wide_len) <= 0) {
        log(ERROR, "%s: failed to convert utf-8 path to utf-16. OS Error: %lu\n", __func__, GetLastError());
        free(wide);
        return NULL;
    }

    return wide;
}

static bool model_mmap_map(ModelMMAP *mmap) {
    mmap->hMapping = CreateFileMapping(mmap->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mmap->hMapping) {
        log(ERROR, "%s: failed to create file mapping. OS Error: %lu\n", __func__, GetLastError());
        return false;
    }

    if (!MapViewOfFile3(mmap->hMapping, GetCurrentProcess(), mmap->base_address, 0, mmap->size,
                        MEM_REPLACE_PLACEHOLDER, PAGE_READONLY, NULL, 0)) {
        log(ERROR, "%s: MapViewOfFile3 failed. OS Error: %lu\n", __func__, GetLastError());
        CloseHandle(mmap->hMapping);
        mmap->hMapping = NULL;
        return false;
    }

    return true;
}

static bool model_mmap_unmap(ModelMMAP *mmap, ULONG flags) {
    if (!UnmapViewOfFile2(GetCurrentProcess(), mmap->base_address, flags)) {
        log(ERROR, "%s: UnmapViewOfFile2 failed at %p. Error: %lu\n", __func__, mmap->base_address, GetLastError());
        return false;
    }

    if (mmap->hMapping) {
        CloseHandle(mmap->hMapping);
        mmap->hMapping = NULL;
    }

    return true;
}

SHARED_EXPORT
void *model_mmap_allocate(char *file_path) {
    ModelMMAP *mmap = calloc(1, sizeof(*mmap));
    wchar_t *file_path_wide = NULL;
    LARGE_INTEGER fs;

    log(DEBUG, "%s: creating ModelMMAP for %s\n", __func__, file_path);

    if (!mmap) {
        log(ERROR, "%s: allocation failed\n", __func__);
        goto fail_alloc;
    }

    file_path_wide = utf8_to_wide(file_path);
    if (!file_path_wide) {
        log(ERROR, "%s: could not convert file path from utf-8\n", __func__);
        goto fail_file;
    }

    mmap->hFile = CreateFileW(file_path_wide, GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              NULL, OPEN_EXISTING,
                              FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    free(file_path_wide);
    file_path_wide = NULL;
    if (mmap->hFile == INVALID_HANDLE_VALUE) {
        log(ERROR, "%s: could not open file (utf-8 path was: %s). OS Error: %lu\n", __func__, file_path, GetLastError());
        goto fail_file;
    }

    GetFileSizeEx(mmap->hFile, &fs);
    mmap->size = fs.QuadPart;

    mmap->base_address = VirtualAlloc2(NULL, NULL, mmap->size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                      PAGE_NOACCESS, NULL, 0);
    if (!mmap->base_address) {
        log(ERROR, "%s: VirtualAlloc2 failed to reserve address space. OS Error: %lu\n", __func__, GetLastError());
        goto fail_base;
    }

    if (!model_mmap_map(mmap)) {
        goto fail_map;
    }

    log(DEBUG, "%s: returning %p (base=%p size=%llu)\n", __func__,
        (void *)mmap, mmap->base_address, (ull)mmap->size);

    return mmap;

fail_map:
    VirtualFree(mmap->base_address, 0, MEM_RELEASE);
fail_base:
fail_file:
    free(file_path_wide);
    if (mmap && mmap->hFile && mmap->hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(mmap->hFile);
    }
fail_alloc:
    free(mmap);
    return NULL;
}

SHARED_EXPORT
void *model_mmap_get(void *model_mmap_ptr) {
    ModelMMAP *mmap = (ModelMMAP *)model_mmap_ptr;

    return mmap ? mmap->base_address : NULL;
}

SHARED_EXPORT
uint64_t model_mmap_get_file_handle(void *model_mmap_ptr) {
    ModelMMAP *mmap = (ModelMMAP *)model_mmap_ptr;

    return mmap ? (uint64_t)(uintptr_t)mmap->hFile : 0;
}

SHARED_EXPORT
bool model_mmap_bounce(void *model_mmap_ptr) {
    ModelMMAP *mmap = (ModelMMAP *)model_mmap_ptr;

    log(DEBUG, "%s: %p: DEBUG in model_mmap bounce\n", __func__, (void *)mmap);

    if (!mmap || !model_mmap_unmap(mmap, MEM_PRESERVE_PLACEHOLDER) || !model_mmap_map(mmap)) {
        log(ERROR, "%s: %p: FAILED in model_mmap bounce\n", __func__, (void *)mmap);
        return false;
    }

    log(DEBUG, "%s: %p: SUCCESS in model_mmap bounce\n", __func__, (void *)mmap);

    return true;
}

SHARED_EXPORT
void model_mmap_deallocate(void *model_mmap_ptr) {
    ModelMMAP *mmap = (ModelMMAP *)model_mmap_ptr;

    if (!mmap) {
        return;
    }

    model_mmap_unmap(mmap, MEM_PRESERVE_PLACEHOLDER);

    if (mmap->base_address && !VirtualFree(mmap->base_address, 0, MEM_RELEASE)) {
        log(ERROR, "%s: VirtualFree failed for %p. Error: %lu\n", __func__, mmap->base_address, GetLastError());
    }

    if (mmap->hFile && mmap->hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(mmap->hFile);
    }

    free(mmap);
}
