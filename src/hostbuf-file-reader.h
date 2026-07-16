#pragma once

bool hostbuf_file_reader_read_cached(int device, uint64_t file_handle,
                                     uint64_t file_offset, uint64_t size,
                                     cudaStream_t stream, uint64_t device_ptr,
                                     void *cache_ptr, bool mark_cold);
bool hostbuf_file_reader_copy_cached(int device, const void *source, uint64_t size,
                                     cudaStream_t stream, uint64_t device_ptr);
