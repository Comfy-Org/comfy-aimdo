import ctypes
import os

from . import control

lib = control.lib

if lib is not None:
    lib.model_mmap_allocate.argtypes = [ctypes.c_char_p]
    lib.model_mmap_allocate.restype = ctypes.c_void_p

    lib.model_mmap_get.argtypes = [ctypes.c_void_p]
    lib.model_mmap_get.restype = ctypes.c_void_p

    lib.model_mmap_get_file_handle.argtypes = [ctypes.c_void_p]
    lib.model_mmap_get_file_handle.restype = ctypes.c_uint64

    lib.model_mmap_bounce.argtypes = [ctypes.c_void_p]
    lib.model_mmap_bounce.restype = ctypes.c_bool

    lib.model_mmap_deallocate.argtypes = [ctypes.c_void_p]

    lib.hostbuf_file_reader_read.argtypes = [
        ctypes.c_int,     # device
        ctypes.c_uint64,  # handle / ModelMMAP state
        ctypes.c_uint64,  # source offset
        ctypes.c_uint64,  # size
        ctypes.c_void_p,  # cuda stream
        ctypes.c_uint64,  # device dest ptr
        ctypes.c_bool,    # mark_cold, unused for mmap
        ctypes.c_int,     # source mode: 0=file handle, 1=model mmap
    ]
    lib.hostbuf_file_reader_read.restype = ctypes.c_bool


class ModelMMAP:
    def __init__(self, filepath):
        if lib is None:
            raise RuntimeError("comfy-aimdo is not initialized")

        normalized_path = os.fspath(filepath)
        if isinstance(normalized_path, bytes):
            filepath_bytes = normalized_path
        elif os.name == "nt":
            filepath_bytes = normalized_path.encode("utf-8")
        else:
            filepath_bytes = os.fsencode(normalized_path)

        self.state = lib.model_mmap_allocate(filepath_bytes)
        if not self.state:
            raise RuntimeError(f"ModelMMAP allocation failed for {filepath}")

    def get(self):
        return lib.model_mmap_get(self.state)

    def get_file_handle(self):
        return int(lib.model_mmap_get_file_handle(self.state))

    def bounce(self):
        return bool(lib.model_mmap_bounce(self.state))

    def read_to_device(self, mmap_offset, size, stream, device_ptr, device):
        if not lib.hostbuf_file_reader_read(int(device), self.state, int(mmap_offset),
                                            int(size), int(stream) or None,
                                            int(device_ptr), False, 1):
            raise RuntimeError("hostbuf_file_reader_read failed")

    def __del__(self):
        state = getattr(self, "state", None)
        if state:
            lib.model_mmap_deallocate(state)
