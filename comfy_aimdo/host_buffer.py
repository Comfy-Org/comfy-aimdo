import ctypes
from . import control

lib = control.lib

if lib is not None:
    lib.hostbuf_create.argtypes = [ctypes.c_size_t]
    lib.hostbuf_create.restype = ctypes.c_void_p

    lib.hostbuf_destroy.argtypes = [ctypes.c_void_p]
    lib.hostbuf_destroy.restype = None

class HostBuffer:
    def __init__(self, size):
        if lib is None:
            raise RuntimeError("aimdo library not initialized")
        self._ptr = lib.hostbuf_create(size)
        if not self._ptr:
            raise RuntimeError(f"Failed to allocate {size} bytes of pinned host memory")
        self._size = size

    @property
    def ptr(self):
        return self._ptr

    @property
    def size(self):
        return self._size

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            lib.hostbuf_destroy(self._ptr)
            self._ptr = None
