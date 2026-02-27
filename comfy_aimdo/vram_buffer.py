import ctypes
from . import control

lib = control.lib

if lib is not None:
    lib.vrambuf_create.argtypes = [ctypes.c_int, ctypes.c_size_t]
    lib.vrambuf_create.restype = ctypes.c_void_p

    lib.vrambuf_grow.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.vrambuf_grow.restype = ctypes.c_bool

    lib.vrambuf_get.argtypes = [ctypes.c_void_p]
    lib.vrambuf_get.restype = ctypes.c_uint64

    lib.vrambuf_destroy.argtypes = [ctypes.c_void_p]

class VRAMBuffer:
    def __init__(self, device=0, max_size=16 * 1024**3):
        self._ptr = lib.vrambuf_create(device, max_size)
        if not self._ptr:
            raise RuntimeError("VRAM reservation failed")
        self.base_addr = lib.vrambuf_get(self._ptr)
        self._allocated = 0

    def size(self):
        return self._allocated

    def get(self, required_size):
        if required_size > self._allocated:
            if not lib.vrambuf_grow(self._ptr, required_size):
                raise RuntimeError(f"VRAM grow failed: {required_size} bytes")

            self._allocated = (required_size + (16*1024**2) - 1) & ~((16*1024**2) - 1)
        return (self, self.base_addr, required_size)

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            lib.vrambuf_destroy(self._ptr)
