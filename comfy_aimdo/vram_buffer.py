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
    lib.vrambuf_destroy.restype = ctypes.c_bool


class VRAMBuffer:
    def __init__(self, max_size, device):
        self._devctx = control.get_devctx(device)
        self.device = device
        self.max_size = max_size
        self._ptr = lib.vrambuf_create(self.device, self.max_size)
        if not self._ptr:
            raise RuntimeError("VRAM reservation failed")

        self.base_addr = lib.vrambuf_get(self._ptr)
        self._allocated = 0
        self._chunk_size = 16 * 1024**2

    def size(self):
        return self._allocated

    def get(self, size, offset=0):
        offset = int(offset)
        size = int(size)
        required_size = size + offset
        if required_size > self._allocated:
            if not lib.vrambuf_grow(self._ptr, required_size):
                raise RuntimeError(f"VRAM grow failed: {required_size} bytes")

            self._allocated = (required_size + self._chunk_size - 1) & ~(self._chunk_size - 1)

        return (self, self.base_addr + offset, size)

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr:
            if not lib.vrambuf_destroy(ptr):
                raise RuntimeError("VRAM destroy failed: device context unavailable")
            self._ptr = None
