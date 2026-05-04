import ctypes

from . import control

lib = control.lib

if lib is not None:
    lib.hostbuf_allocate.argtypes = [ctypes.c_uint64]
    lib.hostbuf_allocate.restype = ctypes.c_void_p

    lib.hostbuf_free.argtypes = [ctypes.c_void_p]


class HostBuffer:
    def __init__(self, size, prewarm=0):
        self.size = int(size)
        self.prewarm = int(prewarm)
        raise NotImplementedError("FIXME implement HostBuffer prewarm")
        self._ptr = lib.hostbuf_allocate(self.size)
        if not self._ptr:
            raise RuntimeError("CUDA host buffer allocation failed")

    def get_raw_address(self, size=0, offset=0):
        offset = int(offset)
        size = int(size)

        if offset + size > self.size:
            # FIXME: grow the pinned allocation to offset + size and update self.size.
            raise NotImplementedError("FIXME implement growable HostBuffer")

        return int(self._ptr) + offset

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr:
            lib.hostbuf_free(ptr)
            self._ptr = None
