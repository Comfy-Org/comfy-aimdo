import ctypes
import os

from . import control

lib = control.lib

if os.name == "nt":
    import msvcrt

if lib is not None:
    lib.hostbuf_allocate.argtypes = [ctypes.c_uint64]
    lib.hostbuf_allocate.restype = ctypes.c_void_p

    lib.hostbuf_free.argtypes = [ctypes.c_void_p]

    lib.hostbuf_get_raw_address.argtypes = [ctypes.c_void_p]
    lib.hostbuf_get_raw_address.restype = ctypes.c_void_p

    lib.hostbuf_extend.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_bool, ctypes.POINTER(ctypes.c_int64)]
    lib.hostbuf_extend.restype = ctypes.c_void_p

    lib.hostbuf_read_file_slice.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,  # handle / fd
        ctypes.c_uint64,  # file_offset
        ctypes.c_uint64,  # size
        ctypes.c_uint64,  # offset
    ]
    lib.hostbuf_read_file_slice.restype = ctypes.c_bool

    lib.hostbuf_register.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    lib.hostbuf_register.restype = ctypes.c_bool

    lib.hostbuf_unregister.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.hostbuf_unregister.restype = ctypes.c_bool

    lib.hostbuf_truncate.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_bool]
    lib.hostbuf_truncate.restype = ctypes.c_bool


def _file_handle(file_obj):
    if isinstance(file_obj, int):
        return file_obj

    fd = file_obj.fileno()
    return msvcrt.get_osfhandle(fd) if os.name == "nt" else fd


class HostBuffer:
    def __init__(self, size, prewarm=0):
        size = int(size)
        self.size = 0
        self.prewarm = int(prewarm)
        self._ptr = lib.hostbuf_allocate(self.prewarm)
        if not self._ptr:
            raise RuntimeError("HostBuffer allocation failed")
        if size:
            self.extend(size)

    def get_raw_address(self):
        ptr = lib.hostbuf_get_raw_address(self._ptr)
        return int(ptr) if ptr else 0

    def extend(self, size, reallocate=False):
        size = int(size)
        size_delta = ctypes.c_int64(0)
        ptr = lib.hostbuf_extend(self._ptr, size, bool(reallocate), ctypes.byref(size_delta))
        self.size += size_delta.value
        if not ptr and size:
            raise RuntimeError("HostBuffer.extend failed")
        return int(ptr) if ptr else 0

    def read_file_slice(self, file_obj, file_offset, size, offset=0):
        if not lib.hostbuf_read_file_slice(self._ptr, _file_handle(file_obj),
                                           int(file_offset), int(size), int(offset)):
            raise RuntimeError("HostBuffer.read_file_slice failed")
        self.size = max(self.size, int(offset) + int(size))

    def register(self, offset, size):
        if not lib.hostbuf_register(self._ptr, int(offset), int(size)):
            raise RuntimeError("HostBuffer.register failed")

    def unregister(self, offset):
        if not lib.hostbuf_unregister(self._ptr, int(offset)):
            raise RuntimeError("HostBuffer.unregister failed")

    def truncate(self, size, do_unregister=True):
        if not lib.hostbuf_truncate(self._ptr, int(size), bool(do_unregister)):
            raise RuntimeError("HostBuffer.truncate failed")
        self.size = int(size)

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr:
            if lib is not None:
                lib.hostbuf_free(ptr)
            self._ptr = None
