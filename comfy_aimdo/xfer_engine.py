import ctypes
from . import control

lib = control.lib

MEMCPY = 1
HTOD   = 2
EVENT  = 3

if lib is not None:
    lib.vbar_slot_create.argtypes = [ctypes.c_void_p]
    lib.vbar_slot_create.restype = ctypes.c_void_p

    lib.vbar_slot_transfer.argtypes = [
        ctypes.c_void_p, 
        ctypes.c_void_p, #src
        ctypes.c_void_p, #dest
        ctypes.c_size_t, #size
        ctypes.c_int     #type
    ]
    lib.vbar_slot_transfer.restype = ctypes.c_bool

    lib.vbar_slot_wait.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.vbar_slot_destroy.argtypes = [ctypes.c_void_p]

class XferEngine:
    def __init__(self, stream_ptr):
        self._ptr = lib.vbar_slot_create(ctypes.c_void_p(stream_ptr))
        if not self._ptr:
            raise RuntimeError("XferEngine creation failed")

    def transfer(self, src, dest, size, xfer_type):
        if not lib.vbar_slot_transfer(
            self._ptr, 
            ctypes.c_void_p(src), 
            ctypes.c_void_p(dest), 
            size, 
            xfer_type
        ):
            raise RuntimeError("XferEngine: enqueue failed")

    def wait(self, external_stream_ptr):
        lib.vbar_slot_wait(self._ptr, ctypes.c_void_p(external_stream_ptr))

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            lib.vbar_slot_destroy(self._ptr)
