import ctypes

from . import control


ERROR = "aimdo memory compile error"


def _stream_handle(stream):
    return stream.cuda_stream


class MallocGraph:
    def __init__(self, stream):
        lib = control.lib
        lib.aimdo_malloc_graph_record.argtypes = [ctypes.c_uint64]
        lib.aimdo_malloc_graph_record.restype = ctypes.c_void_p
        lib.aimdo_malloc_graph_push.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.aimdo_malloc_graph_push.restype = ctypes.c_bool
        lib.aimdo_malloc_graph_pop.argtypes = [ctypes.c_void_p]
        lib.aimdo_malloc_graph_pop.restype = ctypes.c_bool
        lib.aimdo_malloc_graph_replay.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.aimdo_malloc_graph_replay.restype = ctypes.c_bool
        lib.aimdo_malloc_graph_destroy.argtypes = [ctypes.c_void_p]
        lib.aimdo_malloc_graph_stat.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.aimdo_malloc_graph_stat.restype = ctypes.c_size_t
        self._handle = control.lib.aimdo_malloc_graph_record(_stream_handle(stream))
        if not self._handle:
            raise RuntimeError(f"{ERROR}: recording is already active")

    def _check(self, result):
        if not result:
            raise RuntimeError(f"{ERROR}: trace does not match recording")

    def push(self, name):
        self._check(control.lib.aimdo_malloc_graph_push(self._handle, name.encode()))

    def pop(self):
        self._check(control.lib.aimdo_malloc_graph_pop(self._handle))

    def replay(self):
        import torch
        self._check(control.lib.aimdo_malloc_graph_replay(
            self._handle, _stream_handle(torch.cuda.current_stream())))

    @property
    def peak_used(self):
        return control.lib.aimdo_malloc_graph_stat(self._handle, 0)

    @property
    def virtual_bytes(self):
        return control.lib.aimdo_malloc_graph_stat(self._handle, 1)

    @property
    def physical_bytes(self):
        return control.lib.aimdo_malloc_graph_stat(self._handle, 2)

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            control.lib.aimdo_malloc_graph_destroy(handle)
            self._handle = None


def record(stream):
    return MallocGraph(stream)
