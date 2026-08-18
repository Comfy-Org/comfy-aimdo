import ctypes

from . import control


class MallocGraph:
    def __init__(self, handle):
        self._handle = handle

    def _call(self, function, *args):
        if not function(self._handle, *args):
            raise RuntimeError("aimdo memory compile error")

    def push(self, name):
        self._call(control.lib.malloc_graph_push, name.encode())

    def pop(self):
        self._call(control.lib.malloc_graph_pop)

    def replay(self):
        self._call(control.lib.malloc_graph_replay)

    def pause(self):
        self._call(control.lib.malloc_graph_pause, True)

    def resume(self):
        self._call(control.lib.malloc_graph_pause, False)

    def iterate(self, name=None):
        result = control.lib.malloc_graph_iterate(
            self._handle, name.encode() if name is not None else None
        )
        if not result:
            raise RuntimeError("aimdo memory compile error")
        return result == 2

    def _stat(self, which):
        return control.lib.malloc_graph_stat(self._handle, which)

    peak_used = property(lambda self: self._stat(0))
    virtual_bytes = property(lambda self: self._stat(1))
    physical_bytes = property(lambda self: self._stat(2))

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle and control.lib is not None:
            control.lib.malloc_graph_destroy(handle)
            self._handle = None


def record(stream):
    lib = control.lib
    lib.malloc_graph_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.malloc_graph_create.restype = ctypes.c_void_p
    lib.malloc_graph_push.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.malloc_graph_pause.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.malloc_graph_pause.restype = ctypes.c_bool
    lib.malloc_graph_iterate.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.malloc_graph_iterate.restype = ctypes.c_int
    lib.malloc_graph_pop.argtypes = lib.malloc_graph_replay.argtypes = [ctypes.c_void_p]
    lib.malloc_graph_push.restype = lib.malloc_graph_pop.restype = lib.malloc_graph_replay.restype = ctypes.c_bool
    lib.malloc_graph_stat.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.malloc_graph_stat.restype = ctypes.c_uint64
    lib.malloc_graph_destroy.argtypes = [ctypes.c_void_p]
    handle = control.lib.malloc_graph_create(
        control.get_devctx(stream.device.index), ctypes.c_void_p(stream.cuda_stream)
    )
    if not handle:
        raise RuntimeError("aimdo memory compile error: could not start recording")
    return MallocGraph(handle)
