import ctypes

from . import control


def _configure():
    control.lib.malloc_graph_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    control.lib.malloc_graph_create.restype = ctypes.c_void_p
    control.lib.malloc_graph_push.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    control.lib.malloc_graph_push.restype = ctypes.c_bool
    control.lib.malloc_graph_pop.argtypes = [ctypes.c_void_p]
    control.lib.malloc_graph_pop.restype = ctypes.c_bool
    control.lib.malloc_graph_replay.argtypes = [ctypes.c_void_p]
    control.lib.malloc_graph_replay.restype = ctypes.c_bool
    control.lib.malloc_graph_error.argtypes = [ctypes.c_void_p]
    control.lib.malloc_graph_error.restype = ctypes.c_char_p
    control.lib.malloc_graph_stat.argtypes = [ctypes.c_void_p, ctypes.c_int]
    control.lib.malloc_graph_stat.restype = ctypes.c_uint64
    control.lib.malloc_graph_destroy.argtypes = [ctypes.c_void_p]


def _error(handle):
    return control.lib.malloc_graph_error(handle).decode()


class MallocGraph:
    def __init__(self, handle):
        self._handle = handle

    def _call(self, function, *args):
        if not function(self._handle, *args):
            raise RuntimeError(_error(self._handle))

    def push(self, name):
        self._call(control.lib.malloc_graph_push, name.encode())

    def pop(self):
        self._call(control.lib.malloc_graph_pop)

    def replay(self):
        self._call(control.lib.malloc_graph_replay)

    @property
    def peak_used(self):
        return control.lib.malloc_graph_stat(self._handle, 0)

    @property
    def virtual_bytes(self):
        return control.lib.malloc_graph_stat(self._handle, 1)

    @property
    def physical_bytes(self):
        return control.lib.malloc_graph_stat(self._handle, 2)

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle and control.lib is not None:
            control.lib.malloc_graph_destroy(handle)
            self._handle = None


def record(stream):
    _configure()
    handle = control.lib.malloc_graph_create(
        control.get_devctx(stream.device.index), ctypes.c_void_p(stream.cuda_stream)
    )
    if not handle:
        raise RuntimeError("aimdo memory compile error: could not start recording")
    return MallocGraph(handle)
