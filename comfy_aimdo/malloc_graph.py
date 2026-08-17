import ctypes

from . import control


_ERROR = "aimdo memory compile error"


class MallocGraph:
    def __init__(self, stream):
        self._stream = stream
        self._ptr = control.lib.malloc_graph_record(stream.cuda_stream)
        if not self._ptr:
            raise RuntimeError(f"{_ERROR}: cannot start recording")

    def _call(self, fn, *args):
        if fn(self._ptr, *args):
            raise RuntimeError(_ERROR)

    def push(self, name):
        self._call(control.lib.malloc_graph_push, str(name).encode())

    def pop(self):
        self._call(control.lib.malloc_graph_pop)

    def replay(self):
        import torch
        self._call(control.lib.malloc_graph_replay, torch.cuda.current_stream().cuda_stream)

    @property
    def peak_used(self): return control.lib.malloc_graph_stat(self._ptr, 0)
    @property
    def virtual_bytes(self): return control.lib.malloc_graph_stat(self._ptr, 1)
    @property
    def physical_bytes(self): return control.lib.malloc_graph_stat(self._ptr, 2)

    def __del__(self):
        if getattr(self, "_ptr", None):
            control.lib.malloc_graph_destroy(self._ptr)
            self._ptr = None


def record(stream):
    return MallocGraph(stream)
