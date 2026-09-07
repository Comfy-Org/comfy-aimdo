import contextlib
import ctypes

from . import control


class MallocGraph:
    def __init__(self, handle, stream):
        self._handle = handle
        self._stream = stream
        self._scopes = []

    def _call(self, function, *args):
        result = function(self._handle, *args)
        if not result:
            raise RuntimeError("aimdo memory compile error")
        return result

    def push(self, name=None):
        self._call(control.lib.malloc_graph_push, name.encode() if name is not None else None)
        if name is not None:
            self._scopes.append(None)

    def pop(self):
        broken = self._call(control.lib.malloc_graph_pop) == 2
        if self._scopes:
            self._scopes.pop()
        return broken

    def abort(self):
        self._call(control.lib.malloc_graph_abort)
        self._scopes.clear()

    def pause(self, sync=False):
        self._call(control.lib.malloc_graph_pause, True, sync)

    def resume(self, sync=False):
        self._call(control.lib.malloc_graph_pause, False, sync)

    @contextlib.contextmanager
    def use_stream(self, stream):
        previous = self._stream
        self._call(control.lib.malloc_graph_set_stream, ctypes.c_void_p(stream.cuda_stream))
        self._stream = stream
        try:
            yield
        finally:
            self._call(control.lib.malloc_graph_set_stream, ctypes.c_void_p(previous.cuda_stream))
            self._stream = previous

    def iterate(self, name=None):
        broken = False
        if name is None:
            if self._scopes:
                broken = self.pop()
            return broken
        if self._scopes and self._scopes[-1] == name:
            broken = self.pop()
        self.push(name)
        self._scopes[-1] = name
        return broken

    def _stat(self, which):
        return control.lib.malloc_graph_stat(self._handle, which)

    peak_used = property(lambda self: self._stat(0))
    virtual_bytes = property(lambda self: self._stat(1))
    physical_bytes = property(lambda self: self._stat(2))
    rogue_count = property(lambda self: self._stat(3))

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle and control.lib is not None:
            control.lib.malloc_graph_destroy(handle)
            self._handle = None


def record(stream, assert_graph_breaks=False):
    handle = control.lib.malloc_graph_create(
        control.get_devctx(stream.device.index), ctypes.c_void_p(stream.cuda_stream),
        assert_graph_breaks,
    )
    if not handle:
        raise RuntimeError("aimdo memory compile error: could not start recording")
    return MallocGraph(handle, stream)
