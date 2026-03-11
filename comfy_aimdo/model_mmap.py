import ctypes

from . import control

lib = control.lib

if lib is not None:
    lib.model_mmap_allocate.argtypes = [ctypes.c_char_p]
    lib.model_mmap_allocate.restype = ctypes.c_void_p

    lib.model_mmap_get.argtypes = [ctypes.c_void_p]
    lib.model_mmap_get.restype = ctypes.c_void_p

    lib.model_mmap_bounce.argtypes = [ctypes.c_void_p]
    lib.model_mmap_bounce.restype = ctypes.c_bool

    lib.model_mmap_deallocate.argtypes = [ctypes.c_void_p]


class ModelMMAP:
    def __init__(self, filepath):
        if lib is None:
            raise RuntimeError("comfy-aimdo is not initialized")

        self.state = lib.model_mmap_allocate(filepath.encode())
        if not self.state:
            raise RuntimeError(f"ModelMMAP allocation failed for {filepath}")

    def get(self):
        return lib.model_mmap_get(self.state)

    def bounce(self):
        return bool(lib.model_mmap_bounce(self.state))

    def __del__(self):
        state = getattr(self, "state", None)
        if state:
            lib.model_mmap_deallocate(state)
