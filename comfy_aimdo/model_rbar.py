import ctypes

from . import control

class ModelRBAR:
    def __init__(self, filepath):
        self.ptr = control.lib.rbar_allocate(filepath.encode())
        if not self.ptr:
            raise RuntimeError(f"RBAR allocation failed for {filepath}")

    def __call__(self):
        return self.ptr

    def __del__(self):
        control.lib.rbar_deallocate(self.ptr)
