import torch
import ctypes

from . import control

lib = control.lib

class CUDAPluggableAllocator(torch.cuda.memory.CUDAPluggableAllocator):
    def __init__(self, lib, alloc_fn_name: str, free_fn_name: str):
        alloc_fn = ctypes.cast(getattr(lib, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(lib, free_fn_name), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = torch._C._cuda_customAllocator(alloc_fn, free_fn)

allocator = CUDAPluggableAllocator(lib, "alloc_fn", "free_fn")

def disable():
    allocator = None
