import torch
import os
import platform
import ctypes
from pathlib import Path

#1d int8 tensor that the user can then view(dtype=).view(shape=) as whatever they want

def get_tensor_from_raw_ptr(ptr, device, size):

    container = {
        "shape": (size,),
        "typestr": "|u1",
        "data": (ptr, False), #writable
        "version": 3,
    }
    
    class Holder:
        pass
 
    holder = Holder()
    holder.__cuda_array_interface__ = container
    
    return torch.as_tensor(holder, device=device)


def get_lib_path():
    # Get the directory where this script/package is located
    base_path = Path(__file__).parent.resolve()
    
    # Determine extension based on OS
    system = platform.system()
    if system == "Windows":
        lib_name = "aimdo.dll"
    elif system == "Linux":
        lib_name = "aimdo.so"
    else:
        # MacOS usually uses .dylib, though often .so works
        lib_name = "aimdo.so" 
        
    return str(base_path / lib_name)

def get_primary_context():
    allocator = CUDAPluggableAllocator(get_lib_path(), "alloc_fn", "free_fn")
    pool = torch.cuda.MemPool(allocator)
    return torch.cuda.use_mem_pool(pool)

# Load the library
lib_path = get_lib_path()
if not os.path.exists(lib_path):
    raise ImportError(f"Cannot find native library at {lib_path}")

lib = ctypes.CDLL(lib_path)

# Bindings

lib.vbar_allocate.argtypes = [ctypes.c_uint64, ctypes.c_int]
lib.vbar_allocate.restype = ctypes.c_void_p

lib.vbar_prioritize.argtypes = [ctypes.c_void_p]

lib.vbar_get.argtypes = [ctypes.c_void_p]
lib.vbar_get.restype = ctypes.c_uint64

lib.vbar_free.argtypes = [ctypes.c_void_p]

lib.vbar_fault.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
lib.vbar_fault.restype = ctypes.c_int

lib.vbar_unpin.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]

class ModelVBAR:
    def __init__(self, size, device=0):
        self._ptr = lib.vbar_allocate(size, device)
        if not self._ptr:
            raise MemoryError("VBAR allocation failed")
        self.device = device
        self.max_size = size
        self.offset = 0
        self.base_addr = lib.vbar_get(self._ptr)

    def prioritize(self):
        lib.vbar_prioritize(self._ptr)

    def alloc(self, shape, dtype=torch.float32):
        self.offset = (self.offset + 511) & ~511
        num_bytes = dtype.itemsize
        for dim in shape:
            num_bytes *= dim

        if self.offset + num_bytes > self.max_size:
            raise MemoryError("VBAR OOM")

        t = get_tensor_from_raw_ptr(self.base_addr + self.offset, self.device, num_bytes)
        t = t.view(dtype).reshape(shape)

        # Attach metadata for fault/unpin
        t.model_vbar = self
        t.model_vbar_offset = self.offset
        t.model_vbar_size = num_bytes
        
        self.offset += num_bytes
        return t

    #define VBAR_FAULT_SUCCESS  0
    #define VBAR_FAULT_OOM      1
    #define VBAR_FAULT_ERROR    2

    def fault(self, offset, size):
        res = lib.vbar_fault(self._ptr, offset, size)
        if res == 0:
            return True
        elif res == 1:
            return False
        else:
            raise RuntimeError(f"Fault failed: {res}")

    def unpin(self, offset, size):
        lib.vbar_unpin(self._ptr, offset, size)

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            lib.vbar_free(self._ptr)
            self._ptr = None

def vbar_fault(tensor):
    return tensor.model_vbar.fault(tensor.model_vbar_offset, tensor.model_vbar_size)

def vbar_unpin(tensor):
    tensor.model_vbar.unpin(tensor.model_vbar_offset, tensor.model_vbar_size)
