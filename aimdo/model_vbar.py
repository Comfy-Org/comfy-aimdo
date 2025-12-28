import torch
import os
import platform
import ctypes
from pathlib import Path

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
        if not self._ptr: raise MemoryError("VBAR allocation failed")
        self.device, self.max_size, self.offset = device, size, 0
        self.base_addr = lib.vbar_get(self._ptr)

    def prioritize(self):
        lib.vbar_prioritize(self._ptr)

    def alloc(self, shape, dtype=torch.float32):
        self.offset = (self.offset + 511) & ~511
        num_bytes = torch.tensor([], dtype=dtype).element_size()
        for dim in shape: num_bytes *= dim

        if self.offset + num_bytes > self.max_size:
            raise MemoryError("VBAR OOM")

        t = get_tensor_from_raw_ptr(self.base_addr + self.offset, self.device, num_bytes)
        t = t.view(dtype).reshape(shape)

        # Attach metadata for fault/unpin
        t.vbar_ptr = self._ptr
        t.vbar_offset = self.offset
        t.vbar_size = num_bytes
        
        self.offset += num_bytes
        return t

    def fault(self, tensor):
        res = lib.vbar_fault(tensor.vbar_ptr, tensor.vbar_offset, tensor.vbar_size)
        if res != 0: raise RuntimeError(f"Fault failed: {res}")

    def unpin(self, tensor):
        lib.vbar_unpin(tensor.vbar_ptr, tensor.vbar_offset, tensor.vbar_size)

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            lib.vbar_free(self._ptr)
            self._ptr = None