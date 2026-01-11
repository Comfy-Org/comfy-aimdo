import ctypes

from . import control

lib = control.lib

# Bindings

lib.vbar_allocate.argtypes = [ctypes.c_uint64, ctypes.c_int]
lib.vbar_allocate.restype = ctypes.c_void_p

lib.vbar_prioritize.argtypes = [ctypes.c_void_p]

lib.vbar_deprioritize.argtypes = [ctypes.c_void_p]

lib.vbar_get.argtypes = [ctypes.c_void_p]
lib.vbar_get.restype = ctypes.c_uint64

lib.vbar_free.argtypes = [ctypes.c_void_p]

lib.vbar_fault.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32)]
lib.vbar_fault.restype = ctypes.c_int

lib.vbar_unpin.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]

lib.vbar_loaded_size.argtypes = [ctypes.c_void_p]
lib.vbar_loaded_size.restype = ctypes.c_size_t

lib.vbar_free_memory.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
lib.vbar_free_memory.restype = ctypes.c_uint64

class ModelVBAR:
    def __init__(self, size, device):
        self._ptr = lib.vbar_allocate(int(size), device)
        if not self._ptr:
            raise MemoryError("VBAR allocation failed")
        self.device = device
        self.max_size = size
        self.offset = 0
        self.base_addr = lib.vbar_get(self._ptr)

    def prioritize(self):
        lib.vbar_prioritize(self._ptr)

    def deprioritize(self):
        lib.vbar_deprioritize(self._ptr)

    def alloc(self, num_bytes):
        self.offset = (self.offset + 511) & ~511

        if self.offset + num_bytes > self.max_size:
            raise MemoryError("VBAR OOM")

        alloc = self.base_addr + self.offset
        self.offset += num_bytes
        return (self, alloc, num_bytes)

    #define VBAR_PAGE_SIZE (32 << 20)

    #define VBAR_FAULT_SUCCESS      0
    #define VBAR_FAULT_OOM          1
    #define VBAR_FAULT_ERROR        2

    def fault(self, alloc, size):
        offset = alloc - self.base_addr
        # +2, one for misalignment and one for rounding
        signature = (ctypes.c_uint32 * (size // (32 * 1024 ** 2) + 2))()
        res = lib.vbar_fault(self._ptr, offset, size, signature)
        if res == 0:
            return signature
        elif res == 1:
            return None
        else:
            raise RuntimeError(f"Fault failed: {res}")

    def unpin(self, alloc, size):
        offset = alloc - self.base_addr
        lib.vbar_unpin(self._ptr, offset, size)

    def loaded_size(self):
        return lib.vbar_loaded_size(self._ptr)

    def free_memory(self, size_bytes):
        return lib.vbar_free_memory(self._ptr, int(size_bytes))

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            lib.vbar_free(self._ptr)
            self._ptr = None

def vbar_fault(alloc):
    vbar, offset, size = alloc
    return vbar.fault(offset, size)

def vbar_unpin(alloc):
    if alloc is not None:
        vbar, offset, size = alloc
        vbar.unpin(offset, size)

def vbar_signature_compare(a, b):
    if a is None or b is None:
        return False
    if len(a) != len(b):
        raise ValueError(f"Signatures of mismatched length {len(a)} != {len(b)}")
    return memoryview(a) == memoryview(b)

def vbars_analyze():
    lib.vbars_analyze()
