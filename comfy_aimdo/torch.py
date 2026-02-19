from torch._C import _cuda_releasePool, _cuda_beginAllocateCurrentThreadToPool, _cuda_endAllocateToPool
import torch
import ctypes

from . import control

lib = control.lib

def get_tensor_from_raw_ptr(ptr, size, device):
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

def aimdo_to_tensor(alloc, device):
    _, ptr, size = alloc
    return get_tensor_from_raw_ptr(ptr, size, device)

ALLOCATOR = None

#pytorch doesnt have an API for a CUDAPluggableAllocator from an already loaded
#library. Rather than force a second load that pytorch owns, construct these
#pytorch internals outselves as sperate CDLL loads is far too risky.

class CUDAPluggableAllocator(torch.cuda.memory.CUDAPluggableAllocator):
    def __init__(self):
        alloc_fn = ctypes.cast(getattr(control.lib, "alloc_fn"), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(control.lib, "free_fn"), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = torch._C._cuda_customAllocator(alloc_fn, free_fn)

MEMPOOL_PURGATORY = []

def aimdo_torch_setup_thread(device):
    global ALLOCATOR
    if ALLOCATOR is None:
        ALLOCATOR = CUDAPluggableAllocator()
    mempool = torch.cuda.MemPool(ALLOCATOR.allocator())
    #We manually reduce the ref count of the pool to 0 below to get
    #cache emptying working. If we try and destruct later though,
    #~MemPool asserts against this, while also asserting against
    #any attempts to increase the ref-count from 0, so we simply have to
    #leak the actual MemPool. Might need to sigaction() on process exit to
    #just discard the destructor sigabrt for cleanliness.
    MEMPOOL_PURGATORY.append(mempool)
    midx = mempool.id
    _cuda_beginAllocateCurrentThreadToPool(0, midx)
    dummy = torch.empty(1, device=device) #Strong refs the real pool object
    _cuda_releasePool(0, midx) #one for the allocation context
    _cuda_releasePool(0, midx) #one for the pool itself
    return dummy

def aimdo_torch_cleanup_thread():
    pass

def empty_cache_callback():
    #Bail on the aimdo allocator. It can't do anything anymore as there
    #is nothing left to free, and the default mempool has better garbage
    #collector sematics. This baiscally enters --lowvram mode on the fly.
    aimdo_torch_cleanup_thread()
    torch.cuda.empty_cache()

EMPTY_CACHE_CALLBACK_TYPE = ctypes.CFUNCTYPE(None)
empty_cache_callback_ref = EMPTY_CACHE_CALLBACK_TYPE(empty_cache_callback)

# Bindings
if lib is not None:
    lib.set_empty_cache.argtypes = [EMPTY_CACHE_CALLBACK_TYPE]
    lib.vbar_allocate.restype = None

def install_cache_callback():
    lib.set_empty_cache(empty_cache_callback_ref)
