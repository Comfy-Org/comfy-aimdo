import torch
import ctypes

import logging

from . import control

# ---- DLPack interop for Intel XPU (Level Zero) -------------------------------
# torch's __cuda_array_interface__ only adopts CUDA pointers. For the XPU build
# (aimdo backed by Level Zero VMM) we hand torch the raw device pointer via a
# DLPack capsule with device type kDLOneAPI. Proven zero-copy on Arc B570.

_KDL_ONEAPI = 14   # DLDeviceType for SYCL / Level Zero
_KDL_UINT = 1      # DLDataTypeCode (unsigned int)


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int),
                ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p),
                             ("deleter", _DLDELETER)]

# Keeps the ctypes capsule backing objects alive until torch releases the tensor
# and invokes our deleter. aimdo owns the underlying memory, so the deleter only
# drops these Python-side keepalives (it never frees device memory).
_dlpack_alive = {}


def _device_index(device):
    d = torch.device(device)
    if d.index is not None:
        return d.index
    if d.type == "xpu":
        return torch.xpu.current_device()
    return 0


def _xpu_tensor_from_raw_ptr(ptr, size, device):
    shape = (ctypes.c_int64 * 1)(size)
    mt = _DLManagedTensor()
    mt.dl_tensor = _DLTensor(ptr, _DLDevice(_KDL_ONEAPI, _device_index(device)), 1,
                             _DLDataType(_KDL_UINT, 8, 1), shape, None, 0)
    mt.manager_ctx = None

    key = ctypes.addressof(mt)

    def _del(_p):
        _dlpack_alive.pop(key, None)

    deleter = _DLDELETER(_del)
    mt.deleter = deleter
    _dlpack_alive[key] = (mt, shape, deleter)

    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = PyCapsule_New(key, b"dltensor", None)
    return torch.from_dlpack(capsule)


def get_tensor_from_raw_ptr(ptr, size, device):
    if torch.device(device).type == "xpu":
        return _xpu_tensor_from_raw_ptr(ptr, size, device)

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

def hostbuf_to_tensor(hostbuf):
    byte_view = (ctypes.c_uint8 * hostbuf.size).from_address(hostbuf.get_raw_address())
    return torch.frombuffer(byte_view, dtype=torch.uint8)

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

def get_torch_allocator():
    #As of this writing (pytorch 2.10), pytorch MemPools + CUDAPluggableAllocator
    #considers the Mempool and pool usage context each as a hard reference to the
    #tensors completely preventing reasonable garbage collection. A read of the code
    #suggests that the assumptions of cudaGraphs completely prohibits pool cleanup
    #on VRAM pressure which ultimately makes this un-usable for our high pressure
    #allocator.
    logging.warning(f"WARNING: Aimdo+CUDAPluggableAllocator is experimental and unsupported.")
    return None if control.lib is None else CUDAPluggableAllocator()
