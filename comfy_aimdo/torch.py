import torch
import ctypes
import numpy
import json
import os
import struct
import packaging.version

from . import control
from . import model_rbar

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


# Current as of safetensors 0.7.0
_TYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "C64": torch.complex64,
}
if packaging.version.Version(torch.__version__) >= packaging.version.Version("2.3.0"):
    _TYPES.update(
        {
            "U64": torch.uint64,
            "U32": torch.uint32,
            "U16": torch.uint16,
        }
    )

def safetensors_load_rbar(ckpt, autoref=True):
    rbar = model_rbar.ModelRBAR(ckpt)
    ptr = rbar()

    file_size = os.path.getsize(ckpt)

    #FIXME: There is some surgical stuff you can do with ctypes to avoid numpy dep here.
    data_buffer = numpy.ctypeslib.as_array(
        (ctypes.c_uint8 * file_size).from_address(ptr)
    )

    #FIXME: validate the header does not overrun and put a sane clamp on it.
    header_size = struct.unpack("<Q", data_buffer[:8])[0]
    header_json = data_buffer[8 : 8 + header_size].tobytes().decode("utf-8")
    header = json.loads(header_json)

    data_area = torch.from_numpy(data_buffer)[8 + header_size:]
    if autoref:
        storage = data_area.untyped_storage()
        storage.__aimdo_rbar_ref__ = rbar

    sd = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue

        start, end = info["data_offsets"]
        # View into the lazy-loaded memory
        sd[name] = data_area[start:end].view(_TYPES[info["dtype"]]).reshape(info["shape"])

    return sd, header.get("__metadata__", {}), rbar


def get_torch_allocator():
    return None if control.lib is None else CUDAPluggableAllocator()
