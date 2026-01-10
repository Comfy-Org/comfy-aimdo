import gc
import torch
import time
import ctypes
import math

import aimdo.control
from aimdo.model_vbar import ModelVBAR, vbar_fault, vbar_unpin, vbar_signature_compare, vbars_analyze

aimdo.control.set_log_debug()

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

#pytorch doesnt have an API for a CUDAPluggableAllocator from an already loaded
#library. Rather than force a second load that pytorch owns, construct these
#pytorch internals ourselves as sperate CDLL loads is far too risky.

class CUDAPluggableAllocator(torch.cuda.memory.CUDAPluggableAllocator):
    def __init__(self, lib, alloc_fn_name: str, free_fn_name: str):
        alloc_fn = ctypes.cast(getattr(lib, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(lib, free_fn_name), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = torch._C._cuda_customAllocator(alloc_fn, free_fn)

allocator = CUDAPluggableAllocator(aimdo.control.lib, "alloc_fn", "free_fn")

signatures = {}

def run_layer(input_tensor, weight, cpu_source, quiet=False):
    vbar, ptr, size = weight
    o = (ptr - vbar.base_addr) // (1024 ** 2)
    signature = vbar_fault(weight)
    if signature is not None:
        weight_tensor = get_tensor_from_raw_ptr(ptr, size, torch.device("cuda:0")).view(dtype=input_tensor.dtype).view(input_tensor.shape)
        if not vbar_signature_compare(signature, signatures.get(weight, None)):
            weight_tensor.copy_(cpu_source)
            if not quiet:
                print(f"[First Load] Populated weight at offset: {o}M")
        elif not quiet:
            print(f"[No Load Needed] Reusing weight at offset: {o}M")
        w = weight_tensor
        signatures[weight] = signature
    else:
        weight = None
        if not quiet:
            print(f"[Offloaded] offset: {o}M")
        w = cpu_source.to("cuda:0", non_blocking=True)
        
    #Layer math here
    output = input_tensor + w

    if weight is not None:
        vbar_unpin(weight)

    return output

def run_model(weights, cpu_weight, sleep=0):
    with torch.cuda.use_mem_pool(torch.cuda.MemPool(allocator.allocator())):
        x = torch.zeros(cpu_weight.shape, device="cuda:0", dtype=torch.float16)
        for i in range(10): # Iteration loop
            print(f"\nIteration {i}")            
            if (i > 2):
                print("...")

            for layer_weight in weights:
                x = run_layer(x, layer_weight, cpu_weight, quiet=(i > 2))
            time.sleep(sleep) #so you can see nvtop
        #sometimes torch can free mempools while the GPU is still working. Must sync
        #before we gc this mempool
        torch.cuda.synchronize()
    # Torch code comments says some stuff about not actually freeing tensors on mempool
    #context release. Explicitly garbage collect now.
    gc.collect()
    torch.cuda.empty_cache()

#FIXME: Get rid of this
dummy = torch.randn(1, device=torch.device("cuda:0"))

dtype = torch.float16

vbar1 = ModelVBAR(14 * 1024**3, device=0)

# ~200MB weights
shape = (10240, 10240) 
num_layers = 8 * 1024 **3 // (10240 * 10240 * dtype.itemsize)

weights1 = [vbar1.alloc(math.prod(shape) * dtype.itemsize) for _ in range(num_layers)]
#just share one weight in this example, as don't complicate this example
#with RAM usage. in the real world this will be separate weights for every layer
cpu_weight1 = torch.ones(shape, dtype=dtype)

print("##################### Run the first model #######################")
print("Some weights will be loaded and stay there for all iterations")
print("Some weights will be offloaded\n")

run_model(weights1, cpu_weight1)
vbars_analyze() #print some stats

#A smaller second model but with chunkier weights
vbar2 = ModelVBAR(3 * 1024 **3, device=0)
shape = (15443, 20480)
num_layers = 2

weights2 = [ vbar2.alloc(math.prod(shape) * dtype.itemsize) for _ in range(num_layers)]
cpu_weight2 = torch.ones(shape, dtype=dtype)

print("##################### Run the second model #######################")
print("Everything will be loaded and will displace some weights of the first model\n")

run_model(weights2, cpu_weight2, sleep=0.5)
vbars_analyze() #print some stats

print("##################### Run the first model again #######################")
print("Some weights will still be loaded from before and be there first iteration")
print("Some weights will get re-loaded on the first interation")
print("The rest will be offloaded again\n")

vbar1.prioritize()
run_model(weights1, cpu_weight1)
vbars_analyze() #print some stats
