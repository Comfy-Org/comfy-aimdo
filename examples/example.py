import torch
import torch.nn.functional as F
from aimdo.model_vbar import ModelVBAR, vbar_fault, vbar_unpin, get_lib_path

def run_layer(input_tensor, weight_tensor, cpu_source):
    o = weight_tensor.model_vbar_offset
    if vbar_fault(weight_tensor):
        # SUCCESS: VBAR Weight is in VRAM        
        if not getattr(weight_tensor, 'is_populated', False):
            weight_tensor.copy_(cpu_source) 
            weight_tensor.is_populated = True
            print(f"[First Load] Populated 40MB weight at offset: {o}")
        else:
            print(f"[Secondary Fault] Reusing 40MB weight at offset: {o}")
        w = weight_tensor
    else:
        # FAIL: VBAR is under pressure (offloaded)
        print(f"[Offloaded] offset: {o}")
        w = cpu_source.to("cuda:0", non_blocking=True)

    #Layer math here
    output = input_tensor + w

    if w is weight_tensor:
        vbar_unpin(weight_tensor)

    return output

#python does some wild stuff with weakrefs and garbage collection here, so
#whatever you do, do not wrap these in a function.
allocator = torch.cuda.memory.CUDAPluggableAllocator(get_lib_path(), "alloc_fn", "free_fn")
pool = torch.cuda.MemPool(allocator.allocator())

def run_model(weights, cpu_weight):
    torch.cuda.empty_cache()
    x = torch.zeros(cpu_weight.shape, device="cuda:0", dtype=torch.float16)
    for i in range(3): # Iteration loop
        print(f"\nIteration {i}")
        
        for layer_weight in weights:
            x = run_layer(x, layer_weight, cpu_weight)

#This installs aimdo to the pytorch main allocator
with torch.cuda.use_mem_pool(pool):
    #FIXME: prime torch properly somewhere else
    dummy = torch.randn(1, device=torch.device("cuda:0"))

    dtype = torch.float16

    # --- Setup ---
    vbar1 = ModelVBAR(14 * 1024**3, device=0)

    # ~400MB weights
    shape = (10240, 20480) 
    num_layers = 12 * 1024 **3 // (20480 * 10240 * dtype.itemsize)

    print(f"allocating {num_layers} 400MB layers")
    weights1 = [vbar1.alloc(shape, dtype=dtype) for _ in range(num_layers)]
    #just share one weight in this example, as don't complicate this example
    #with RAM usage. in the real world this will be separate weights for every layer
    cpu_weight1 = torch.randn(shape, dtype=dtype)

    print("##################### Run the first model #######################")
    print("Some weights will be loaded and stay there for all iterations")
    print("Some weights will be offloaded")

    # --- Start Inference ---
    run_model(weights1, cpu_weight1)

    #A smaller second model but with chunkier weights
    vbar2 = ModelVBAR(3 * 1024 **3, device=0)
    shape = (20480, 20480)
    num_layers = 2

    print(f"allocating {num_layers} 800MB layers")
    weights2 = [ vbar2.alloc(shape, dtype=dtype) for _ in range(num_layers)]
    cpu_weight2 = torch.randn(shape, dtype=dtype)

    print("##################### Run the second model #######################")
    print("Everything will be loaded and will displace weights of the first model")

    run_model(weights2, cpu_weight2)

    print("##################### Run the first model again #######################")
    print("Some weights will still be loaded from before and be there first iteration")
    print("Some weights will get re-loaded on the first interation")
    print("The rest will be offloaded again")

    vbar1.prioritize()
    run_model(weights1, cpu_weight1)
