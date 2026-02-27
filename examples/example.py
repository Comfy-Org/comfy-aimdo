import gc
import time
import ctypes
import math

##VERY IMPORTANT: comfy_aimfo.control.init() must be called before torch is imported or anything that
#imports torch (including comfy_aimdo.torch(

import comfy_aimdo.control
comfy_aimdo.control.init()
comfy_aimdo.control.set_log_info()
#comfy_aimdo.control.set_log_debug() #use this to see much more information
#comfy_aimdo.control.set_log_verbose() #use this to see even more information (there is also vverbose)

import torch
import comfy_aimdo.torch
from comfy_aimdo.model_vbar import ModelVBAR, vbar_fault, vbar_unpin, vbar_signature_compare

comfy_aimdo.control.init_device(torch.device(torch.cuda.current_device()).index)

signatures = {}

M = (1024 ** 2)

def run_layer(input_tensor, weight, cpu_source, weight_offset): #NOTE: offset just for prints
    vbar, ptr, size = weight
    signature = vbar_fault(weight)
    if signature is not None:
        weight_tensor = comfy_aimdo.torch.aimdo_to_tensor(weight, torch.device("cuda:0")).view(dtype=input_tensor.dtype).view(input_tensor.shape)
        if not vbar_signature_compare(signature, signatures.get(weight, None)):
            weight_tensor.copy_(cpu_source)
            if weight_offset is not None:
                print(f"[First Load] Populated weight at offset: {weight_offset / M}M")
        elif weight_offset is not None:
            print(f"[No Load Needed] Reusing weight at offset: {weight_offset / M}M")
        w = weight_tensor
        signatures[weight] = signature
    else:
        weight = None
        if weight_offset is not None:
            print(f"[Offloaded] offset: {weight_offset / M}M")
        w = cpu_source.to("cuda:0", non_blocking=True)
        
    #Layer math here
    output = input_tensor + w

    if weight is not None:
        vbar_unpin(weight)

    return output

def run_model(weights, cpu_weight, sleep=0):
    x = torch.zeros(cpu_weight.shape, device="cuda:0", dtype=torch.float16)
    for i in range(6): # Iteration loop
        print(f"\nIteration {i}")
        weight_offset = 0 #just for print messages
        if (i > 2):
            print("...")
            weight_offset = None

        for layer_weight in weights:
            x = run_layer(x, layer_weight, cpu_weight, weight_offset)
            if weight_offset is not None:
                weight_offset += cpu_weight.numel() * cpu_weight.element_size()
        time.sleep(sleep) #so you can see nvtop
    gc.collect()
    torch.cuda.empty_cache()

#FIXME: Get rid of this
dummy = torch.randn(1, device=torch.device("cuda:0"))

gpu_size = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
dtype = torch.float16

#A big model, with 30 weights filling 1.5X the available VRAM
num_layers = 30
scale_factor = gpu_size * 3 // (2 *num_layers * (1024 * 1024) * dtype.itemsize)
vbar1 = ModelVBAR(gpu_size * 5, device=0) #The vbar can be much bigger than VRAM
shape = (1024, 1024, scale_factor)

weights1 = [vbar1.alloc(math.prod(shape) * dtype.itemsize) for _ in range(num_layers)]
#just share one weight in this example, as don't complicate this example
#with RAM usage. in the real world this will be separate weights for every layer
cpu_weight1 = torch.ones(shape, dtype=dtype)

print("##################### Run the first model #######################")
print("Some weights will be loaded and stay there for all iterations")
print("Some weights will be offloaded\n")

run_model(weights1, cpu_weight1)
comfy_aimdo.control.analyze() #print some stats

#A smaller second model but with chunkier weights
num_layers=3
vbar2 = ModelVBAR(gpu_size * 5, device=0) #The vbar can be much bigger than VRAM
shape = (1024, 1024, 2, scale_factor)

weights2 = [ vbar2.alloc(math.prod(shape) * dtype.itemsize) for _ in range(num_layers)]
cpu_weight2 = torch.ones(shape, dtype=dtype)

print("##################### Run the second model #######################")
print("Everything will be loaded and will displace some weights of the first model\n")

run_model(weights2, cpu_weight2, sleep=0.5)
comfy_aimdo.control.analyze() #print some stats

print("##################### Run the first model again #######################")
print("Some weights will still be loaded from before and be there first iteration")
print("Some weights will get re-loaded on the first interation")
print("The rest will be offloaded again\n")

vbar1.prioritize()
run_model(weights1, cpu_weight1)
comfy_aimdo.control.analyze() #print some stats
