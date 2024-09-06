from comfy.ldm.flux.layers import DoubleStreamBlock as DSBold
import copy
import torch
from .xflux.src.flux.modules.layers import DoubleStreamBlock as DSBnew
from .layers import (DoubleStreamBlockLoraProcessor,
                     DoubleStreamBlockProcessor,
                     DoubleStreamBlockLorasMixerProcessor,
                     DoubleStreamMixerProcessor)

from comfy.utils import get_attr, set_attr

import numpy as np

def CopyDSB(oldDSB):

    if isinstance(oldDSB, DSBold):
        tyan = copy.copy(oldDSB)

        if hasattr(tyan.img_mlp[0], 'out_features'):
            mlp_hidden_dim = tyan.img_mlp[0].out_features
        else:
            mlp_hidden_dim = 12288

        mlp_ratio = mlp_hidden_dim / tyan.hidden_size
        bi = DSBnew(hidden_size=tyan.hidden_size, num_heads=tyan.num_heads, mlp_ratio=mlp_ratio)
        #better use __dict__ but I bit scared
        (
            bi.img_mod, bi.img_norm1, bi.img_attn, bi.img_norm2,
            bi.img_mlp, bi.txt_mod, bi.txt_norm1, bi.txt_attn, bi.txt_norm2, bi.txt_mlp
        ) = (
            tyan.img_mod, tyan.img_norm1, tyan.img_attn, tyan.img_norm2,
            tyan.img_mlp, tyan.txt_mod, tyan.txt_norm1, tyan.txt_attn, tyan.txt_norm2, tyan.txt_mlp
        )
        bi.set_processor(DoubleStreamBlockProcessor())

        return bi
    return oldDSB

def copy_model(orig, new):
    new = copy.copy(new)
    new.model = copy.copy(orig.model)
    new.model.diffusion_model = copy.copy(orig.model.diffusion_model)
    new.model.diffusion_model.double_blocks = copy.deepcopy(orig.model.diffusion_model.double_blocks)
    count = len(new.model.diffusion_model.double_blocks)
    for i in range(count):
        new.model.diffusion_model.double_blocks[i] = copy.copy(orig.model.diffusion_model.double_blocks[i])
        new.model.diffusion_model.double_blocks[i].load_state_dict(orig.model.diffusion_model.double_blocks[0].state_dict())
"""
class PbarWrapper:
    def __init__(self):
        self.count = 1
        self.weights = []
        self.counts = []
        self.w8ts = []
        self.rn = 0
        self.rnf = 0.0
    def add(self, count, weight):
        self.weights.append(weight)
        self.counts.append(count)
        wa = np.array(self.weights)
        wa = wa/np.sum(wa)
        ca = np.array(self.counts)
        ml = np.multiply(ca, wa)
        cas = np.sum(ml)
        self.count=int(cas)
        self.w8ts = wa.tolist()
    def start(self):
        self.rnf = 0.0
        self.rn = 0
    def __call__(self):
        self.rn+=1
        return 1
"""
def FluxUpdateModules(flux_model, pbar=None):
    save_list = {}
    #print((flux_model.diffusion_model.double_blocks))
    #for k,v in flux_model.diffusion_model.double_blocks:
        #if "double" in k:
    count = len(flux_model.diffusion_model.double_blocks)
    patches = {}

    for i in range(count):
        if pbar is not None:
            pbar.update(1)
        patches[f"double_blocks.{i}"]=CopyDSB(flux_model.diffusion_model.double_blocks[i])
        flux_model.diffusion_model.double_blocks[i]=CopyDSB(flux_model.diffusion_model.double_blocks[i])
    return patches

def is_model_pathched(model):
    def test(mod):
        if isinstance(mod, DSBnew):
            return True
        else:
            for p in mod.children():
                if test(p):
                    return True
        return False
    result = test(model)
    return result



def attn_processors(model_flux):
    # set recursively
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, procs):

        if hasattr(module, "set_processor"):
            procs[f"{name}.processor"] = module.processor
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, procs)

        return procs

    for name, module in model_flux.named_children():
        fn_recursive_add_processors(name, module, processors)
    return processors
def merge_loras(lora1, lora2):
    new_block = DoubleStreamMixerProcessor()
    if isinstance(lora1, DoubleStreamMixerProcessor):
        new_block.set_loras(*lora1.get_loras())
        new_block.set_ip_adapters(lora1.get_ip_adapters())
    elif isinstance(lora1, DoubleStreamBlockLoraProcessor):
        new_block.add_lora(lora1)
    else:
        pass
    if isinstance(lora2, DoubleStreamMixerProcessor):
        new_block.set_loras(*lora2.get_loras())
        new_block.set_ip_adapters(lora2.get_ip_adapters())
    elif isinstance(lora2, DoubleStreamBlockLoraProcessor):
        new_block.add_lora(lora2)
    else:
        pass
    return new_block

def set_attn_processor(model_flux, processor):
    r"""
    Sets the attention processor to use to compute attention.

    Parameters:
        processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
            The instantiated processor class or a dictionary of processor classes that will be set as the processor
            for **all** `Attention` layers.

            If `processor` is a dict, the key needs to define the path to the corresponding cross attention
            processor. This is strongly recommended when setting trainable attention processors.

    """
    count = len(attn_processors(model_flux).keys())
    if isinstance(processor, dict) and len(processor) != count:
        raise ValueError(
            f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
            f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
        )

    def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor"):
            if isinstance(module.get_processor(), DoubleStreamBlockLorasMixerProcessor):
                block = copy.copy(module.get_processor())
                module.set_processor(copy.deepcopy(module.get_processor()))
                new_block = DoubleStreamBlockLorasMixerProcessor()
                #q1, q2, p1, p2, w1 = block.get_loras()
                new_block.set_loras(*block.get_loras())
                if not isinstance(processor, dict):
                    new_block.add_lora(processor)
                else:

                    new_block.add_lora(processor.pop(f"{name}.processor"))
                module.set_processor(new_block)
                #block = set_attr(module, "", new_block)
            elif isinstance(module.get_processor(), DoubleStreamBlockLoraProcessor):
                block = DoubleStreamBlockLorasMixerProcessor()
                block.add_lora(copy.copy(module.get_processor()))
                if not isinstance(processor, dict):
                    block.add_lora(processor)
                else:
                    block.add_lora(processor.pop(f"{name}.processor"))
                module.set_processor(block)
            else:
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in model_flux.named_children():
        fn_recursive_attn_processor(name, module, processor)

class LATENT_PROCESSOR_COMFY:
    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors =[
                    [-0.0404,  0.0159,  0.0609],
                    [ 0.0043,  0.0298,  0.0850],
                    [ 0.0328, -0.0749, -0.0503],
                    [-0.0245,  0.0085,  0.0549],
                    [ 0.0966,  0.0894,  0.0530],
                    [ 0.0035,  0.0399,  0.0123],
                    [ 0.0583,  0.1184,  0.1262],
                    [-0.0191, -0.0206, -0.0306],
                    [-0.0324,  0.0055,  0.1001],
                    [ 0.0955,  0.0659, -0.0545],
                    [-0.0504,  0.0231, -0.0013],
                    [ 0.0500, -0.0008, -0.0088],
                    [ 0.0982,  0.0941,  0.0976],
                    [-0.1233, -0.0280, -0.0897],
                    [-0.0005, -0.0530, -0.0020],
                    [-0.1273, -0.0932, -0.0680]
                ]
    def __call__(self, x):
        return (x / self.scale_factor) + self.shift_factor
    def go_back(self, x):
        return (x - self.shift_factor) * self.scale_factor



def check_is_comfy_lora(sd):
    for k in sd:
        if "lora_down" in k or "lora_up" in k:
            return True
    return False

def comfy_to_xlabs_lora(sd):
    sd_out = {}
    for k in sd:
        if "diffusion_model" in k:
            new_k =  (k
                    .replace(".lora_down.weight", ".down.weight")
                    .replace(".lora_up.weight", ".up.weight")
                    .replace(".img_attn.proj.", ".processor.proj_lora1.")
                    .replace(".txt_attn.proj.", ".processor.proj_lora2.")
                    .replace(".img_attn.qkv.", ".processor.qkv_lora1.")
                    .replace(".txt_attn.qkv.", ".processor.qkv_lora2."))
            new_k = new_k[len("diffusion_model."):]
        else:
            new_k=k
        sd_out[new_k] = sd[k]
    return sd_out

def LinearStrengthModel(start, finish, size):
    return [
        (start + (finish - start) * (i / (size - 1))) for i in range(size)
        ]
def FirstHalfStrengthModel(start, finish, size):
    sizehalf = size//2
    arr = [
        (start + (finish - start) * (i / (sizehalf - 1))) for i in range(sizehalf)
        ]
    return arr+[finish]*(size-sizehalf)
def SecondHalfStrengthModel(start, finish, size):
    sizehalf = size//2
    arr = [
        (start + (finish - start) * (i / (sizehalf - 1))) for i in range(sizehalf)
        ]
    return [start]*(size-sizehalf)+arr
def SigmoidStrengthModel(start, finish, size):
    def fade_out(x, x1, x2):
        return 1 / (1 + np.exp(-(x - (x1 + x2) / 2) * 8 / (x2 - x1)))
    arr = [start + (finish - start) * (fade_out(i, 0, size) - 0.5) for i in range(size)]
    return arr

class ControlNetContainer:
    def __init__(
            self, controlnet, controlnet_cond, 
            controlnet_gs, controlnet_start_step,
            controlnet_end_step,
            
            ):
        self.controlnet_cond = controlnet_cond
        self.controlnet_gs = controlnet_gs
        self.controlnet_start_step = controlnet_start_step
        self.controlnet_end_step = controlnet_end_step
        self.controlnet = controlnet