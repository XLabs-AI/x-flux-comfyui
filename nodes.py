import os
import comfy.model_management as mm
import comfy.model_patcher as mp
from comfy.utils import ProgressBar
import copy

import folder_paths
import torch
#from .xflux.src.flux.modules.layers import DoubleStreamBlockLoraProcessor, DoubleStreamBlockProcessor
#from .xflux.src.flux.model import Flux as ModFlux

from .xflux.src.flux.util import (configs, load_ae, load_clip,
                            load_flow_model, load_t5, load_safetensors, load_flow_model_quintized, load_from_repo_id,
                            load_controlnet, Annotator)


from .utils import FluxUpdateModules, attn_processors, set_attn_processor, is_model_pathched, merge_loras, tensor_to_pil, LATENT_PROCESSOR_COMFY
from .layers import DoubleStreamBlockLoraProcessor, DoubleStreamBlockProcessor, DoubleStreamBlockLorasMixerProcessor
from .model_init import Flux as ModFlux
from .model_init import double_blocks_init, single_blocks_init


from comfy.utils import get_attr, set_attr


dir_xlabs = os.path.join(folder_paths.models_dir, "xlabs")
os.makedirs(dir_xlabs, exist_ok=True)
dir_xlabs_loras = os.path.join(dir_xlabs, "loras")
os.makedirs(dir_xlabs_loras, exist_ok=True)
dir_xlabs_controlnets = os.path.join(dir_xlabs, "controlnets")
os.makedirs(dir_xlabs_controlnets, exist_ok=True)
dir_xlabs_flux = os.path.join(dir_xlabs, "flux")
os.makedirs(dir_xlabs_flux, exist_ok=True)



folder_paths.folder_names_and_paths["xlabs"] = ([dir_xlabs], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["xlabs_loras"] = ([dir_xlabs_loras], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["xlabs_controlnets"] = ([dir_xlabs_controlnets], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["xlabs_flux"] = ([dir_xlabs_flux], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["xlabs_flux_json"] = ([dir_xlabs_flux], set({'.json',}))



from .sampling import get_noise, prepare, get_schedule, denoise, denoise_controlnet, unpack
import numpy as np

def load_flux_lora(path):
    if path is not None:
        if '.safetensors' in path:
            checkpoint = load_safetensors(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')
    else:
        checkpoint = None
        print("Invalid path")
    a1 = sorted(list(checkpoint[list(checkpoint.keys())[0]].shape))[0]
    a2 = sorted(list(checkpoint[list(checkpoint.keys())[1]].shape))[0]
    if a1==a2:
        return checkpoint, int(a1)
    return checkpoint, 16

def cleanprint(a):
    print(a)
    return a

def print_if_not_empty(a):
    b = list(a.items())
    if len(b)<1:
        return "{}"
    return b[0]
class LoadFluxLora:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "lora_name": (cleanprint(folder_paths.get_filename_list("xlabs_loras")), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "loadmodel"
    CATEGORY = "XLabsNodes"

    def loadmodel(self, model, lora_name, strength_model):
        debug=True
        
        pbar = ProgressBar(5)
        device=mm.get_torch_device()
        offload_device=mm.unet_offload_device()
        
        is_patched = is_model_pathched(model.model)
        
        print(f"Is model already patched? {is_patched}")
        
        bi = model.clone()
        tyanochky = bi.model
        
        if debug:
            print("\n", (print_if_not_empty(bi.object_patches_backup)), "\n___\n", (print_if_not_empty(bi.object_patches)), "\n")
            try:
                print(get_attr(tyanochky, "diffusion_model.double_blocks.0.processor.lora_weight"))
            except:
                pass
        
        pbar.update(1)
        bi.model.to(device)
        checkpoint, lora_rank = load_flux_lora(os.path.join(dir_xlabs_loras, lora_name))
        pbar.update(2)
        if not is_patched:
            patches=FluxUpdateModules(tyanochky)
            set_attn_processor(model.model.diffusion_model, DoubleStreamBlockProcessor())
        else:
            print("Model already updated")
        pbar.update(3)
        #TYANOCHKYBY=16
        
        lora_attn_procs = {}   
        if checkpoint is not None:
            cached_proccesors =  attn_processors(tyanochky.diffusion_model).items()
            for name, _ in attn_processors(tyanochky.diffusion_model).items():
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=lora_rank, lora_weight=strength_model)
                lora_state_dict = {}
                for k in checkpoint.keys():
                    if name in k:
                        lora_state_dict[k[len(name) + 1:]] = checkpoint[k]
                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(device)
                tmp=DoubleStreamBlockLorasMixerProcessor()
                tmp.add_lora(lora_attn_procs[name])
                lora_attn_procs[name]=tmp
                pbar.update(4)
        #set_attn_processor(tyanochky.diffusion_model, lora_attn_procs)
        if debug:
            try:
                if isinstance(
                        get_attr(tyanochky, "diffusion_model.double_blocks.0.processor"), 
                        DoubleStreamBlockLorasMixerProcessor
                    ):
                    pedovki = get_attr(tyanochky, "diffusion_model.double_blocks.0.processor").lora_weight
                    if len(pedovki)>0:
                        altushki="".join([f"{pedov:.2f}, " for pedov in pedovki])
                        print(f"Loras applied: {altushki}")
            except:
                pass
        
        for name, _ in attn_processors(tyanochky.diffusion_model).items():
            attribute = f"diffusion_model.{name}"
            #old = copy.copy(get_attr(bi.model, attribute))
            if attribute in model.object_patches.keys():
                old = copy.copy((model.object_patches[attribute]))
            else:
                old = None
            lora = merge_loras(old, lora_attn_procs[name])
            bi.add_object_patch(attribute, lora)
            
        
        if debug:
            print("\n", (print_if_not_empty(bi.object_patches_backup)), "\n_b_\n", (print_if_not_empty(bi.object_patches)), "\n")
            print("\n", (print_if_not_empty(model.object_patches_backup)), "\n_m__\n", (print_if_not_empty(model.object_patches)), "\n")
            
            for a, b in bi.object_patches.items():
                print(b.lora_weight)
                break
            
        #print(get_attr(tyanochky, "diffusion_model.double_blocks.0.processor"))
        pbar.update(5)
        return (bi,)

def load_checkpoint_controlnet(local_path):
    if local_path is not None:
        if '.safetensors' in local_path:
            checkpoint = load_safetensors(local_path)
        else:
            checkpoint = torch.load(local_path, map_location='cpu')
    else:
        checkpoint=None
        print("Invalid path")
    return checkpoint

class LoadFluxControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (["flux-dev", "flux-dev-fp8", "flux-schnell"],),
                              "controlnet_path": (folder_paths.get_filename_list("xlabs_controlnets"), ),
                              }}

    RETURN_TYPES = ("FluxControlNet",)
    RETURN_NAMES = ("ControlNet",)
    FUNCTION = "loadmodel"
    CATEGORY = "XLabsNodes"

    def loadmodel(self, model_name, controlnet_path):
        device=mm.get_torch_device()

        controlnet = load_controlnet(model_name, device)
        checkpoint = load_checkpoint_controlnet(os.path.join(dir_xlabs_controlnets, controlnet_path))
        if checkpoint is not None:
            controlnet.load_state_dict(checkpoint)
            control_type = "canny"
            annotator = Annotator()
        ret_controlnet = {
            "model": controlnet,
            "control_type": control_type,
            "annotator": annotator,
        }
        return (ret_controlnet,)
    
class ApplyFluxControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"controlnet": ("FluxControlNet",),
                             "conditioning": ("CONDITIONING",),
                             "IMAGE": ("image",),
                              }}

    RETURN_TYPES = ("ControlNetCondition",)
    RETURN_NAMES = ("controlnet_condition",)
    FUNCTION = "prepare"
    CATEGORY = "XLabsNodes"

    def prepare(self, controlnet, conditioning, image):
        device=mm.get_torch_device()
        print(image.size())
        b, h, w, c = image.size()
        img = tensor_to_pil(image.permute(0, 1, 2, 3))
        controlnet_image = controlnet["annotator"](img, w, h, controlnet["control_type"])
        controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
        controlnet_image = controlnet_image.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(device)
        
        ret_cont = {
            "img": controlnet_image,
            "model": controlnet["model"],
        }
        return (ret_cont,)

class XlabsSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "model": ("MODEL",),
                    "conditioning": ("CONDITIONING",),
                    "neg_conditioning": ("CONDITIONING",),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT",  {"default": 20, "min": 1, "max": 100}),
                    "timestep_to_start_cfg": ("INT",  {"default": 20, "min": 0, "max": 100}),
                    "true_gs": ("INT",  {"default": 3, "min": 0, "max": 100}),
                },
            "optional": {
                    "latent_image": ("LATENT", {"default": None}),
                    "controlnet_condition": ("ControlNetCondition", {"default": None}),
                    
                     }
                }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sampling"
    CATEGORY = "XLabsNodes"

    def sampling(self, model, conditioning, neg_conditioning, noise_seed, steps, timestep_to_start_cfg, true_gs, latent_image=None, controlnet_condition=None):
        
        mm.load_model_gpu(model)
        inmodel = model.model
        #print(conditioning[0][0].shape) #//t5
        #print(conditioning[0][1]['pooled_output'].shape) #//clip
        #print(latent_image['samples'].shape) #// torch.Size([1, 4, 64, 64]) // bc, 4, w//8, h//8
        guidance=conditioning[0][1]['guidance']
        
        device=mm.get_torch_device()
        dtype_model = model.model.diffusion_model.img_in.weight.dtype
        offload_device=mm.unet_offload_device()
        
        torch.manual_seed(noise_seed)
        
        bc, c, w, h = latent_image['samples'].shape
        height=h*8
        width=w*8
        
        if c==16:
            x=latent_image['samples']
            x.to(dtype=dtype_model)
        else:
            x = get_noise(
                1, height, width, device=device,
                dtype=dtype_model, seed=noise_seed
            )
        
        timesteps = get_schedule(
            steps,
            (width // 8) * (height // 8) // (16 * 16)*2,
            shift=False,
        )
        try:
            inmodel.to(device)
        except:
            pass
        x.to(device)
        
        inmodel.diffusion_model.to(device)
        inp_cond = prepare(conditioning[0][0], conditioning[0][1]['pooled_output'], img=x)
        neg_inp_cond = prepare(neg_conditioning[0][0], neg_conditioning[0][1]['pooled_output'], img=x)
        if controlnet_condition is None:
            x = denoise(inmodel.diffusion_model, **inp_cond, timesteps=timesteps, guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs
            )
        
        else:
            controlnet = controlnet_condition['model']
            controlnet_image = controlnet_condition['img']
            controlnet.to(device, dtype=dtype_model)
            controlnet_image.to(device, dtype=dtype_model)
            x = denoise_controlnet(
                inmodel.diffusion_model, **inp_cond, controlnet=controlnet,
                timesteps=timesteps, guidance=guidance,
                controlnet_cond=controlnet_image,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs
            )
            controlnet.to(offload_device)
        
        x=unpack(x,height,width)
        lat_processor = LATENT_PROCESSOR_COMFY()
        x=lat_processor(x)
        lat_ret = {"samples": x}
        #model.model.to(offload_device)
        return (lat_ret,)

    
import json
from optimum.quanto import requantize
from safetensors.torch import load_file as load_sft
class LoadFluxModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                            #"model_name": (["flux-dev", "flux-dev-fp8", "flux-schnell"],),
                            "model_path": (
                                folder_paths.get_filename_list("xlabs_flux") 
                                #+folder_paths.get_filename_list("checkpoints")
                                #+folder_paths.get_filename_list("unet")
                                            ,),
                            "config_path": (
                                folder_paths.get_filename_list("xlabs_flux_json") 
                                #+folder_paths.get_filename_list("checkpoints")
                                #+folder_paths.get_filename_list("unet")
                                            ,),
                              }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "XLabsNodes"

    def loadmodel(self, model_path, config_path):
        device=mm.get_torch_device()
        offload_device=mm.unet_offload_device()
        
        file_path_x = os.path.join(dir_xlabs_flux, config_path)
        сkpt_path_x = os.path.join(dir_xlabs_flux, model_path)
        #file_path_u = os.path.join(dir_xlabs_flux, config_path)
        #file_path_c = os.path.join(dir_xlabs_flux, config_path)
        pbar = ProgressBar(10)
        file_path = None
        ckpt_path = None
        dtype = None
        if os.path.exists(file_path_x):
            file_path = file_path_x
            ckpt_path = сkpt_path_x
        else:
            assert("File doesn't exist")
            return (None,)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            dtype = data['img_in']['weights']
            if dtype == "fp16":
                assert("We currently don't support fp16 type")
                pass
            elif dtype == "qfloat8_e4m3fn":
                pass
            else:
                assert("We currently don't support this type of model")
        except:
            assert("Something went wrong, try to reproduce again and contact with devs")
            return (None,)
        pbar.update(1)
        name = model_path.split(".")[0]
        print(name)
        model = ModFlux(configs[name].params)
        model.to(torch.bfloat16)
        pbar.update(2)
        print("1/3 loaded")
        double_blocks_init(model, configs[name].params, torch.bfloat16)
        pbar.update(3)
        print("2/3 loaded")
        single_blocks_init(model, configs[name].params, torch.bfloat16)
        pbar.update(4)
        print("3/3 loaded")
        model.to(torch.bfloat16)
        pbar.update(5)
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device='cpu')
        pbar.update(6)
        with open(file_path, "r") as f:
            quantization_map = json.load(f)
        if dtype=="qfloat8_e4m3fn":
            print("Start a quantization process...")
            requantize(model, sd, quantization_map, device=torch.device('cpu'))
            print("Model is quantized!")
            pbar.update(7)
        ret_controlnet = mp.Model_Patcher(model, load_device=torch.device('cpu'), offload_device=offload_device)
        print(model)
        print(ret_controlnet)
        pbar.update(10)
        return (ret_controlnet,)
     


NODE_CLASS_MAPPINGS = {
    "FluxLoraLoader": LoadFluxLora,
    "LoadFluxControlNet": LoadFluxControlNet,
    "ApplyFluxControlNet": ApplyFluxControlNet,
    "XlabsSampler": XlabsSampler,
    "LoadFluxModel": LoadFluxModel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoraLoader": "Load Flux LoRA",
    "LoadFluxControlNet": "Load Flux ControlNet",
    "ApplyFluxControlNet": "Apply Flux ControlNet",
    "XlabsSampler": "Xlabs Sampler",
    "LoadFluxModel": "Load Flux Model",
}