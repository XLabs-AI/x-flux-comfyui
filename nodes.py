import os

import comfy.model_management as mm
import comfy.model_patcher as mp
from comfy.utils import ProgressBar
from comfy.clip_vision import load as load_clip_vision
from comfy.clip_vision import clip_preprocess, Output
import latent_preview
import copy

import folder_paths

import torch
#from .xflux.src.flux.modules.layers import DoubleStreamBlockLoraProcessor, DoubleStreamBlockProcessor
#from .xflux.src.flux.model import Flux as ModFlux

from .xflux.src.flux.util import (configs, load_ae, load_clip,
                            load_flow_model, load_t5, load_safetensors, load_from_repo_id,
                            load_controlnet)


from .utils import (FirstHalfStrengthModel, FluxUpdateModules, LinearStrengthModel, 
                SecondHalfStrengthModel, SigmoidStrengthModel, attn_processors, 
                set_attn_processor,
                is_model_pathched, merge_loras, LATENT_PROCESSOR_COMFY,
                ControlNetContainer,
                comfy_to_xlabs_lora, check_is_comfy_lora)
from .layers import (DoubleStreamBlockLoraProcessor,
                     DoubleStreamBlockProcessor,
                     DoubleStreamBlockLorasMixerProcessor,
                     DoubleStreamMixerProcessor,
                     IPProcessor,
                     ImageProjModel)
from .xflux.src.flux.model import Flux as ModFlux
#from .model_init import double_blocks_init, single_blocks_init


from comfy.utils import get_attr, set_attr
from .clip import FluxClipViT


dir_xlabs = os.path.join(folder_paths.models_dir, "xlabs")
os.makedirs(dir_xlabs, exist_ok=True)
dir_xlabs_loras = os.path.join(dir_xlabs, "loras")
os.makedirs(dir_xlabs_loras, exist_ok=True)
dir_xlabs_controlnets = os.path.join(dir_xlabs, "controlnets")
os.makedirs(dir_xlabs_controlnets, exist_ok=True)
dir_xlabs_flux = os.path.join(dir_xlabs, "flux")
os.makedirs(dir_xlabs_flux, exist_ok=True)
dir_xlabs_ipadapters = os.path.join(dir_xlabs, "ipadapters")
os.makedirs(dir_xlabs_ipadapters, exist_ok=True)


folder_paths.folder_names_and_paths["xlabs"] = ([dir_xlabs], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["xlabs_loras"] = ([dir_xlabs_loras], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["xlabs_controlnets"] = ([dir_xlabs_controlnets], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["xlabs_ipadapters"] = ([dir_xlabs_ipadapters], folder_paths.supported_pt_extensions)
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
        debug=False


        device=mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        is_patched = is_model_pathched(model.model)

        print(f"Is model already patched? {is_patched}")
        mul = 1
        if is_patched:
            pbar = ProgressBar(5)
        else:
            mul = 3
            count = len(model.model.diffusion_model.double_blocks)
            pbar = ProgressBar(5*mul+count)

        bi = model.clone()
        tyanochky = bi.model

        if debug:
            print("\n", (print_if_not_empty(bi.object_patches_backup)), "\n___\n", (print_if_not_empty(bi.object_patches)), "\n")
            try:
                print(get_attr(tyanochky, "diffusion_model.double_blocks.0.processor.lora_weight"))
            except:
                pass

        pbar.update(mul)
        bi.model.to(device)
        checkpoint, lora_rank = load_flux_lora(os.path.join(dir_xlabs_loras, lora_name))
        pbar.update(mul)
        if not is_patched:
            print("We are patching diffusion model, be patient please")
            patches=FluxUpdateModules(tyanochky, pbar)
            #set_attn_processor(model.model.diffusion_model, DoubleStreamBlockProcessor())
        else:
            print("Model already updated")
        pbar.update(mul)
        #TYANOCHKYBY=16

        lora_attn_procs = {}
        if checkpoint is not None:
            if check_is_comfy_lora(checkpoint):
                checkpoint = comfy_to_xlabs_lora(checkpoint)
            #cached_proccesors =  attn_processors(tyanochky.diffusion_model).items()
            for name, _ in attn_processors(tyanochky.diffusion_model).items():
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                    dim=3072, rank=lora_rank, lora_weight=strength_model)
                lora_state_dict = {}
                for k in checkpoint.keys():
                    if name in k:
                        lora_state_dict[k[len(name) + 1:]] = checkpoint[k]
                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(device)
                tmp=DoubleStreamMixerProcessor()
                tmp.add_lora(lora_attn_procs[name])
                lora_attn_procs[name]=tmp
            pbar.update(mul)
        #set_attn_processor(tyanochky.diffusion_model, lora_attn_procs)
        if debug:
            try:
                if isinstance(
                        get_attr(tyanochky, "diffusion_model.double_blocks.0.processor"),
                        DoubleStreamMixerProcessor
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

            for _, b in bi.object_patches.items():
                print(b.lora_weight)
                break

        #print(get_attr(tyanochky, "diffusion_model.double_blocks.0.processor"))
        pbar.update(mul)
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
        ret_controlnet = {
            "model": controlnet,
            "control_type": control_type,
        }
        return (ret_controlnet,)

class ApplyFluxControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "controlnet": ("FluxControlNet",),
                    "image": ("IMAGE", ),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                },
                "optional": {
                    "controlnet_condition": ("ControlNetCondition", {"default": None}),
                }
        }

    RETURN_TYPES = ("ControlNetCondition",)
    RETURN_NAMES = ("controlnet_condition",)
    FUNCTION = "prepare"
    CATEGORY = "XLabsNodes"

    def prepare(self, controlnet, image, strength, controlnet_condition = None):
        device=mm.get_torch_device()
        controlnet_image = torch.from_numpy((np.array(image) * 2) - 1)
        controlnet_image = controlnet_image.permute(0, 3, 1, 2).to(torch.bfloat16).to(device)

        if controlnet_condition is None:
            ret_cont = [{
                "img": controlnet_image,
                "controlnet_strength": strength,
                "model": controlnet["model"],
                "start": 0.0,
                "end": 1.0
            }]
        else:
            ret_cont = controlnet_condition+[{
                "img": controlnet_image,
                "controlnet_strength": strength,
                "model": controlnet["model"],
                "start": 0.0,
                "end": 1.0
            }]
        return (ret_cont,)

class ApplyAdvancedFluxControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "controlnet": ("FluxControlNet",),
                    "image": ("IMAGE", ),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                },
                "optional": {
                    "controlnet_condition": ("ControlNetCondition", {"default": None}),
                }
            }

    RETURN_TYPES = ("ControlNetCondition",)
    RETURN_NAMES = ("controlnet_condition",)
    FUNCTION = "prepare"
    CATEGORY = "XLabsNodes"

    def prepare(self, controlnet, image, strength, start, end, controlnet_condition = None):

        device=mm.get_torch_device()
        controlnet_image = torch.from_numpy((np.array(image) * 2) - 1)
        controlnet_image = controlnet_image.permute(0, 3, 1, 2).to(torch.bfloat16).to(device)

        ret_cont = {
            "img": controlnet_image,
            "controlnet_strength": strength,
            "model": controlnet["model"],
            "start": start,
            "end": end
        }
        if controlnet_condition is None:
            ret_cont = [ret_cont]
        else:
            ret_cont = controlnet_condition+[ret_cont]
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
                    "true_gs": ("FLOAT",  {"default": 3, "min": 0, "max": 100}),
                    "image_to_image_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def sampling(self, model, conditioning, neg_conditioning,
                 noise_seed, steps, timestep_to_start_cfg, true_gs,
                 image_to_image_strength, denoise_strength,
                 latent_image=None, controlnet_condition=None
                 ):
        additional_steps = 11 if controlnet_condition is None else 12
        mm.load_model_gpu(model)
        inmodel = model.model
        #print(conditioning[0][0].shape) #//t5
        #print(conditioning[0][1]['pooled_output'].shape) #//clip
        #print(latent_image['samples'].shape) #// torch.Size([1, 4, 64, 64]) // bc, 4, w//8, h//8
        try:
            guidance = conditioning[0][1]['guidance']
        except:
            guidance = 1.0

        device=mm.get_torch_device()
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        if torch.cuda.is_bf16_supported():
            dtype_model = torch.bfloat16
        else:
            dtype_model = torch.float16
        #dtype_model = torch.bfloat16#model.model.diffusion_model.img_in.weight.dtype
        offload_device=mm.unet_offload_device()

        torch.manual_seed(noise_seed)

        bc, c, h, w = latent_image['samples'].shape
        height = (h//2) * 16
        width = (w//2) * 16

        x = get_noise(
            bc, height, width, device=device,
            dtype=dtype_model, seed=noise_seed
        )
        orig_x = None
        if c==16:
            orig_x=latent_image['samples']
            lat_processor2 = LATENT_PROCESSOR_COMFY()
            orig_x=lat_processor2.go_back(orig_x)
            orig_x=orig_x.to(device, dtype=dtype_model)

        
        timesteps = get_schedule(
            steps,
            (width // 8) * (height // 8) // 4,
            shift=True,
        )
        try:
            inmodel.to(device)
        except:
            pass
        x.to(device)
        
        inmodel.diffusion_model.to(device)
        inp_cond = prepare(conditioning[0][0], conditioning[0][1]['pooled_output'], img=x)
        neg_inp_cond = prepare(neg_conditioning[0][0], neg_conditioning[0][1]['pooled_output'], img=x)

        if denoise_strength<=0.99:
            try:
                timesteps=timesteps[:int(len(timesteps)*denoise_strength)]
            except:
                pass
        # for sampler preview
        x0_output = {}
        callback = latent_preview.prepare_callback(model, len(timesteps) - 1, x0_output)

        if controlnet_condition is None:
            x = denoise(
                inmodel.diffusion_model, **inp_cond, timesteps=timesteps, guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs,
                image2image_strength=image_to_image_strength,
                orig_image=orig_x,
                callback=callback,
                width=width,
                height=height,
            )

        else:
            def prepare_controlnet_condition(controlnet_condition):
                controlnet = controlnet_condition['model']
                controlnet_image = controlnet_condition['img']
                controlnet_image = torch.nn.functional.interpolate(
                    controlnet_image, size=(height, width), scale_factor=None, mode='bicubic',)
                controlnet_strength = controlnet_condition['controlnet_strength']
                controlnet_start = controlnet_condition['start']
                controlnet_end = controlnet_condition['end']
                controlnet.to(device, dtype=dtype_model)
                controlnet_image=controlnet_image.to(device, dtype=dtype_model)
                return {
                    "img": controlnet_image,
                    "controlnet_strength": controlnet_strength,
                    "model": controlnet,
                    "start": controlnet_start,
                    "end": controlnet_end,
                }


            cnet_conditions = [prepare_controlnet_condition(el) for el in controlnet_condition]
            containers = []
            for el in cnet_conditions:
                start_step = int(el['start']*len(timesteps))
                end_step = int(el['end']*len(timesteps))
                container = ControlNetContainer(el['model'], el['img'], el['controlnet_strength'], start_step, end_step)
                containers.append(container)

            mm.load_models_gpu([model,])
            #mm.load_model_gpu(controlnet)

            total_steps = len(timesteps)

            x = denoise_controlnet(
                inmodel.diffusion_model, **inp_cond, 
                controlnets_container=containers,
                timesteps=timesteps, guidance=guidance,
                #controlnet_cond=controlnet_image,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs,
                #controlnet_gs=controlnet_strength,
                image2image_strength=image_to_image_strength,
                orig_image=orig_x,
                callback=callback,
                width=width,
                height=height,
                #controlnet_start_step=start_step,
                #controlnet_end_step=end_step
            )
            #controlnet.to(offload_device)

        x = unpack(x, height, width)
        lat_processor = LATENT_PROCESSOR_COMFY()
        x = lat_processor(x)
        lat_ret = {"samples": x}

        #model.model.to(offload_device)
        return (lat_ret,)



class LoadFluxIPAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "ipadatper": (folder_paths.get_filename_list("xlabs_ipadapters"),),
                "clip_vision": (folder_paths.get_filename_list("clip_vision"),),
                "provider": (["CPU", "GPU",],),
            }
        }
    RETURN_TYPES = ("IP_ADAPTER_FLUX",)
    RETURN_NAMES = ("ipadapterFlux",)
    FUNCTION = "loadmodel"
    CATEGORY = "XLabsNodes"

    def loadmodel(self, ipadatper, clip_vision, provider):
        pbar = ProgressBar(6)
        device=mm.get_torch_device()
        offload_device=mm.unet_offload_device()
        pbar.update(1)
        ret_ipa = {}
        path = os.path.join(dir_xlabs_ipadapters, ipadatper)
        ckpt = load_safetensors(path)
        pbar.update(1)
        path_clip = folder_paths.get_full_path("clip_vision", clip_vision)
        
        try: 
            clip = FluxClipViT(path_clip)
        except:
            clip = load_clip_vision(path_clip).model
        
        ret_ipa["clip_vision"] = clip
        prefix = "double_blocks."
        blocks = {}
        proj = {}
        for key, value in ckpt.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value
        pbar.update(1)
        img_vec_in_dim=768
        context_in_dim=4096
        num_ip_tokens=16        
        if ckpt['ip_adapter_proj_model.proj.weight'].shape[0]//4096==4:
            num_ip_tokens=4
        else:
            num_ip_tokens=16
        improj = ImageProjModel(context_in_dim, img_vec_in_dim, num_ip_tokens)
        improj.load_state_dict(proj)
        pbar.update(1)
        ret_ipa["ip_adapter_proj_model"] = improj

        ret_ipa["double_blocks"] = torch.nn.ModuleList([IPProcessor(4096, 3072) for i in range(19)])
        ret_ipa["double_blocks"].load_state_dict(blocks)
        pbar.update(1)
        return (ret_ipa,)



class ApplyFluxIPAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "ip_adapter_flux": ("IP_ADAPTER_FLUX",),
                              "image": ("IMAGE",),
                              "ip_scale": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.001}),
                              #"text_scale": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "applymodel"
    CATEGORY = "XLabsNodes"

    def applymodel(self, model, ip_adapter_flux, image, ip_scale):
        debug=False


        device=mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        is_patched = is_model_pathched(model.model)

        print(f"Is model already patched? {is_patched}")
        mul = 1
        if is_patched:
            pbar = ProgressBar(5)
        else:
            mul = 3
            count = len(model.model.diffusion_model.double_blocks)
            pbar = ProgressBar(5*mul+count)

        bi = model.clone()
        tyanochky = bi.model

        clip = ip_adapter_flux['clip_vision']
        
        if isinstance(clip, FluxClipViT):
            #torch.Size([1, 526, 526, 3])
            #image = torch.permute(image, (0, ))
            #print(image.shape)
            #print(image)
            clip_device = next(clip.model.parameters()).device
            image = torch.clip(image*255, 0.0, 255)
            out = clip(image).to(dtype=torch.bfloat16)
            neg_out = clip(torch.zeros_like(image)).to(dtype=torch.bfloat16)
        else:
            print("Using old vit clip")
            clip_device = next(clip.parameters()).device
            pixel_values = clip_preprocess(image.to(clip_device)).float()
            out = clip(pixel_values=pixel_values)
            neg_out = clip(pixel_values=torch.zeros_like(pixel_values))    
            neg_out = neg_out[2].to(dtype=torch.bfloat16)
            out = out[2].to(dtype=torch.bfloat16)
        pbar.update(mul)
        if not is_patched:
            print("We are patching diffusion model, be patient please")
            patches=FluxUpdateModules(tyanochky, pbar)
            print("Patched succesfully!")
        else:
            print("Model already updated")
        pbar.update(mul)

        #TYANOCHKYBY=16
        ip_projes_dev = next(ip_adapter_flux['ip_adapter_proj_model'].parameters()).device
        ip_adapter_flux['ip_adapter_proj_model'].to(dtype=torch.bfloat16)
        ip_projes = ip_adapter_flux['ip_adapter_proj_model'](out.to(ip_projes_dev, dtype=torch.bfloat16)).to(device, dtype=torch.bfloat16)
        ip_neg_pr = ip_adapter_flux['ip_adapter_proj_model'](neg_out.to(ip_projes_dev, dtype=torch.bfloat16)).to(device, dtype=torch.bfloat16)


        ipad_blocks = []
        for block in ip_adapter_flux['double_blocks']:
            ipad = IPProcessor(block.context_dim, block.hidden_dim, ip_projes, ip_scale)
            ipad.load_state_dict(block.state_dict())
            ipad.in_hidden_states_neg = ip_neg_pr
            ipad.in_hidden_states_pos = ip_projes
            ipad.to(dtype=torch.bfloat16)
            npp = DoubleStreamMixerProcessor()
            npp.add_ipadapter(ipad)
            ipad_blocks.append(npp)
        pbar.update(mul)
        i=0
        for name, _ in attn_processors(tyanochky.diffusion_model).items():
            attribute = f"diffusion_model.{name}"
            #old = copy.copy(get_attr(bi.model, attribute))
            if attribute in model.object_patches.keys():
                old = copy.copy((model.object_patches[attribute]))
            else:
                old = None
            processor = merge_loras(old, ipad_blocks[i])
            processor.to(device, dtype=torch.bfloat16)
            bi.add_object_patch(attribute, processor)
            i+=1
        pbar.update(mul)
        return (bi,)



class ApplyAdvancedFluxIPAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "ip_adapter_flux": ("IP_ADAPTER_FLUX",),
                              "image": ("IMAGE",),
                              #"text_scale": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              "begin_strength": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              "end_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              "smothing_type": (["Linear", "First half", "Second half", "Sigmoid"],),
                              }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "applymodel"
    CATEGORY = "XLabsNodes"

    def applymodel(self, model, ip_adapter_flux, image, begin_strength, end_strength, smothing_type):
        debug=False


        device=mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        is_patched = is_model_pathched(model.model)

        print(f"Is model already patched? {is_patched}")
        mul = 1
        if is_patched:
            pbar = ProgressBar(5)
        else:
            mul = 3
            count = len(model.model.diffusion_model.double_blocks)
            pbar = ProgressBar(5*mul+count)

        bi = model.clone()
        tyanochky = bi.model

        clip = ip_adapter_flux['clip_vision']
        
        if isinstance(clip, FluxClipViT):
            #torch.Size([1, 526, 526, 3])
            #image = torch.permute(image, (0, ))
            #print(image.shape)
            #print(image)
            clip_device = next(clip.model.parameters()).device
            image = torch.clip(image*255, 0.0, 255)
            out = clip(image).to(dtype=torch.bfloat16)
            neg_out = clip(torch.zeros_like(image)).to(dtype=torch.bfloat16)
        else:
            print("Using old vit clip")
            clip_device = next(clip.parameters()).device
            pixel_values = clip_preprocess(image.to(clip_device)).float()
            out = clip(pixel_values=pixel_values)
            neg_out = clip(pixel_values=torch.zeros_like(pixel_values))    
            neg_out = neg_out[2].to(dtype=torch.bfloat16)
            out = out[2].to(dtype=torch.bfloat16)
        
        pbar.update(mul)
        if not is_patched:
            print("We are patching diffusion model, be patient please")
            patches=FluxUpdateModules(tyanochky, pbar)
            print("Patched succesfully!")
        else:
            print("Model already updated")
        pbar.update(mul)

        #TYANOCHKYBY=16
        ip_projes_dev = next(ip_adapter_flux['ip_adapter_proj_model'].parameters()).device
        ip_adapter_flux['ip_adapter_proj_model'].to(dtype=torch.bfloat16)
        out=torch.mean(out, 0)
        neg_out=torch.mean(neg_out, 0)
        ip_projes = ip_adapter_flux['ip_adapter_proj_model'](out.to(ip_projes_dev, dtype=torch.bfloat16)).to(device, dtype=torch.bfloat16)
        ip_neg_pr = ip_adapter_flux['ip_adapter_proj_model'](neg_out.to(ip_projes_dev, dtype=torch.bfloat16)).to(device, dtype=torch.bfloat16)


        count = len(ip_adapter_flux['double_blocks'])

        if smothing_type == "Linear":
            strength_model = LinearStrengthModel(begin_strength, end_strength, count)
        elif smothing_type == "First half":
            strength_model = FirstHalfStrengthModel(begin_strength, end_strength, count)
        elif smothing_type == "Second half":
            strength_model = SecondHalfStrengthModel(begin_strength, end_strength, count)
        elif smothing_type == "Sigmoid":
            strength_model = SigmoidStrengthModel(begin_strength, end_strength, count)
        else:
            raise ValueError("Invalid smothing type")


        ipad_blocks = []
        for i, block in enumerate(ip_adapter_flux['double_blocks']):
            ipad = IPProcessor(block.context_dim, block.hidden_dim, ip_projes, strength_model[i])
            ipad.load_state_dict(block.state_dict())
            ipad.in_hidden_states_neg = ip_neg_pr
            ipad.in_hidden_states_pos = ip_projes
            ipad.to(dtype=torch.bfloat16)
            npp = DoubleStreamMixerProcessor()
            npp.add_ipadapter(ipad)
            ipad_blocks.append(npp)
        pbar.update(mul)
        i=0
        for name, _ in attn_processors(tyanochky.diffusion_model).items():
            attribute = f"diffusion_model.{name}"
            #old = copy.copy(get_attr(bi.model, attribute))
            if attribute in model.object_patches.keys():
                old = copy.copy((model.object_patches[attribute]))
            else:
                old = None
            processor = merge_loras(old, ipad_blocks[i])
            processor.to(device, dtype=torch.bfloat16)
            bi.add_object_patch(attribute, processor)
            i+=1
        pbar.update(mul)
        return (bi,)



NODE_CLASS_MAPPINGS = {
    "FluxLoraLoader": LoadFluxLora,
    "LoadFluxControlNet": LoadFluxControlNet,
    "ApplyFluxControlNet": ApplyFluxControlNet,
    "ApplyAdvancedFluxControlNet": ApplyAdvancedFluxControlNet,
    "XlabsSampler": XlabsSampler,
    "ApplyFluxIPAdapter": ApplyFluxIPAdapter,
    "LoadFluxIPAdapter": LoadFluxIPAdapter,
    "ApplyAdvancedFluxIPAdapter": ApplyAdvancedFluxIPAdapter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoraLoader": "Load Flux LoRA",
    "LoadFluxControlNet": "Load Flux ControlNet",
    "ApplyFluxControlNet": "Apply Flux ControlNet",
    "ApplyAdvancedFluxControlNet": "Apply Advanced Flux ControlNet",
    "XlabsSampler": "Xlabs Sampler",
    "ApplyFluxIPAdapter": "Apply Flux IPAdapter",
    "LoadFluxIPAdapter": "Load Flux IPAdatpter",
    "ApplyAdvancedFluxIPAdapter": "Apply Advanced Flux IPAdapter",
}
