import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from .xflux.src.flux.math import attention, rope
from .xflux.src.flux.modules.layers import LoRALinearLayer

from torch.nn import functional as F
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

class DoubleStreamBlockLorasMixerProcessor(nn.Module):
    def __init__(self,):
        super().__init__()
        self.qkv_lora1 = []
        self.proj_lora1 = []
        self.qkv_lora2 = []
        self.proj_lora2 = []
        self.lora_weight = []
        self.names = []
    def add_lora(self, processor):
        if isinstance(processor, DoubleStreamBlockLorasMixerProcessor):
            self.qkv_lora1+=processor.qkv_lora1
            self.qkv_lora2+=processor.qkv_lora2
            self.proj_lora1+=processor.proj_lora1
            self.proj_lora2+=processor.proj_lora2
            self.lora_weight+=processor.lora_weight
        else:
            if hasattr(processor, "qkv_lora1"):
                self.qkv_lora1.append(processor.qkv_lora1)
            if hasattr(processor, "proj_lora1"):
                self.proj_lora1.append(processor.proj_lora1)
            if hasattr(processor, "qkv_lora2"):
                self.qkv_lora2.append(processor.qkv_lora2)
            if hasattr(processor, "proj_lora2"):
                self.proj_lora2.append(processor.proj_lora2)
            if hasattr(processor, "lora_weight"):
                self.lora_weight.append(processor.lora_weight)
    def get_loras(self):
        return (
            self.qkv_lora1, self.qkv_lora2, 
            self.proj_lora1, self.proj_lora2,
            self.lora_weight
        )
    def set_loras(self, qkv1s, qkv2s, proj1s, proj2s, w8s):
        for el in qkv1s:
            self.qkv_lora1.append(el)
        for el in qkv2s:
            self.qkv_lora2.append(el)
        for el in proj1s:
            self.proj_lora1.append(el)
        for el in proj2s:
            self.proj_lora2.append(el)
        for el in w8s:
            self.lora_weight.append(el)
        
    def add_shift(self, layer, origin, inputs, gating = 1.0):
        #shift = torch.zeros_like(origin)
        count = len(layer)
        for i in range(count):
            origin += layer[i](inputs)*self.lora_weight[i]*gating
        
    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        
        #img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_qkv = attn.img_attn.qkv(img_modulated)
        #print(self.qkv_lora1)
        self.add_shift(self.qkv_lora1, img_qkv, img_modulated)
            
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        
        
        #txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        self.add_shift(self.qkv_lora2, txt_qkv, txt_modulated)
        
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        #img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn) 
        self.add_shift(self.proj_lora1, img, img_attn, img_mod1.gate)
        
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)
        
        # calculate the txt bloks
        #txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) 
        self.add_shift(self.proj_lora2, txt, txt_attn, txt_mod1.gate)
        
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class DoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class DoubleStreamBlockProcessor(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt
    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        self.__call__(attn, img, txt, vec, pe, **attention_kwargs)


class IPProcessor(nn.Module):
    def __init__(self, context_dim, hidden_dim, ip_hidden_states=None, ip_scale=None, text_scale=None):
        super().__init__()
        self.ip_hidden_states = ip_hidden_states
        self.ip_scale = ip_scale
        self.text_scale = text_scale
        self.in_hidden_states_neg = None
        self.in_hidden_states_pos = ip_hidden_states
        # Ensure context_dim matches the dimension of ip_hidden_states
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        if text_scale is None:
            self.text_scale=1.0
        if self.text_scale is None:
            self.text_scale=1.0
        if self.ip_scale is None:
            self.ip_scale=1.0
        if self.text_scale == 0:
            self.text_scale = 0.0001
        # Initialize projections for IP-adapter
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)
        
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

    def forward(self, img_q, attn):
        #img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # IP-adapter processing
        ip_query = img_q  # latent sample query
        ip_key = self.ip_adapter_double_stream_k_proj(self.ip_hidden_states)
        ip_value = self.ip_adapter_double_stream_v_proj(self.ip_hidden_states)
        
        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads)
        #img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query, 
            ip_key, 
            ip_value, 
            dropout_p=0.0, 
            is_causal=False
        )
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads)
        return ip_attention*self.ip_scale

class ImageProjModel(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class DoubleStreamMixerProcessor(DoubleStreamBlockLorasMixerProcessor):
    def __init__(self,):
        super().__init__()
        self.ip_adapters = nn.ModuleList()
        
    def add_ipadapter(self, ip_adapter):
        self.ip_adapters.append(ip_adapter)

    def get_ip_adapters(self):
        return self.ip_adapters
    def set_ip_adapters(self, ip_adapters):
        self.ip_adapters = ip_adapters
    def shift_ip(self, img_qkv, attn, x):
        for block in self.ip_adapters:
            #x = x*block.text_scale
            x += torch.mean(block(img_qkv, attn), dim=0, keepdim=True)
        return x
    def scale_txt(self, txt):
        for block in self.ip_adapters:
            txt = txt * block.text_scale
        return txt
    def add_lora(self, processor):
        if isinstance(processor, DoubleStreamBlockLorasMixerProcessor):
            self.qkv_lora1+=processor.qkv_lora1
            self.qkv_lora2+=processor.qkv_lora2
            self.proj_lora1+=processor.proj_lora1
            self.proj_lora2+=processor.proj_lora2
            self.lora_weight+=processor.lora_weight
        elif isinstance(processor, DoubleStreamMixerProcessor):
            self.qkv_lora1+=processor.qkv_lora1
            self.qkv_lora2+=processor.qkv_lora2
            self.proj_lora1+=processor.proj_lora1
            self.proj_lora2+=processor.proj_lora2
            self.lora_weight+=processor.lora_weight
        else:
            if hasattr(processor, "qkv_lora1"):
                self.qkv_lora1.append(processor.qkv_lora1)
            if hasattr(processor, "proj_lora1"):
                self.proj_lora1.append(processor.proj_lora1)
            if hasattr(processor, "qkv_lora2"):
                self.qkv_lora2.append(processor.qkv_lora2)
            if hasattr(processor, "proj_lora2"):
                self.proj_lora2.append(processor.proj_lora2)
            if hasattr(processor, "lora_weight"):
                self.lora_weight.append(processor.lora_weight)

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        
        #img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_qkv = attn.img_attn.qkv(img_modulated)
        #print(self.qkv_lora1)
        self.add_shift(self.qkv_lora1, img_qkv, img_modulated)
            
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        
        
        #txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        self.add_shift(self.qkv_lora2, txt_qkv, txt_modulated)
        
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        #img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        self.add_shift(self.proj_lora1, img, img_attn, img_mod1.gate)        
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        
        img = self.shift_ip(img_q, attn, img)
        # calculate the txt bloks
        #txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) 
        
        
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        #txt = self.scale_txt(txt)
        self.add_shift(self.proj_lora2, txt, txt_attn, txt_mod1.gate)

        return img, txt
