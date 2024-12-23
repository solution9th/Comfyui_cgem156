import torch
import numpy as np
import math
import os
from PIL import Image,ImageOps
from ... import ROOT_NAME
import base64
from io import BytesIO
import node_helpers

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CATEGORY_NAME = ROOT_NAME + "batch_condition"

def lcm(a, b):
    return a * b // math.gcd(a, b)

def lcm_for_list(numbers):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm

class LoadMaskListBase64:
    _color_channels = ["red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "masks_b64": ("STRING", {"multiline": True}),
            "channel": (s._color_channels, ),
            }}

    RETURN_TYPES = ("MASKLIST",)
    OUTPUT_IS_LIST = (True,)
    CATEGORY = CATEGORY_NAME
    FUNCTION = "load_image"
    
    def load_image(self, masks_b64, channel):
        masks = []
        for mask_b64 in masks_b64.split("\n"):
            mask_b = base64.b64decode(mask_b64)
            # i = Image.open(BytesIO(mask))
            i = node_helpers.pillow(Image.open, BytesIO(mask_b))
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.getbands() != ("R", "G", "B", "A"):
                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                i = i.convert("RGBA")
            mask = None
            c = channel[0].upper()
            if c in i.getbands():
                mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask)
                if c == 'A':
                    mask = 1. - mask
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            masks.append(mask.unsqueeze(0))
        return (masks,)

class CLIPTextEncodeList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texts": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = ("CONDITIONINGLIST",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "encode"

    CATEGORY = CATEGORY_NAME
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, clip, texts):
        conds = []
        for text in texts.split("\n"):
            tokens = clip.tokenize(text)
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            conds.append([[cond, output]])
        return (conds, )

class CLIPTextEncodeBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ), 
                "texts":("BATCH_STRING", )
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY_NAME

    def encode(self, clip, texts):
        conds = []
        pooleds = []
        num_tokens = []
        for text in texts:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conds.append(cond)
            pooleds.append(pooled)
            num_tokens.append(cond.shape[1])
        
        # Make number of tokens equal
        # attn(q, k, v) == attn(q, [k]*n, [v]*n)
        lcm = lcm_for_list(num_tokens)
        repeats = [lcm//num for num in num_tokens]
        conds = torch.cat([cond.repeat(1, repeat, 1) for cond, repeat in zip(conds, repeats)])
        pooleds = torch.cat(pooleds)
        return ([[conds, {"pooled_output": pooleds}]], )
    
class StringInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "text": ("STRING", {"multiline": True})
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"

    CATEGORY = CATEGORY_NAME

    def encode(self, text):
        return (text, )
    
class BatchString:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("BATCH_STRING",)
    FUNCTION = "encode"

    CATEGORY = CATEGORY_NAME

    def encode(self, **kwargs):
        return ([kwargs[f"text{i+1}"] for i in range(len(kwargs))], )
    
class PrefixString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prefix": ("STRING", {"multiline": True}),
                "prompts": ("BATCH_STRING", )
            }
        }
    RETURN_TYPES = ("BATCH_STRING",)
    FUNCTION = "encode"

    CATEGORY = CATEGORY_NAME

    def encode(self, prefix, prompts):
        return ([prefix + prompt for prompt in prompts], )
        
    
class SaveBatchString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": ("BATCH_STRING", ),
                "folder": ("STRING", {"default": ""}),
                "extension": ("STRING", {"default": "txt"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_NAME

    def save(self, prompts, folder, extension, seed):
        os.makedirs(os.path.join(CURRENT_DIR, folder), exist_ok=True)
        for i, prompt in enumerate(prompts):
            path = os.path.join(CURRENT_DIR, folder, f"{seed:06}_{i:03}.{extension}")
            with open(path, "w") as f:
                f.write(prompt)
        return {}
    
class SaveImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "folder": ("STRING", {"default": ""}),
                "extension": ("STRING", {"default": "png"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_NAME

    def save(self, images, folder, extension, seed):
        os.makedirs(os.path.join(CURRENT_DIR, folder), exist_ok=True)
        for i, image in enumerate(images):
            path = os.path.join(CURRENT_DIR, folder, f"{seed:06}_{i:03}.{extension}")
            Image.fromarray((image.float().cpu() * 255).numpy().astype('uint8')).save(path)
        return {}
    
class SaveLatentBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT", ),
                "folder": ("STRING", {"default": ""}),
                "extension": (["npy", "npz"], {"default": "npy"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_NAME

    def save(self, latents, folder, extension, seed):
        os.makedirs(os.path.join(CURRENT_DIR, folder), exist_ok=True)
        for i, latent in enumerate(latents["samples"]):
            path = os.path.join(CURRENT_DIR, folder, f"{seed:06}_{i:03}.{extension}")
            if extension == "npy":
                np.save(path, latent.float().cpu().numpy())
            else:
                original_size = (latent.shape[1] * 8, latent.shape[2] * 8)
                crop_ltrb = (0, 0, 0, 0)
                np.savez(
                    path, 
                    latents=latent.float().cpu().numpy(),
                    original_size=np.array(original_size),
                    crop_ltrb=np.array(crop_ltrb),
                )
        return {}
