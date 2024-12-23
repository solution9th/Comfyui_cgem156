from .load import LoraLoaderFromWeight, LoraLoaderWeightOnly
from .merge import LoraMerge
from .save import LoraSave
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"LoraLoaderFromWeight{NODE_SURFIX}": LoraLoaderFromWeight,
    f"LoraLoaderWeightOnly{NODE_SURFIX}": LoraLoaderWeightOnly,
    f"LoraMerger{NODE_SURFIX}": LoraMerge,
    f"LoraSave{NODE_SURFIX}": LoraSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"LoraLoaderFromWeight{NODE_SURFIX}": f"LoRA Loader From Weight {SYMBOL}",
    f"LoraLoaderWeightOnly{NODE_SURFIX}": f"LoRA Loader Weight Only {SYMBOL}",
    f"LoraMerger{NODE_SURFIX}": f"LoRA Merge {SYMBOL}",
    f"LoraSave{NODE_SURFIX}": f"LoRA Save {SYMBOL}",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
