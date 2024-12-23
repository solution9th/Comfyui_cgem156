import importlib
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
SYMBOL = "🍌"
NODE_SURFIX = f"|cgem156"
ROOT_NAME = f"cgem156 {SYMBOL}/"

scripts = [
    "batch_condition", 
    "dart", 
    "lortnoc",
    "attention_couple", 
    "cd_tuner", 
    "lora_merger",
    "multiple_lora_loader",
    "custom_samplers", 
    "custom_schedulers",
    "custom_guiders",
    "custom_noise",
    "scale_crafter", 
    "aesthetic_shadow",
    "for_test",
    "lora_xy"
]

try:
    import timm
except ImportError:
    print("timm is not installed. Skipping load wd-tagger.")
else:
    scripts.append("wd-tagger")
    
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

import_module = ".".join(current_directory.split("/")[-2:])
for script in scripts:
    module = importlib.import_module(f"{import_module}.scripts.{script}")
    if hasattr(module, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS.update(getattr(module, 'NODE_CLASS_MAPPINGS'))
    if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'))

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
