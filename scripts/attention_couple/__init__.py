from .node import AttentionCouple,AttentionCoupleOne
from ... import SYMBOL, NODE_SURFIX

NODE_CLASS_MAPPINGS = {
    f"AttentionCouple{NODE_SURFIX}": AttentionCouple,
    f"AttentionCoupleOne{NODE_SURFIX}": AttentionCoupleOne
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"AttentionCouple{NODE_SURFIX}": f"Attention Couple {SYMBOL}",
    f"AttentionCoupleOne{NODE_SURFIX}": f"Attention Couple One{SYMBOL}"
}

