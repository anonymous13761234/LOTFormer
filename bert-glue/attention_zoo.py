"""Factory for the swappable attention modules used in the conversion experiments.

Every attention here shares one interface so it drops into the same BERT wrapper and
the same swap -> distill -> finetune pipeline:

    module.forward(Q, K, V, mask=None, distill=False) -> context (B, H, N, D)
    module._last_ce  # softmax-mimicry cross-entropy when distill=True, else None

Config dict must contain ``head_dim`` and ``num_head``; other keys are attention-specific.
"""

from lot_attention import LinearSinkAttention
from hedgehog_attention import HedgehogAttention

ATTENTIONS = {
    "lot": LinearSinkAttention,       # LOTFormer (ours)
    "hedgehog": HedgehogAttention,    # Hedgehog baseline
}


def build_attention(name, config):
    if name not in ATTENTIONS:
        raise ValueError(f"unknown attention '{name}'. available: {list(ATTENTIONS)}")
    return ATTENTIONS[name](config)
