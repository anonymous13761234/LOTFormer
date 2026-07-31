"""Factory for the swappable attention cores."""

from .core import LotAttentionCore, HedgehogAttentionCore

CORES = {
    "lot": LotAttentionCore,        # LOTFormer (ours)
    "hedgehog": HedgehogAttentionCore,  # Hedgehog baseline
}


def build_core(name, num_heads, head_dim, cfg=None):
    cfg = dict(cfg or {})
    if name == "lot":
        return LotAttentionCore(
            num_heads, head_dim,
            num_refs=cfg.get("num_refs", 32), sink_eps=cfg.get("sink_eps", 1.0),
            max_iter=cfg.get("max_iter", 5), learn_z=cfg.get("learn_z", False),
            attention_eps=cfg.get("attention_eps", 1e-6))
    if name == "hedgehog":
        return HedgehogAttentionCore(
            num_heads, head_dim,
            phi_features=cfg.get("phi_features", None),
            attention_eps=cfg.get("attention_eps", 1e-6))
    raise ValueError(f"unknown attention '{name}'. available: {list(CORES)}")
