"""Convert a HuggingFace bidirectional encoder/embedder to LOTFormer (or a baseline).

Uses the modern transformers ``AttentionInterface`` (v4.48+): we register a single
custom attention function that dispatches to a per-module attention *core* (LOTFormer or
Hedgehog) holding the only new trainable parameters. Loading is then:

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                      attn_implementation="eager")
    attach_linear_attention(model, attn_name="lot", cfg={...})

After this, every attention module runs the linear core instead of softmax, reusing the
model's own q/k/v/o projections and RoPE (RoPE is applied by the model *before* the
attention function, so the core sees rotated Q/K).

IMPORTANT: this targets bidirectional (non-causal) models — encoder / embedding LLMs such
as gte-Qwen2, NV-Embed, GritLM, LLM2Vec. Do not apply to causal decoders.

This file imports transformers lazily so the attention cores remain importable without it.
The integration is version/model-sensitive and is provided as a compile-checked scaffold;
run it in an environment with transformers + a GPU.
"""

import torch

from linear_attention import build_core


def _repeat_kv(x, n_rep):
    """(B, H_kv, N, D) -> (B, H_kv*n_rep, N, D), matching transformers.repeat_kv."""
    if n_rep == 1:
        return x
    B, H, N, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, N, D).reshape(B, H * n_rep, N, D)


def _key_padding_from_mask(attention_mask, device):
    """Derive (B, Tk) key padding (True = pad) from a HF attention mask, or None."""
    if attention_mask is None:
        return None
    am = attention_mask
    if am.dim() == 2:                       # (B, Tk): 1 keep, 0 pad
        return (am == 0)
    if am.dim() == 4:                       # (B, 1, Tq, Tk) additive: 0 keep, <0 pad
        return (am[:, 0, 0, :] < -1.0)
    return None


def linear_attention_interface(module, query, key, value, attention_mask,
                               scaling=None, dropout=0.0, **kwargs):
    """AttentionInterface-compatible function. query/key/value: (B, H, N, D).
    Returns (attn_output (B, N, H, D), attn_weights=None)."""
    n_groups = getattr(module, "num_key_value_groups", 1)
    key = _repeat_kv(key, n_groups)
    value = _repeat_kv(value, n_groups)

    kpm = _key_padding_from_mask(attention_mask, query.device)
    core = module._linear_core
    out = core(query, key, value, key_padding_mask=kpm,
               distill=getattr(module, "_distill_enabled", False))   # (B, H, N, D)
    out = out.transpose(1, 2).contiguous()                          # (B, N, H, D)
    return out, None


_REGISTERED = False


def register_interface(name="linear_attn"):
    """Register the custom attention function with transformers (idempotent)."""
    global _REGISTERED
    from transformers import AttentionInterface
    if not _REGISTERED:
        AttentionInterface.register(name, linear_attention_interface)
        _REGISTERED = True
    return name


def _iter_attention_modules(model):
    """Yield the self-attention modules of a Llama/Qwen2-style model."""
    for module in model.modules():
        if all(hasattr(module, p) for p in ("q_proj", "k_proj", "v_proj")):
            yield module


def attach_linear_attention(model, attn_name="lot", cfg=None):
    """Attach an attention core to every self-attention module and switch the model to it.

    Returns (model, n_converted). Only the attached cores' parameters are new; all
    projection weights are the model's originals.
    """
    impl = register_interface()
    n = 0
    for module in _iter_attention_modules(model):
        num_heads = getattr(module, "num_heads", None) or model.config.num_attention_heads
        head_dim = getattr(module, "head_dim", None) or (model.config.hidden_size // num_heads)
        core = build_core(attn_name, num_heads, head_dim, cfg)
        core = core.to(next(module.parameters()).device, dtype=next(module.parameters()).dtype)
        module._linear_core = core
        module._distill_enabled = False
        n += 1
    model.config._attn_implementation = impl
    return model, n


def set_distill(model, enabled: bool):
    for module in _iter_attention_modules(model):
        if hasattr(module, "_linear_core"):
            module._distill_enabled = enabled


def collect_distill_losses(model):
    """Mean of the per-layer softmax-mimicry CE losses (call after a forward pass)."""
    losses = [m._linear_core.last_ce for m in _iter_attention_modules(model)
              if hasattr(m, "_linear_core") and m._linear_core.last_ce is not None]
    if not losses:
        raise RuntimeError("No distillation losses collected; is set_distill(model, True) set?")
    return torch.stack(losses).mean()


def linear_core_parameters(model):
    """Iterator over just the new (attention-core) parameters — for stage-1 optimization."""
    for module in _iter_attention_modules(model):
        if hasattr(module, "_linear_core"):
            yield from module._linear_core.parameters()
