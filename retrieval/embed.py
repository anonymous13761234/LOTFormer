"""Load a (converted) HF embedder and encode texts to embeddings.

``load_encoder`` loads a bidirectional embedding model and optionally converts its
attention to LOTFormer / Hedgehog. ``encode`` produces pooled, L2-normalized embeddings.

Needs: transformers, torch. Run on a GPU box.
"""

import torch
import torch.nn.functional as F

from convert import attach_linear_attention


def load_encoder(model_id, attn="softmax", cfg=None, dtype="bfloat16",
                 device="cuda", trust_remote_code=True):
    """Return (model, tokenizer). attn='softmax' keeps the original model unchanged;
    attn in {'lot','hedgehog'} converts every self-attention module."""
    from transformers import AutoModel, AutoTokenizer
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype, attn_implementation="eager").to(device).eval()
    if attn != "softmax":
        model, n = attach_linear_attention(model, attn_name=attn, cfg=cfg)
        print(f"converted {n} attention modules -> {attn}")
    return model, tokenizer


def _pool(last_hidden, attention_mask, mode):
    if mode == "mean":
        w = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        return (last_hidden * w).sum(1) / w.sum(1).clamp_min(1e-6)
    if mode == "cls":
        return last_hidden[:, 0]
    if mode == "lasttoken":
        # index of the last non-pad token per row
        idx = attention_mask.sum(1) - 1
        return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idx]
    raise ValueError(f"unknown pooling '{mode}'")


@torch.no_grad()
def encode(model, tokenizer, texts, pooling="mean", max_length=8192,
           batch_size=8, device="cuda", normalize=True):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
        out = model(**enc)
        h = out.last_hidden_state
        e = _pool(h, enc["attention_mask"], pooling)
        if normalize:
            e = F.normalize(e, p=2, dim=-1)
        embs.append(e.float().cpu())
    return torch.cat(embs, 0)
