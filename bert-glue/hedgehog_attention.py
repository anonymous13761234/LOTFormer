"""Hedgehog linear attention baseline for the GLUE/BERT conversion experiments.

A drop-in with the same interface as ``lot_attention.LinearSinkAttention``:
``forward(Q, K, V, mask, distill=False)`` returns the context and, when
``distill=True``, stores the softmax-mimicry cross-entropy in ``self._last_ce``.

Hedgehog (Zhang et al., 2024) replaces softmax attention with a learnable feature
map ``phi(x) = exp(W x + b)`` (per head, shared for Q and K) and the linear-attention
kernel trick. Using it here as a baseline isolates the *attention family*: it shares
LOTFormer's exact distillation objective, so any quality gap is attributable to the
attention, not the training recipe.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PhiExpPerHead(nn.Module):
    """Per-head linear map + exp, shared between Q and K. (B,H,N,D) -> (B,H,N,F)."""

    def __init__(self, heads: int, d: int, f: int, clip: float = 10.0):
        super().__init__()
        self.clip = clip
        self.W = nn.Parameter(torch.zeros(heads, d, f))
        self.b = nn.Parameter(torch.zeros(heads, f))
        nn.init.xavier_uniform_(self.W, gain=0.5)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.einsum("b h n d, h d f -> b h n f", x, self.W) + self.b[None, :, None, :]
        pre = pre.clamp(min=-self.clip, max=self.clip)
        phi = torch.exp(pre)
        return torch.nan_to_num(phi, nan=0.0, posinf=1e6, neginf=0.0)


class HedgehogAttention(nn.Module):
    """Config dict keys: head_dim, num_head, (optional) phi_features, attention_eps,
    temperature, qk_norm. forward(Q, K, V, mask, distill=False) with (B, H, N, D)."""

    def __init__(self, config):
        super().__init__()
        d = int(config["head_dim"])
        h = int(config["num_head"])
        f = int(config.get("phi_features", d))
        self.eps = float(config.get("attention_eps", 1e-6))
        self.temperature = float(config.get("temperature", 1.0))
        self.qk_layernorm = bool(config.get("qk_norm", False))

        self.phi = PhiExpPerHead(h, d, f)
        self.q_norm = nn.LayerNorm(d) if self.qk_layernorm else nn.Identity()
        self.k_norm = nn.LayerNorm(d) if self.qk_layernorm else nn.Identity()
        self.register_buffer("_feat_scale", torch.tensor(1.0 / math.sqrt(f)))
        self._last_ce: Optional[torch.Tensor] = None

    @staticmethod
    def _keep_mask_from_hf(mask, B, H, N, device):
        if mask is None:
            return None
        m = mask
        if m.dtype.is_floating_point and torch.any(m < 0):
            keep = (m == 0).to(torch.float32)
        else:
            keep = m.to(torch.float32)
        if keep.dim() == 2 and keep.size(-1) == N:
            return keep[:, None, :].expand(B, H, N)
        if keep.dim() == 4 and keep.size(-1) == N and keep.size(-2) == 1:
            return keep.reshape(B, 1, 1, N).expand(B, H, 1, N).reshape(B, H, N)
        if keep.dim() == 4 and keep.size(-1) == N and keep.size(-2) == N:
            key_keep = keep[..., 0, :] if torch.allclose(keep, keep[..., 0:1, :]) else keep.amax(dim=-2)
            return key_keep.expand(B, H, N)
        if keep.dim() == 3 and keep.size() == (B, H, N):
            return keep
        return keep.reshape(B, -1, N).expand(B, H, N)

    def forward(self, Q, K, V, mask=None, distill=False):
        """Q, K, V: (B, H, N, D) -> context (B, H, N, D)."""
        B, H, N, D = Q.shape
        keep = self._keep_mask_from_hf(mask, B, H, N, Q.device)

        q = self.q_norm(Q)
        k = self.k_norm(K)
        qf = self.phi(q) * self._feat_scale                        # (B,H,N,F)
        kf = self.phi(k) * self._feat_scale                        # (B,H,N,F)

        if keep is not None:
            kf = kf * keep[..., None]
            V = V * keep[..., None]

        kv = torch.einsum("b h n f, b h n d -> b h f d", kf, V)
        num = torch.einsum("b h n f, b h f d -> b h n d", qf, kv)
        ksum = kf.sum(dim=2)
        den = torch.einsum("b h n f, b h f -> b h n", qf, ksum).clamp_min(self.eps)[..., None]
        out = num / den
        if keep is not None:
            out = out * keep[..., None]

        self._last_ce = None
        if distill:
            self._last_ce = self._distill_ce(Q, K, qf, kf, keep, D)
        return out

    def _distill_ce(self, Q, K, qf, kf, keep, D):
        logits = torch.einsum("b h n d, b h m d -> b h n m", Q * (1.0 / math.sqrt(D)), K)
        if keep is not None:
            logits = logits + (1.0 - keep)[:, :, None, :] * -1e9
        teacher = torch.softmax(logits / self.temperature, dim=-1).detach()

        S = torch.einsum("b h n f, b h m f -> b h n m", qf, kf)
        if keep is not None:
            S = S * keep[:, :, None, :]
        student = S / S.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        ce = -(teacher * torch.log(student.clamp_min(1e-8))).sum(dim=-1)   # (B,H,N)
        if keep is not None:
            ce = ce * keep
            return ce.sum() / keep.sum().clamp_min(1.0)
        return ce.mean()
