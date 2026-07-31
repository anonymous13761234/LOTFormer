"""Attention cores for the retrieval conversion experiments.

Each core is an ``nn.Module`` holding the *only* new trainable parameters introduced by
the conversion (LOTFormer's pivot, or Hedgehog's feature map). It operates on projected,
RoPE-applied tensors in ``(B, H, N, D)`` layout — exactly the tensors a HuggingFace
attention function receives — so a core can be attached to any Llama/Qwen2-style attention
module and driven by a registered ``AttentionInterface`` function (see ``convert.py``).

    core(q, k, v, key_padding_mask=None, distill=False) -> context (B, H, Tq, D)
    core.last_ce   # softmax-mimicry cross-entropy when distill=True, else None

``key_padding_mask``: (B, Tk) with True = pad (masked out). Both cores are bidirectional
(non-causal) — for use in encoder / embedding models only.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .sinkhorn_utils import SinkhornDistance


def _softmax_teacher_ce(q, k, scale, student_rownorm, keep):
    """Row-wise CE between softmax(q k^T * scale) (teacher, detached) and a
    row-normalized student attention matrix. All in (BH, Tq, Tk)."""
    logits = q @ k.transpose(-1, -2) * scale
    if keep is not None:
        logits = logits + (1.0 - keep)[:, None, :] * -1e9
    teacher = torch.softmax(logits, dim=-1).detach()
    ce = -(teacher * torch.log(student_rownorm.clamp_min(1e-8))).sum(dim=-1)   # (BH, Tq)
    return ce.mean()


class LotAttentionCore(nn.Module):
    """LOTFormer: doubly-stochastic linear attention via two Sinkhorn OT couplings
    against a learnable pivot measure of size ``num_refs``."""

    def __init__(self, num_heads, head_dim, num_refs=32, sink_eps=1.0, max_iter=5,
                 learn_z=False, attention_eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.attn_eps = attention_eps
        self.sink = SinkhornDistance(eps=sink_eps, max_iter=max_iter)
        self.ref = nn.Parameter(torch.randn(num_heads, num_refs, head_dim) * 0.02)
        self.learn_z = bool(learn_z)
        self.z_logits = nn.Parameter(torch.zeros(num_heads, num_refs)) if learn_z else None
        self.last_ce: Optional[torch.Tensor] = None

    def forward(self, q, k, v, key_padding_mask=None, distill=False):
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        BH = B * H
        qb = (q * self.scale).reshape(BH, Tq, D).float()
        kb = (k * self.scale).reshape(BH, Tk, D).float()
        vb = v.reshape(BH, Tk, D).float()
        Z = self.ref[None].expand(B, -1, -1, -1).reshape(BH, -1, D).float()
        R = Z.size(1)

        keep = None
        if key_padding_mask is not None:
            keep = (~key_padding_mask.bool()).to(torch.float32)              # (B,Tk)
            keep = keep[:, None, :].expand(B, H, Tk).reshape(BH, Tk)

        C_QZ = qb @ Z.transpose(1, 2)                                        # (BH,Tq,R)
        C_ZK = Z @ kb.transpose(1, 2)                                        # (BH,R,Tk)
        if keep is not None:
            C_ZK = C_ZK + (1.0 - keep)[:, None, :] * 1e6
        P_QZ = self.sink(C_QZ)
        P_ZK = self.sink(C_ZK)

        if self.learn_z:
            z = torch.softmax(self.z_logits, dim=-1).repeat_interleave(B, dim=0)
        else:
            z = torch.full((BH, R), 1.0 / R, device=q.device, dtype=torch.float32)
        inv_sqrt_z = (1.0 / (z + 1e-16)).sqrt()
        Phi_Q = P_QZ * inv_sqrt_z[:, None, :]                               # (BH,Tq,R)
        Phi_K = inv_sqrt_z[:, :, None] * P_ZK                               # (BH,R,Tk)

        KV = Phi_K @ vb
        out = Phi_Q @ KV
        ones = torch.ones(BH, Tk, 1, device=q.device, dtype=torch.float32)
        denom = (Phi_Q @ (Phi_K @ ones)).clamp_min(self.attn_eps)
        out = out / denom

        self.last_ce = None
        if distill:
            student = Phi_Q @ Phi_K
            if keep is not None:
                student = student * keep[:, None, :]
            student = student / student.sum(dim=-1, keepdim=True).clamp_min(self.attn_eps)
            self.last_ce = _softmax_teacher_ce(qb, kb, 1.0, student, keep)
        return out.view(B, H, Tq, D).to(q.dtype)


class HedgehogAttentionCore(nn.Module):
    """Hedgehog baseline: learnable feature map phi(x) = exp(W x + b), per head."""

    def __init__(self, num_heads, head_dim, phi_features=None, attention_eps=1e-6, clip=10.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.attn_eps = attention_eps
        self.clip = clip
        f = phi_features or head_dim
        self.W = nn.Parameter(torch.zeros(num_heads, head_dim, f))
        self.b = nn.Parameter(torch.zeros(num_heads, f))
        nn.init.xavier_uniform_(self.W, gain=0.5)
        self.register_buffer("_feat_scale", torch.tensor(1.0 / math.sqrt(f)))
        self.last_ce: Optional[torch.Tensor] = None

    def _phi(self, x):  # x: (B,H,N,D) -> (B,H,N,F)
        pre = torch.einsum("b h n d, h d f -> b h n f", x, self.W) + self.b[None, :, None, :]
        pre = pre.clamp(min=-self.clip, max=self.clip)
        return torch.nan_to_num(torch.exp(pre), nan=0.0, posinf=1e6, neginf=0.0)

    def forward(self, q, k, v, key_padding_mask=None, distill=False):
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        qf = self._phi(q) * self._feat_scale                                # (B,H,Tq,F)
        kf = self._phi(k) * self._feat_scale                                # (B,H,Tk,F)

        keep2 = None
        if key_padding_mask is not None:
            keep2 = (~key_padding_mask.bool()).to(qf.dtype)[:, None, :, None]  # (B,1,Tk,1)
            kf = kf * keep2
            v = v * keep2

        kv = torch.einsum("b h n f, b h n d -> b h f d", kf, v.float())
        num = torch.einsum("b h n f, b h f d -> b h n d", qf, kv)
        den = torch.einsum("b h n f, b h f -> b h n", qf, kf.sum(dim=2)).clamp_min(self.attn_eps)[..., None]
        out = num / den

        self.last_ce = None
        if distill:
            BH = B * H
            qb = (q * self.scale).reshape(BH, Tq, D).float()
            kb = k.reshape(BH, Tk, D).float()
            keep = None
            if key_padding_mask is not None:
                keep = (~key_padding_mask.bool()).to(torch.float32)[:, None, :].expand(B, H, Tk).reshape(BH, Tk)
            S = torch.einsum("b h n f, b h m f -> b h n m", qf, kf).reshape(BH, Tq, Tk)
            if keep is not None:
                S = S * keep[:, None, :]
            student = S / S.sum(dim=-1, keepdim=True).clamp_min(self.attn_eps)
            self.last_ce = _softmax_teacher_ce(qb, kb, 1.0, student, keep)
        return out.to(q.dtype)
