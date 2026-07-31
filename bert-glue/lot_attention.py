"""LOTFormer linear attention for the GLUE/BERT plug-and-play experiments.

``LinearSinkAttention`` computes doubly-stochastic linear attention via two Sinkhorn
optimal-transport couplings against a learnable pivot (landmark) measure. It is a
drop-in replacement for softmax self-attention that keeps the O(n) kernel trick.

When ``distill=True`` the forward pass additionally computes the row-wise
cross-entropy between the softmax attention matrix (teacher, detached) and the
row-normalized LOTFormer coupling ``Phi_Q @ Phi_K`` (student). This is the
softmax-mimicry objective used in stage-1 attention distillation, following the
Hedgehog recipe (Zhang et al., 2024).
"""

import math
from typing import Optional

import torch
from torch import nn

from sinkhorn_utils import SinkhornDistance


class LinearSinkAttention(nn.Module):
    """
    Config dict keys: head_dim, num_head, (optional) num_refs, max_iter, sink_eps,
    attention_eps, learn_z.
    forward(Q, K, V, mask, distill=False) with (B, H, N, D) tensors.
    """

    def __init__(self, config):
        super().__init__()
        d = config["head_dim"]
        h = config["num_head"]
        r = config.get("num_refs", 16)
        max_iter = config.get("max_iter", 5)
        sink_eps = config.get("sink_eps", 1)
        self.attn_eps = config.get("attention_eps", 1e-6)
        self.temperature = float(config.get("temperature", 1.0))

        self.scale = d ** -0.5
        self.sink = SinkhornDistance(eps=sink_eps, max_iter=max_iter)

        # learnable pivot / landmarks per head: (H, R, D)
        self.ref = nn.Parameter(torch.randn(h, r, d) * 0.02)

        # optional learnable prior over landmarks per head
        self.learn_z = bool(config.get("learn_z", False))
        self.z_logits = nn.Parameter(torch.zeros(h, r)) if self.learn_z else None

        self._last_ce: Optional[torch.Tensor] = None  # distillation loss, set when distill=True

    # ---- mask helpers (normalize HF masks to (BH, N) binary keep) ----
    @staticmethod
    def _mask_rows_cost(C, keep_mask, big=1e6):
        if keep_mask is None:
            return C
        return C + (1.0 - keep_mask)[:, :, None] * big

    @staticmethod
    def _mask_cols_cost(C, keep_mask, big=1e6):
        if keep_mask is None:
            return C
        return C + (1.0 - keep_mask)[:, None, :] * big

    def _prep_masks(self, mask, B, H, N, device):
        if mask is None:
            return None, None
        m = mask
        if m.dtype.is_floating_point:
            keep = (m == 0).to(torch.float32) if torch.any(m < 0) else m.to(torch.float32)
        else:
            keep = m.to(torch.float32)

        if keep.dim() == 2 and keep.size(-1) == N:
            keep = keep[:, None, :].expand(B, H, N).reshape(B * H, N).to(device)
            return keep, keep
        if keep.dim() == 4 and keep.size(1) == 1 and keep.size(2) == 1 and keep.size(3) == N:
            keep = keep.reshape(B, 1, 1, N).expand(B, H, 1, N).reshape(B * H, N).to(device)
            return keep, keep
        if keep.dim() == 4 and keep.size(1) == 1 and keep.size(2) == N and keep.size(3) == N:
            if torch.allclose(keep, keep[..., 0:1, :]):
                key_keep = keep[..., 0, :]
            else:
                key_keep = keep.amax(dim=2)
            key_keep = key_keep.expand(B, H, N).reshape(B * H, N).to(device)
            return key_keep, key_keep
        if keep.dim() == 3 and keep.size(0) == B and keep.size(1) == H and keep.size(2) == N:
            keep = keep.reshape(B * H, N).to(device)
            return keep, keep
        keep = keep.reshape(B, -1, N).expand(B, H, N).reshape(B * H, N).to(device)
        return keep, keep

    def forward(self, Q, K, V, mask=None, distill=False):
        """Q, K, V: (B, H, N, D) -> context (B, H, N, D)."""
        B, H, N, D = Q.shape
        device, out_dtype = Q.device, Q.dtype
        BH = B * H

        Qs = Q * self.scale
        Ks = K * self.scale

        Z = self.ref[None].expand(B, -1, -1, -1).reshape(BH, -1, D)   # (BH, R, D)
        R = Z.size(1)

        Q_bh = Qs.reshape(BH, N, D)
        K_bh = Ks.reshape(BH, N, D)
        V_bh = V.reshape(BH, N, D)
        qmask_bh, kmask_bh = self._prep_masks(mask, B, H, N, device)

        C_QZ = Q_bh.float() @ Z.float().transpose(1, 2)              # (BH, N, R)
        C_ZK = Z.float() @ K_bh.float().transpose(1, 2)             # (BH, R, N)
        C_QZ = self._mask_rows_cost(C_QZ, qmask_bh)
        C_ZK = self._mask_cols_cost(C_ZK, kmask_bh)

        P_QZ = self.sink(C_QZ)[0]                                    # (BH, N, R)
        P_ZK = self.sink(C_ZK)[0]                                    # (BH, R, N)

        if self.learn_z:
            z_head = torch.softmax(self.z_logits, dim=-1)           # (H, R)
            z = z_head.repeat_interleave(B, dim=0)                  # (BH, R)
        else:
            z = torch.full((BH, R), 1.0 / R, device=device, dtype=torch.float32)
        inv_sqrt_z = (1.0 / (z + 1e-16)).sqrt()

        Phi_Q = P_QZ * inv_sqrt_z[:, None, :]                       # (BH, N, R)
        Phi_K = inv_sqrt_z[:, :, None] * P_ZK                       # (BH, R, N)

        KV = Phi_K @ V_bh.float()                                   # (BH, R, D)
        out_bh = Phi_Q @ KV                                        # (BH, N, D)

        # row-normalize so rows sum to 1 (like softmax)
        ones = torch.ones(BH, N, 1, device=device, dtype=torch.float32)
        denom = (Phi_Q @ (Phi_K @ ones)).clamp_min(self.attn_eps)  # (BH, N, 1)
        out_bh = out_bh / denom
        if qmask_bh is not None:
            out_bh = out_bh * qmask_bh[:, :, None]

        # ---- optional attention distillation (softmax mimicry) ----
        self._last_ce = None
        if distill:
            self._last_ce = self._distill_ce(Q_bh, K_bh, Phi_Q, Phi_K, qmask_bh, kmask_bh, BH, N)

        return out_bh.to(out_dtype).view(B, H, N, D)

    def _distill_ce(self, Q_bh, K_bh, Phi_Q, Phi_K, qmask_bh, kmask_bh, BH, N):
        # teacher: softmax attention over keys (Q_bh/K_bh already carry one `scale`)
        logits = Q_bh.float() @ K_bh.float().transpose(1, 2)        # (BH, N, N)
        if kmask_bh is not None:
            logits = logits + (1.0 - kmask_bh)[:, None, :] * -1e9
        teacher = torch.softmax(logits / self.temperature, dim=-1).detach()

        # student: row-normalized LOTFormer coupling Phi_Q @ Phi_K
        S = Phi_Q @ Phi_K                                          # (BH, N, N)
        if kmask_bh is not None:
            S = S * kmask_bh[:, None, :]
        student = S / S.sum(dim=-1, keepdim=True).clamp_min(self.attn_eps)

        ce = -(teacher * torch.log(student.clamp_min(1e-8))).sum(dim=-1)   # (BH, N)
        if qmask_bh is not None:
            ce = ce * qmask_bh
            return ce.sum() / qmask_bh.sum().clamp_min(1.0)
        return ce.mean()
