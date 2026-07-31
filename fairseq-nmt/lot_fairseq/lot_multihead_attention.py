"""LOTFormer multi-head attention as a fairseq-compatible drop-in.

``LotMultiheadAttention`` matches fairseq ``MultiheadAttention``'s constructor and
call convention (query/key/value of shape ``(T, B, C)``, ``key_padding_mask`` of shape
``(B, S)`` with ``True`` = pad), but computes LOTFormer's doubly-stochastic linear
attention (two Sinkhorn optimal-transport couplings against a learnable pivot measure)
instead of softmax.

Because doubly-stochastic / low-rank OT attention aggregates over the whole key set,
it is **not causal** — this module is intended for the bidirectional **encoder
self-attention** and (optionally) the **encoder-decoder cross-attention**, both of which
are non-causal. It raises on a causal ``attn_mask`` (decoder self-attention), which
should keep softmax. This mirrors how Sinkformers apply doubly-stochastic attention.
"""

import torch
import torch.nn as nn

from .sinkhorn_utils import SinkhornDistance


class LotMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 num_refs=32, sink_eps=1.0, max_iter=5, learn_z=False, attention_eps=1e-6):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_eps = attention_eps
        self.dropout_module = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.sink = SinkhornDistance(eps=sink_eps, max_iter=max_iter)
        self.ref = nn.Parameter(torch.randn(num_heads, num_refs, self.head_dim) * 0.02)
        self.learn_z = bool(learn_z)
        self.z_logits = nn.Parameter(torch.zeros(num_heads, num_refs)) if learn_z else None

    def forward(self, query, key=None, value=None, key_padding_mask=None,
                attn_mask=None, need_weights=False, need_head_weights=False,
                incremental_state=None, static_kv=False, before_softmax=False, **kwargs):
        if attn_mask is not None:
            raise ValueError(
                "LotMultiheadAttention is non-causal; do not use it for causal "
                "(decoder self-)attention. Apply it to encoder self-attention or cross-attention.")
        if incremental_state is not None:
            raise ValueError("LotMultiheadAttention does not support incremental decoding.")

        if key is None:
            key = query
        if value is None:
            value = query

        T, B, C = query.shape
        S = key.shape[0]
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(query).view(T, B, H, D).permute(1, 2, 0, 3)   # (B,H,T,D)
        k = self.k_proj(key).view(S, B, H, D).permute(1, 2, 0, 3)     # (B,H,S,D)
        v = self.v_proj(value).view(S, B, H, D).permute(1, 2, 0, 3)   # (B,H,S,D)

        keep = None
        if key_padding_mask is not None:
            keep = (~key_padding_mask.bool()).to(torch.float32)       # (B,S) 1=keep

        out = self._lot_attention(q, k, v, keep)                      # (B,H,T,D)
        out = out.permute(2, 0, 1, 3).reshape(T, B, C)
        out = self.dropout_module(self.out_proj(out))
        return out, None

    def _lot_attention(self, q, k, v, keep):
        B, H, T, D = q.shape
        S = k.shape[2]
        BH = B * H

        Qs = (q * self.scale).reshape(BH, T, D).float()
        Ks = (k * self.scale).reshape(BH, S, D).float()
        Vb = v.reshape(BH, S, D).float()
        Z = self.ref[None].expand(B, -1, -1, -1).reshape(BH, -1, D).float()   # (BH,R,D)
        R = Z.size(1)

        C_QZ = Qs @ Z.transpose(1, 2)                                # (BH,T,R)
        C_ZK = Z @ Ks.transpose(1, 2)                                # (BH,R,S)
        if keep is not None:
            keep_bh = keep[:, None, :].expand(B, H, S).reshape(BH, S)
            C_ZK = C_ZK + (1.0 - keep_bh)[:, None, :] * 1e6          # mask padded keys

        P_QZ = self.sink(C_QZ)                                       # (BH,T,R)
        P_ZK = self.sink(C_ZK)                                       # (BH,R,S)

        if self.learn_z:
            z = torch.softmax(self.z_logits, dim=-1).repeat_interleave(B, dim=0)   # (BH,R)
        else:
            z = torch.full((BH, R), 1.0 / R, device=q.device, dtype=torch.float32)
        inv_sqrt_z = (1.0 / (z + 1e-16)).sqrt()

        Phi_Q = P_QZ * inv_sqrt_z[:, None, :]                        # (BH,T,R)
        Phi_K = inv_sqrt_z[:, :, None] * P_ZK                        # (BH,R,S)

        KV = Phi_K @ Vb                                              # (BH,R,D)
        out = Phi_Q @ KV                                            # (BH,T,D)
        ones = torch.ones(BH, S, 1, device=q.device, dtype=torch.float32)
        denom = (Phi_Q @ (Phi_K @ ones)).clamp_min(self.attn_eps)   # (BH,T,1)
        out = out / denom
        return out.view(B, H, T, D).to(q.dtype)

    @classmethod
    def from_fairseq(cls, mha, num_refs=32, sink_eps=1.0, max_iter=5, learn_z=False):
        """Build a LOTFormer attention from a fairseq MultiheadAttention, copying q/k/v/out weights."""
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        bias = mha.q_proj.bias is not None
        new = cls(embed_dim, num_heads, dropout=0.0, bias=bias,
                  num_refs=num_refs, sink_eps=sink_eps, max_iter=max_iter, learn_z=learn_z)
        new.q_proj.load_state_dict(mha.q_proj.state_dict())
        new.k_proj.load_state_dict(mha.k_proj.state_dict())
        new.v_proj.load_state_dict(mha.v_proj.state_dict())
        new.out_proj.load_state_dict(mha.out_proj.state_dict())
        return new
