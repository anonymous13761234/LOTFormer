"""
DeiT / ViT with LOTFormer attention.

The self-attention in every transformer block is replaced by LOTFormer's
doubly-stochastic linear attention: two entropic optimal-transport problems
(queries -> pivot, pivot -> keys) are solved with Sinkhorn iterations against a
learnable pivot measure, then composed into a linear-time attention.

The attention mechanism is ported from the Swin implementation
(``imagenet_swin/models/lot_swin.py``). Two ViT-specific changes:

  * LOTFormer linear attention is applied to the image **patch tokens only**; the
    class token and the (DeiT) distillation token use ordinary **softmax attention**
    over the full sequence, since they are not part of the spatial grid.
  * The depthwise-convolution "local" branch, which needs a spatial H x W grid, is
    likewise applied to the patch tokens only.
"""

import math
from functools import partial

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinkhorn solver (ported verbatim from imagenet_swin/models/lot_swin.py)
# ---------------------------------------------------------------------------
class SinkhornDistance(nn.Module):
    def __init__(self, eps, max_iter, reduction='none'):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, c, mu=None, nu=None):
        B, N1, N2 = c.shape
        C = -c
        x_points = C.shape[-2]
        y_points = C.shape[-1]
        batch_size = C.shape[0]
        if mu is None:
            mu = torch.empty(batch_size, x_points, dtype=torch.float,
                             requires_grad=False, device=C.device).fill_(1.0 / x_points).squeeze()
        if nu is None:
            nu = torch.empty(batch_size, y_points, dtype=torch.float,
                             requires_grad=False, device=C.device).fill_(1.0 / y_points).squeeze()

        if mu.dim() < 2:
            mu = mu.view(-1, 1)
        if nu.dim() < 2:
            nu = nu.view(-1, 1)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        thresh = 1e-12

        mu_zero_mass_mask = (mu == 0.0)
        nu_zero_mass_mask = (nu == 0.0)
        mu_zero_mass_mask_expanded = mu_zero_mass_mask.unsqueeze(-1).repeat(1, 1, N2)
        nu_zero_mass_mask_expanded = nu_zero_mass_mask.unsqueeze(-1).repeat(1, 1, N1).transpose(-2, -1)

        err = c.new_tensor(0.0)
        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u.clone()
                M = self.M(C, u, v)
                M[mu_zero_mass_mask_expanded] = -torch.inf
                M[nu_zero_mass_mask_expanded] = -torch.inf
                u = self.eps * (torch.log(mu) - torch.logsumexp(M, dim=-1)) + u
                u[mu_zero_mass_mask] = 0.0
                err = (u - u1).abs().sum(-1).mean()
            else:
                M = self.M(C, u, v)
                M[mu_zero_mass_mask_expanded] = -torch.inf
                M[nu_zero_mass_mask_expanded] = -torch.inf
                v = self.eps * (torch.log(nu) -
                                torch.logsumexp(M.transpose(-2, -1), dim=-1)) + v
                v[nu_zero_mass_mask] = 0.0
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        return pi, C, U, V

    def M(self, C, u, v):
        "Modified cost for logarithmic updates: (-C + u_i + v_j) / eps"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


# ---------------------------------------------------------------------------
# LOTFormer attention for a (Dei)ViT sequence
# ---------------------------------------------------------------------------
class LotAttention(nn.Module):
    r"""LOTFormer doubly-stochastic linear attention.

    Args:
        dim: token embedding dimension.
        num_heads: number of attention heads.
        num_patches: number of spatial patch tokens (must be a perfect square).
        num_special_tokens: leading non-spatial tokens (1 for cls, 2 for DeiT cls+dist).
        num_refs: size of the learnable pivot measure (r << n).
        sink_eps: entropic regularization for the Sinkhorn transports.
        sink_max_iter: number of Sinkhorn iterations.
        kernel_size: depthwise-conv kernel for the local branch (patch tokens only).
    """

    def __init__(self, dim, num_heads, num_patches, num_special_tokens=2,
                 num_refs=32, sink_eps=1.0, sink_max_iter=5, qkv_bias=True,
                 attn_drop=0., proj_drop=0., kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_patches = num_patches
        self.num_special_tokens = num_special_tokens
        self.grid = int(round(num_patches ** 0.5))
        assert self.grid * self.grid == num_patches, \
            "num_patches must be a perfect square for the depthwise-conv branch"

        self.sink = SinkhornDistance(eps=float(sink_eps), max_iter=int(sink_max_iter))

        # Learnable pivot measure (support points) and its marginal logits.
        self.ref = nn.Parameter(torch.randn(num_heads, num_refs, head_dim) * (1 / math.sqrt(head_dim)))
        self.z_logits = nn.Parameter(torch.zeros(num_heads, num_refs))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Local enhancement over the value map (patch tokens arranged as grid x grid).
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim,
                             kernel_size=kernel_size, groups=head_dim,
                             padding=kernel_size // 2)

        # Positional encoding added to keys, over the full token sequence.
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, num_special_tokens + num_patches, dim))

    def forward(self, x):
        """x: (B, N, C) with N = num_special_tokens + num_patches.

        Special tokens (cls / distillation) use softmax attention over the whole
        sequence; patch tokens use LOTFormer doubly-stochastic linear attention.
        """
        B, N, _ = x.shape
        D = self.head_dim
        H = self.num_heads
        BH = B * H
        S = self.num_special_tokens
        P = self.num_patches

        qkv = self.qkv(x).reshape(B, N, 3, H * D).permute(2, 0, 1, 3)
        Q, K, V = qkv.unbind(0)
        K = K + self.positional_encoding

        # (B, H, N, D)
        Qh = Q.reshape(B, N, H, D).permute(0, 2, 1, 3)
        Kh = K.reshape(B, N, H, D).permute(0, 2, 1, 3)
        Vh = V.reshape(B, N, H, D).permute(0, 2, 1, 3)

        out = x.new_zeros(B, H, N, D)

        # ---- softmax attention for the special tokens (cls / dist) ----
        # Their queries attend over the full token sequence (all keys/values).
        if S > 0:
            q_s = Qh[:, :, :S] * self.scale                       # (B, H, S, D)
            attn = torch.matmul(q_s, Kh.transpose(-1, -2))        # (B, H, S, N)
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out[:, :, :S] = torch.matmul(attn, Vh)                # (B, H, S, D)

        # ---- LOTFormer linear attention for the patch tokens ----
        Qp = Qh[:, :, S:].reshape(BH, P, D).to(torch.float32) * self.scale
        Kp = Kh[:, :, S:].reshape(BH, P, D).to(torch.float32) * self.scale
        Vp = Vh[:, :, S:].reshape(BH, P, D).to(torch.float32)

        Z = self.ref[None].expand(B, -1, -1, -1).reshape(BH, -1, self.head_dim).to(torch.float32)

        z_head = torch.softmax(self.z_logits / 0.5, dim=-1)       # (H, R)
        z = z_head.repeat_interleave(BH // H, dim=0)              # (BH, R)

        # queries -> pivot and pivot -> keys entropic transports (patches only)
        S_qz = torch.matmul(Qp, Z.transpose(-1, -2))             # (BH, P, R)
        S_zk = torch.matmul(Kp, Z.transpose(-1, -2)).transpose(-1, -2)  # (BH, R, P)

        P_QZ = self.sink(S_qz, nu=z)[0]                           # (BH, P, R)
        P_ZK = self.sink(S_zk, mu=z)[0]                           # (BH, R, P)

        z_zero_mask = z == 0
        inv_sqrt_z = z.sqrt().reciprocal()
        inv_sqrt_z = torch.masked_fill(inv_sqrt_z, z_zero_mask, 0.0)

        Phi_Q = self.attn_drop(P_QZ * inv_sqrt_z[:, None, :])     # (BH, P, R)
        Phi_K = inv_sqrt_z[:, :, None] * P_ZK                     # (BH, R, P)

        KV = torch.matmul(Phi_K, Vp)                             # (BH, R, D)
        out_p = P * torch.matmul(Phi_Q, KV)                     # (BH, P, D)

        # Local branch: depthwise conv over the patch value grid.
        v_local = Vp.reshape(BH, self.grid, self.grid, D).permute(0, 3, 1, 2)
        v_local = self.dwc(v_local)
        v_local = v_local.permute(0, 2, 3, 1).reshape(BH, P, D)
        out_p = out_p + v_local

        out[:, :, S:] = out_p.reshape(B, H, P, D).to(out.dtype)

        out = out.permute(0, 2, 1, 3).reshape(B, N, H * D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, num_heads={self.num_heads}, '
                f'num_patches={self.num_patches}, num_special_tokens={self.num_special_tokens}')


# ---------------------------------------------------------------------------
# Transformer block / MLP
# ---------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep_prob) * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_patches, num_special_tokens, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_refs=32, sink_eps=1.0, sink_max_iter=5, kernel_size=5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LotAttention(
            dim, num_heads=num_heads, num_patches=num_patches,
            num_special_tokens=num_special_tokens, num_refs=num_refs,
            sink_eps=sink_eps, sink_max_iter=sink_max_iter, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, kernel_size=kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, Hh, Ww = x.shape
        assert (Hh, Ww) == self.img_size, \
            f"input size {(Hh, Ww)} does not match model {self.img_size}"
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ---------------------------------------------------------------------------
# LOTFormer DeiT (distilled ViT)
# ---------------------------------------------------------------------------
class LotDeiT(nn.Module):
    """Distilled Vision Transformer (DeiT) with LOTFormer attention.

    When ``distilled=True`` (default, matching DeiT) the model carries a cls token
    and a distillation token, and returns the average of the two heads at inference
    or the tuple ``(cls_logits, dist_logits)`` in training (for the distillation loss).
    Set ``distilled=False`` for a plain ViT with a single cls token.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, distilled=True,
                 num_refs=32, sink_eps=1.0, sink_max_iter=5, kernel_size=5):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.distilled = distilled
        self.num_special_tokens = 2 if distilled else 1

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_special_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, num_patches=num_patches,
                  num_special_tokens=self.num_special_tokens, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, num_refs=num_refs,
                  sink_eps=sink_eps, sink_max_iter=sink_max_iter, kernel_size=kernel_size)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        if distilled:
            nn.init.trunc_normal_(self.dist_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = {'pos_embed', 'cls_token'}
        if self.distilled:
            names.add('dist_token')
        for i in range(len(self.blocks)):
            names.add(f'blocks.{i}.attn.ref')
            names.add(f'blocks.{i}.attn.z_logits')
            names.add(f'blocks.{i}.attn.positional_encoding')
        return names

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls, dist, x), dim=1)
        else:
            x = torch.cat((cls, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.distilled:
            return x[:, 0], x[:, 1]
        return x[:, 0]

    def forward(self, x):
        if self.distilled:
            x_cls, x_dist = self.forward_features(x)
            x_cls = self.head(x_cls)
            x_dist = self.head_dist(x_dist)
            if self.training:
                return x_cls, x_dist
            return (x_cls + x_dist) / 2
        return self.head(self.forward_features(x))


# ---------------------------------------------------------------------------
# Model factory (DeiT sizes)
# ---------------------------------------------------------------------------
def lot_deit_tiny_patch16_224(**kwargs):
    return LotDeiT(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)


def lot_deit_small_patch16_224(**kwargs):
    return LotDeiT(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)


def lot_deit_base_patch16_224(**kwargs):
    return LotDeiT(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)


MODELS = {
    'lot_deit_tiny_patch16_224': lot_deit_tiny_patch16_224,
    'lot_deit_small_patch16_224': lot_deit_small_patch16_224,
    'lot_deit_base_patch16_224': lot_deit_base_patch16_224,
}


def build_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}'. available: {list(MODELS)}")
    return MODELS[name](**kwargs)
