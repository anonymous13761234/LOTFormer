import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sinkhorn_utils import SinkhornDistance

class LinearSinkAttentionCore(nn.Module):
    def __init__(self, head_dim, num_heads, *, num_refs=16, max_iter=5,
                 sink_eps=10.0, learn_z=True, debug=False,
                 inv_sqrt_cap=1e2, z_temp=0.5):
        super().__init__()
        self.num_heads    = int(num_heads)
        self.scale        = head_dim ** -0.5
        self.debug        = bool(debug)
        self.inv_sqrt_cap = float(inv_sqrt_cap)
        self.z_temp       = float(z_temp)
        self.learn_z      = bool(learn_z)
        
        self.sink = SinkhornDistance(
            eps=float(sink_eps),
            max_iter=int(max_iter))
        
        self.ref = nn.Parameter(torch.randn(num_heads, num_refs, head_dim)*0.02)
        
        self.attn_drop = nn.Identity()
    
    def forward(self, Q, K, V, mask=None):
        B, H, N, D = Q.shape
        assert H == self.num_heads
        BH = B * H
        device = Q.device
        out_dtype = Q.dtype
        
        Z  = self.ref[None].expand(B, -1, -1, -1).reshape(BH, -1, D).to(torch.float32)
        R  = Z.size(1)
        
        # Apply scaling more carefully to avoid overflow
        Qb = Q.reshape(BH, N, D).to(torch.float32) * self.scale
        Kb = K.reshape(BH, N, D).to(torch.float32) * self.scale
        Vb = V.reshape(BH, N, D).to(torch.float32)
        
        # Compute attention scores with gradient clipping
        S_qz = torch.matmul(Qb, Z.transpose(-1, -2))  # (BH,N,R)
        S_zk = torch.matmul(Z, Kb.transpose(-1, -2))  # (BH,R,N)
        
        # Clamp extreme values before masking
        S_qz = torch.clamp(S_qz, min=-50, max=50)
        S_zk = torch.clamp(S_zk, min=-50, max=50)

        P_QZ = self.sink(S_qz)[0]  # (BH,N,R)
        P_ZK = self.sink(S_zk)[0]  # (BH,R,N)
        
        inv_sqrt_z = torch.full((BH, R), R**-0.5, device=device, dtype=torch.float32)
        
        # Apply normalization with numerical stability
        Phi_Q = self.attn_drop(P_QZ * inv_sqrt_z[:, None, :])  # (BH,N,R)
        Phi_K = inv_sqrt_z[:, :, None] * P_ZK                  # (BH,R,N)
        
        KV = torch.matmul(Phi_K, Vb)  # (BH,R,D)
        
        out = N * torch.matmul(Phi_Q, KV)  # (BH,N,D)
        
        return out.view(B, H, N, D).to(out_dtype)

class LinearSinkAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.core = LinearSinkAttentionCore(
            head_dim     = int(config["head_dim"]),
            num_heads    = int(config["num_head"]),
            num_refs     = int(config.get("num_refs", 16)),
            max_iter     = int(config.get("max_iter", 8)),
            sink_eps     = float(config.get("sink_eps", 10.0)),
            learn_z      = bool(config.get("learn_z", True)),
            debug        = bool(config.get("debug", False)),
            inv_sqrt_cap = float(config.get("inv_sqrt_cap", 1e2)),
            z_temp       = float(config.get("z_temp", 0.5)),
        )
    
    def forward(self, Q, K, V, mask=None):
        return self.core(Q, K, V, mask=mask)