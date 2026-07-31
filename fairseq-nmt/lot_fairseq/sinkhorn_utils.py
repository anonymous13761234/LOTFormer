"""Sinkhorn solver used by LOTFormer's linear attention (fairseq NMT plug-in)."""

import torch
import torch.nn as nn


class SinkhornDistance(nn.Module):
    """Entropic optimal transport via log-domain Sinkhorn iterations (uniform marginals).

    forward(c) treats ``c`` as a similarity (cost = -c) and returns the transport plan.
    """

    def __init__(self, eps, max_iter):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, c):
        C = -c
        x_points, y_points = C.shape[-2], C.shape[-1]
        batch_size = C.shape[0]
        mu = torch.full((batch_size, x_points), 1.0 / x_points, device=C.device, dtype=torch.float32)
        nu = torch.full((batch_size, y_points), 1.0 / y_points, device=C.device, dtype=torch.float32)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        thresh = 1e-12
        err = c.new_tensor(0.0)
        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u
                u = self.eps * (torch.log(mu) - torch.logsumexp(self._M(C, u, v), dim=-1)) + u
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) -
                                torch.logsumexp(self._M(C, u, v).transpose(-2, -1), dim=-1)) + v
            if err.item() < thresh:
                break
        return torch.exp(self._M(C, u, v))

    def _M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
