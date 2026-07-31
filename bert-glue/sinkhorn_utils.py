"""Sinkhorn solver used by LOTFormer's linear attention (GLUE/BERT experiments)."""

import torch
import torch.nn as nn


class SinkhornDistance(nn.Module):
    """Entropic optimal transport via Sinkhorn iterations in log space.

    forward(c) treats ``c`` as a similarity (cost = -c) and returns the transport
    plan ``pi`` (uniform marginals) along with the cost and dual potentials.
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, c):
        C = -c
        x_points = C.shape[-2]
        y_points = C.shape[-1]
        batch_size = C.shape[0]
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / y_points).squeeze()
        if mu.dim() < 2:
            mu = mu.view(-1, 1)
        if nu.dim() < 2:
            nu = nu.view(-1, 1)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        thresh = 1e-12
        err = c.new_tensor(0.0)

        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u
                u = self.eps * (torch.log(mu) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) -
                                torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        return pi, C, U, V

    def M(self, C, u, v):
        "Modified cost for logarithmic updates: (-C + u_i + v_j) / eps"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
