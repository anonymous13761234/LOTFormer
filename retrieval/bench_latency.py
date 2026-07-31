"""Latency / memory benchmark vs. sequence length (produces Table 2).

Compares plain softmax attention against the linear cores (LOTFormer, Hedgehog) on
random (B, H, N, D) tensors, sweeping the context length N. Runs on CPU or GPU; on GPU
it also reports peak memory. This isolates the attention op — the end-to-end model
speed-up is smaller because the FFN cost is shared.
"""

import argparse
import time

import torch

from linear_attention import build_core


def softmax_attention(q, k, v):
    scale = q.shape[-1] ** -0.5
    attn = torch.softmax((q * scale) @ k.transpose(-1, -2), dim=-1)
    return attn @ v


@torch.no_grad()
def _time_op(fn, iters, device):
    # warmup
    for _ in range(3):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000.0  # ms
    mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
    return dt, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--num_refs", type=int, default=32)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--methods", nargs="+", default=["softmax", "lot", "hedgehog"])
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    B, H, D = args.batch, args.heads, args.head_dim
    cores = {
        "lot": build_core("lot", H, D, dict(num_refs=args.num_refs, max_iter=5)).to(device),
        "hedgehog": build_core("hedgehog", H, D, dict()).to(device),
    }

    print(f"device={device} B={B} H={H} D={D} num_refs={args.num_refs}")
    header = "  N   | " + " | ".join(f"{m:>18s}" for m in args.methods)
    print(header)
    print("-" * len(header))
    for N in args.lengths:
        q = torch.randn(B, H, N, D, device=device)
        k = torch.randn(B, H, N, D, device=device)
        v = torch.randn(B, H, N, D, device=device)
        cells = []
        for m in args.methods:
            try:
                if m == "softmax":
                    fn = lambda: softmax_attention(q, k, v)
                else:
                    core = cores[m]
                    fn = lambda core=core: core(q, k, v)
                dt, mem = _time_op(fn, args.iters, device)
                cell = f"{dt:7.1f}ms/{mem:4.1f}G" if device.type == "cuda" else f"{dt:9.2f} ms"
            except RuntimeError as e:
                cell = "OOM" if "memory" in str(e).lower() else "ERR"
            cells.append(f"{cell:>18s}")
        print(f"{N:6d} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
