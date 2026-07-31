"""Stage 1 — attention distillation (softmax mimicry).

Freeze the base model; train ONLY the attention cores (LOTFormer pivot / Hedgehog phi) so
each converted layer's attention matrix matches the original softmax attention. The loss
is the mean per-layer row-wise cross-entropy computed inside the cores (``distill=True``).

Distillation runs on unlabeled long documents (no retrieval labels needed).
Needs: transformers, datasets, torch. Run on a GPU box.
"""

import argparse

import torch

from embed import load_encoder
from convert import set_distill, collect_distill_losses, linear_core_parameters


def load_corpus(name, config, split, text_field, limit):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    ds = ds.select(range(min(limit, len(ds))))
    return [ex[text_field] for ex in ds if ex[text_field] and ex[text_field].strip()]


def main():
    ap = argparse.ArgumentParser("Stage-1 attention distillation")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--attn", default="lot", choices=["lot", "hedgehog"])
    ap.add_argument("--num_refs", type=int, default=32)
    ap.add_argument("--sink_eps", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=5)
    ap.add_argument("--learn_z", action="store_true")
    # distillation corpus
    ap.add_argument("--corpus", default="wikitext")
    ap.add_argument("--corpus_config", default="wikitext-103-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--max_length", type=int, default=4096)
    # optim
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save", default="./stage1_cores.pt")
    args = ap.parse_args()

    cfg = dict(num_refs=args.num_refs, sink_eps=args.sink_eps, max_iter=args.max_iter,
               learn_z=args.learn_z, phi_features=None)
    model, tokenizer = load_encoder(args.model_id, attn=args.attn, cfg=cfg, device=args.device)

    # freeze base; train only the cores
    for p in model.parameters():
        p.requires_grad = False
    core_params = list(linear_core_parameters(model))
    for p in core_params:
        p.requires_grad = True
    opt = torch.optim.AdamW(core_params, lr=args.lr)
    print(f"stage-1: training {sum(p.numel() for p in core_params)} core params")

    texts = load_corpus(args.corpus, args.corpus_config, args.split, args.text_field, args.limit)
    print(f"loaded {len(texts)} documents for distillation")

    model.train()
    set_distill(model, True)
    step = 0
    while step < args.steps:
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt").to(args.device)
            model(**enc)                              # forward populates core.last_ce
            loss = collect_distill_losses(model)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 20 == 0:
                print(f"step {step}/{args.steps}  distill_ce {loss.item():.4f}")
            step += 1
            if step >= args.steps:
                break
    set_distill(model, False)

    cores = {n: m._linear_core.state_dict()
             for n, m in model.named_modules() if hasattr(m, "_linear_core")}
    torch.save({"attn": args.attn, "cfg": cfg, "cores": cores}, args.save)
    print(f"saved distilled cores -> {args.save}")


if __name__ == "__main__":
    main()
