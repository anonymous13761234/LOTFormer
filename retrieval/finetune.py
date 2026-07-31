"""Stage 2 — contrastive LoRA finetuning of the converted embedder.

Loads the model, converts attention, (optionally) loads the stage-1 distilled cores, wraps
the base with LoRA, and trains with an in-batch InfoNCE loss on (query, positive) pairs.
The attention cores are trained together with the LoRA adapters.

Needs: transformers, peft, datasets, torch. Run on a GPU box.
"""

import argparse

import torch
import torch.nn.functional as F

from embed import load_encoder, _pool
from convert import linear_core_parameters


def load_pairs(name, config, split, q_field, d_field, limit):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    ds = ds.select(range(min(limit, len(ds))))
    return [(ex[q_field], ex[d_field]) for ex in ds]


def info_nce(q_emb, d_emb, temperature):
    q = F.normalize(q_emb, dim=-1)
    d = F.normalize(d_emb, dim=-1)
    logits = q @ d.t() / temperature                 # (B, B): diagonal = positives
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


def main():
    ap = argparse.ArgumentParser("Stage-2 contrastive LoRA finetuning")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--attn", default="lot", choices=["lot", "hedgehog"])
    ap.add_argument("--num_refs", type=int, default=32)
    ap.add_argument("--stage1_cores", default=None, help="path to distilled cores from distill.py")
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls", "lasttoken"])
    # data
    ap.add_argument("--pairs", default="sentence-transformers/msmarco")
    ap.add_argument("--pairs_config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--q_field", default="query")
    ap.add_argument("--d_field", default="positive")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--max_length", type=int, default=512)
    # optim / LoRA
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save", default="./stage2_model")
    args = ap.parse_args()

    cfg = dict(num_refs=args.num_refs, sink_eps=1.0, max_iter=5, learn_z=False, phi_features=None)
    model, tokenizer = load_encoder(args.model_id, attn=args.attn, cfg=cfg, device=args.device)

    if args.stage1_cores:
        ckpt = torch.load(args.stage1_cores, map_location=args.device)
        by_name = {n: m for n, m in model.named_modules() if hasattr(m, "_linear_core")}
        for name, sd in ckpt["cores"].items():
            if name in by_name:
                by_name[name]._linear_core.load_state_dict(sd)
        print(f"loaded stage-1 cores from {args.stage1_cores}")

    # LoRA on the base projections; cores stay fully trainable
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora)
    core_params = list(linear_core_parameters(model))
    for p in core_params:
        p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"stage-2: {sum(p.numel() for p in trainable)} trainable params (LoRA + cores)")

    pairs = load_pairs(args.pairs, args.pairs_config, args.split, args.q_field, args.d_field, args.limit)
    model.train()
    for epoch in range(args.epochs):
        for i in range(0, len(pairs) - args.batch_size, args.batch_size):
            batch = pairs[i:i + args.batch_size]
            q_txt = [p[0] for p in batch]; d_txt = [p[1] for p in batch]
            eq = tokenizer(q_txt, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(args.device)
            ed = tokenizer(d_txt, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(args.device)
            q_emb = _pool(model(**eq).last_hidden_state, eq["attention_mask"], args.pooling)
            d_emb = _pool(model(**ed).last_hidden_state, ed["attention_mask"], args.pooling)
            loss = info_nce(q_emb, d_emb, args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            if (i // args.batch_size) % 50 == 0:
                print(f"epoch {epoch} step {i // args.batch_size}  infonce {loss.item():.4f}")

    model.save_pretrained(args.save)
    tokenizer.save_pretrained(args.save)
    print(f"saved stage-2 model -> {args.save}")


if __name__ == "__main__":
    main()
