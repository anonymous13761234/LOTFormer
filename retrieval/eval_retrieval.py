"""Evaluate a (converted) embedder on long-context retrieval via MTEB.

Wraps the model as an MTEB-compatible encoder and runs retrieval tasks (LoCo / LongEmbed /
BEIR), reporting nDCG@10. Compare runs with --attn softmax (baseline), lot, hedgehog to
fill Tables 1 and 7.

Needs: transformers, mteb, torch. Run on a GPU box.
"""

import argparse
import json

import torch

from embed import load_encoder, encode


class MtebEncoder:
    """Minimal MTEB-compatible wrapper exposing encode()."""

    def __init__(self, model, tokenizer, pooling, max_length, batch_size, device):
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    def encode(self, sentences, **kwargs):
        embs = encode(self.model, self.tokenizer, list(sentences), pooling=self.pooling,
                      max_length=self.max_length, batch_size=self.batch_size, device=self.device)
        return embs.numpy()

    # MTEB retrieval calls these when present; default to encode()
    def encode_queries(self, queries, **kwargs):
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus, **kwargs):
        texts = [((d.get("title", "") + " " + d.get("text", "")).strip() if isinstance(d, dict) else d)
                 for d in corpus]
        return self.encode(texts, **kwargs)


# A few long-context retrieval tasks. Adjust to the exact MTEB task names available.
DEFAULT_TASKS = [
    "LEMBSummScreenFDRetrieval",
    "LEMBQMSumRetrieval",
    "LEMBWikimQARetrieval",
    "LEMBNarrativeQARetrieval",
]


def main():
    ap = argparse.ArgumentParser("Long-context retrieval eval (MTEB)")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--attn", default="softmax", choices=["softmax", "lot", "hedgehog"])
    ap.add_argument("--num_refs", type=int, default=32)
    ap.add_argument("--stage2_model", default=None, help="optional finetuned LoRA dir to load")
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls", "lasttoken"])
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="./mteb_results")
    args = ap.parse_args()

    cfg = dict(num_refs=args.num_refs, sink_eps=1.0, max_iter=5)
    model, tokenizer = load_encoder(args.model_id, attn=args.attn, cfg=cfg, device=args.device)

    if args.stage2_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.stage2_model).to(args.device).eval()
        print(f"loaded stage-2 adapters from {args.stage2_model}")

    import mteb
    enc = MtebEncoder(model, tokenizer, args.pooling, args.max_length, args.batch_size, args.device)
    tasks = mteb.get_tasks(tasks=args.tasks)
    results = mteb.MTEB(tasks=tasks).run(enc, output_folder=args.output, encode_kwargs={"batch_size": args.batch_size})

    summary = {}
    for r in results:
        for split, scores in r.scores.items():
            for s in scores:
                summary.setdefault(r.task_name, {})[split] = s.get("ndcg_at_10", s.get("main_score"))
    print("nDCG@10:", json.dumps(summary, indent=2))
    with open(f"{args.output}/summary_{args.attn}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
