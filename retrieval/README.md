# LOTFormer — Long-Context Retrieval (large bidirectional embedders)

Plug-and-play conversion of large **bidirectional** embedding models (gte-Qwen2-1.5B/7B,
NV-Embed, GritLM, LLM2Vec) to LOTFormer's doubly-stochastic linear attention, then
evaluate long-context retrieval. **Hedgehog** is included as a controlled baseline: it runs
the *identical* swap → distill → finetune pipeline, so any quality gap is attributable to
the attention family, not the recipe.

> Applicability: these embedders use **bidirectional** (non-causal) attention, so every
> self-attention block is converted. Causal last-token embedders (E5-Mistral, SFR, Linq)
> are **out of scope** — LOTFormer is non-causal.

## Pipeline

```
convert (AttentionInterface swap)  →  stage 1: attention distillation  →  stage 2: contrastive LoRA
```

- **`convert.py`** — swaps attention via the modern transformers `AttentionInterface`:
  a registered function dispatches to a per-layer attention **core** (LOTFormer / Hedgehog)
  that holds the only new parameters and reuses the model's own Q/K/V/O projections + RoPE.
- **`distill.py`** — stage 1: freeze the base, train only the cores to mimic softmax
  attention (row-wise CE), on unlabeled long documents.
- **`finetune.py`** — stage 2: LoRA + contrastive InfoNCE on (query, positive) pairs.
- **`eval_retrieval.py`** — MTEB long-context retrieval (LoCo / LongEmbed), nDCG@10.
- **`bench_latency.py`** — latency / memory vs. sequence length.
- **`linear_attention/`** — the reusable attention cores (`lot`, `hedgehog`) + Sinkhorn.

## Files & dependencies

- Core math: `torch` only (`linear_attention/`, `bench_latency.py`) — runnable anywhere.
- Full pipeline: `transformers>=4.48` (for `AttentionInterface`), `datasets`, `peft`, `mteb`,
  and a GPU. The 7B runs need a large-memory GPU.

## Run

```shell
# core math + efficiency table run anywhere:
python bench_latency.py --heads 16 --head_dim 64 --lengths 512 1024 2048 4096 8192

# full pipeline (GPU box):
MODEL=Alibaba-NLP/gte-Qwen2-1.5B-instruct bash run_pipeline.sh
```

Swap the attention with a single flag everywhere: `--attn {softmax,lot,hedgehog}`.

## What each script produces (maps to the paper tables)

| Table | Meaning | Produced by |
|---|---|---|
| **Table 1** | Retrieval quality at fixed scale (softmax vs Hedgehog vs LOTFormer), nDCG@10 | `eval_retrieval.py` × `--attn` |
| **Table 2** | Latency / memory vs. context length | `bench_latency.py` |
| **Table 7** | Retention across model scale (1.5B → 7B, per method) | `eval_retrieval.py` across `--model_id` |
| **Recovery ablation** | zero-shot swap → +distill → +finetune | run `eval_retrieval.py` after each stage |

## Status / caveats

- The attention **cores** and the **latency benchmark** are unit-tested in pure PyTorch.
- The HuggingFace integration (`convert.py`) uses the `AttentionInterface` API and is a
  **compile-checked scaffold** — it is version/model-sensitive (RoPE handling, GQA
  `repeat_kv`, mask shapes, pooling) and has not been executed against a live 7B checkpoint
  in this repo. Expect to adjust per model/transformers version when you run it on a GPU box.
