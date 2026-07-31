# Plug-and-play attention conversion: baselines

This folder implements a **model-agnostic conversion recipe**: take a pretrained
Transformer, replace its (non-causal) softmax self-attention with a subquadratic
attention, optionally **distill** the new attention to mimic softmax, then **finetune**.
The same recipe is used for LOTFormer and for baselines, so comparisons are controlled.

## The shared pipeline

```
swap  →  stage 1: attention distillation  →  stage 2: task finetune
```

- **Swap.** `swap_all_self_attention_with_lot` replaces every `BertSelfAttention` with a
  wrapper that keeps the original Q/K/V/out weights and only changes the attention math.
- **Stage 1 (distill).** Freeze the model; train *only* the swapped attention's parameters
  to minimize the row-wise cross-entropy between the softmax attention matrix (teacher,
  detached) and the converted attention's row-normalized weights (student).
- **Stage 2 (finetune).** Unfreeze and train on the task objective.

Every attention module exposes the identical interface, so all of the above is shared:

```python
module.forward(Q, K, V, mask=None, distill=False) -> context (B, H, N, D)
module._last_ce   # softmax-mimicry cross-entropy when distill=True, else None
```

## Attentions (`attention_zoo.py`)

| `--attn` | Module | Cost | Notes |
|---|---|---|---|
| `lot` | `LinearSinkAttention` (`lot_attention.py`) | O(n·r) | LOTFormer: doubly-stochastic OT with a learnable pivot (ours) |
| `hedgehog` | `HedgehogAttention` (`hedgehog_attention.py`) | O(n) | Learnable feature map `φ(x)=exp(Wx+b)` (Zhang et al., 2024) |

Select the attention with `--attn`; everything else (data, distillation, finetuning,
metrics) is identical.

```shell
# LOTFormer (ours)
python lot_two_stage.py --model_id JeremiahZ/bert-base-uncased-cola --task_name cola --attn lot
# Hedgehog baseline — same recipe, same budget
python lot_two_stage.py --model_id JeremiahZ/bert-base-uncased-cola --task_name cola --attn hedgehog
```

## Why Hedgehog is the right first baseline

Hedgehog uses the **same distillation objective** (softmax mimicry) as LOTFormer. Running
it through the *identical* pipeline means any quality gap is attributable to the
**attention family** — LOTFormer's doubly-stochastic optimal-transport coupling vs.
Hedgehog's exponential feature map — **not** to a difference in the training recipe. That
is the controlled comparison we want.

### Fairness rule
All baselines must get the **same conversion budget**: same distillation steps, same
corpus, same finetuning config. Otherwise the comparison is meaningless. The single
`--attn` switch enforces this by construction — only the attention module changes.

## Scaling to large long-context retrieval

This folder is the **small-scale reference** (BERT-base, GLUE). The identical recipe is
what we intend to apply to **large bidirectional embedding models** (e.g. gte-Qwen2-1.5B/7B,
NV-Embed, GritLM) for **long-context retrieval** (LoCo / LongEmbed / MTEB), where the
O(n·r) cost of LOTFormer is the real payoff at long sequence lengths. In that setting:

- **Convert** every encoder self-attention (fully bidirectional — no causal blocks).
- **Distill** attention maps on a long-document corpus, then **LoRA-finetune** with a
  contrastive (InfoNCE) objective.
- **Report** retrieval quality retention vs. softmax, the ranking against baselines
  (Hedgehog, and later Nyströmformer / Sinkformer / Performer), efficiency vs. context
  length, and how retention/speed-up behave across model scale.

The attention modules here are backbone-agnostic, so the same `HedgehogAttention` and
`LinearSinkAttention` drop into that large-model harness unchanged.
