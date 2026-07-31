# LOTFormer — BERT / GLUE Conversion Experiments

Convert a pretrained softmax **BERT** into a **LOTFormer** (linear, doubly-stochastic
optimal-transport attention) model on GLUE tasks, then evaluate. Two protocols are
provided:

| Script | Protocol |
|---|---|
| `plugplay_eval_then_train.py` | **Plug-and-play:** swap attention → evaluate (zero-shot) → finetune → evaluate. No distillation. |
| `lot_two_stage.py` | **Two-stage:** swap attention → **stage 1** distill LOTFormer attention to mimic softmax (train pivot only) → **stage 2** finetune all params. |

The two-stage protocol follows the Hedgehog softmax-mimicry recipe (Zhang et al., 2024),
with LOTFormer's Sinkhorn attention in place of the Hedgehog feature map.

**Baselines.** Both scripts take `--attn {lot,hedgehog}` to select the attention that is
converted in (`lot` = LOTFormer, ours; `hedgehog` = baseline), keeping the rest of the
pipeline identical for a controlled comparison. See [`BASELINES.md`](BASELINES.md).

## How it works

`swap_all_self_attention_with_lot` replaces every `BertSelfAttention` with
`BertSelfAttentionLotSink`, which keeps the original Q/K/V projection weights and only
swaps the attention computation for `LinearSinkAttention` (`lot_attention.py`).

**Attention distillation (stage 1).** With `distill=True`, `LinearSinkAttention` computes
a row-wise cross-entropy between:
- **teacher** = `softmax(Q Kᵀ / √d)` (the original softmax attention, detached), and
- **student** = the row-normalized LOTFormer coupling `Φ_Q · Φ_K`.

Only the LOTFormer pivot parameters (`ref`, and `z_logits` if `--learn_z`) are trained in
stage 1; everything else is frozen. Stage 2 unfreezes the model and finetunes on the task.

## Files

```
lot_attention.py             LinearSinkAttention (+ optional distillation CE)
sinkhorn_utils.py            Sinkhorn solver
lot_two_stage.py             distill -> finetune
plugplay_eval_then_train.py  eval -> finetune (no distillation)
```

## Dependencies

- PyTorch, `transformers`, `datasets`, `scikit-learn`, `numpy`

## Usage

Two-stage conversion on CoLA, starting from a public fine-tuned BERT checkpoint:

```shell
python lot_two_stage.py \
    --model_id JeremiahZ/bert-base-uncased-cola --task_name cola \
    --num_refs 32 --sink_eps 0.05 --learn_z \
    --stage1_epochs 1 --stage1_lr 1e-3 \
    --stage2_epochs 10 --stage2_lr 1e-5 --batch_size 8 \
    --pre_eval --output_dir ./lot-2stage-cola
```

Plug-and-play (no distillation):

```shell
python plugplay_eval_then_train.py \
    --model_id JeremiahZ/bert-base-uncased-cola --task_name cola \
    --num_refs 32 --sink_eps 0.05 --epochs 10 --lr 1e-5 --batch_size 8 \
    --output_dir ./lot-plugplay-cola
```

Supported GLUE tasks: `cola`, `sst2`, `mrpc`, `qqp`, `stsb`, `mnli`, `qnli`, `rte`, `wnli`.

## LOTFormer hyperparameters

- `--num_refs` — pivot-measure size *r* (landmarks per head).
- `--sink_eps` — entropic regularization for the Sinkhorn transports.
- `--max_iter` — Sinkhorn iterations.
- `--learn_z` — learn the pivot prior `z` (otherwise uniform).
- `--temperature` — softmax temperature for the stage-1 teacher (two-stage only).
