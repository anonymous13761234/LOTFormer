# LOTFormer: Doubly-Stochastic Linear Attention via Low-Rank Optimal Transport

This repository contains the anonymized reference implementation for **LOTFormer**, a
linear-time attention mechanism that views attention as a coupling between query and key
measures through optimal transport. By conditioning on a learnable pivot measure, LOTFormer
solves two entropic transport problems (queries→pivot and pivot→keys) that compose into a
doubly-stochastic coupling, operating in `O(n·r)` time without materializing the full
`n×n` attention matrix.

## Repository layout

```
imagenet_swin/        Image classification with Swin / PVT.
                      LOTFormer attention in models/lot_swin.py;
                      PolaFormer models included as baselines.

deit-imagenet/        Image classification with DeiT (distilled ViT).
                      Hybrid attention: softmax for the cls/distillation tokens,
                      LOTFormer for the image patch tokens (models/lot_deit.py).

LRA/                  Long Range Arena experiments.
                      LOTFormer attention: linear_sinkformer.py + sinkhorn_utils.py.
                      Standard LRA baselines: attention_{bigbird,informer,linformer,
                      nystrom,performer,reformer,skyformer}.py.

bert-glue/            BERT -> LOTFormer conversion on GLUE.
                      Plug-and-play (eval -> finetune) and two-stage
                      (softmax-mimicry distillation -> finetune) protocols.

fairseq-nmt/          LOTFormer plug-and-play for fairseq NMT (IWSLT'14 De-En).
                      Swaps the non-causal (encoder / cross) attention for
                      LOTFormer; decoder self-attention stays softmax.

retrieval/            Long-context retrieval with large bidirectional embedders
                      (gte-Qwen2 1.5B/7B, NV-Embed, GritLM). Plug-and-play
                      conversion (swap -> distill -> contrastive LoRA), with
                      Hedgehog as a controlled baseline.
```

See each subfolder's README for setup: `imagenet_swin/README.md`,
`deit-imagenet/README.md`, `bert-glue/README.md`, `fairseq-nmt/README.md`, and
`retrieval/README.md`.

## Long Range Arena (LRA)

All LRA methods share one training entry point; the attention variant is chosen via
`--attn`. LOTFormer corresponds to `sinkhorn_linear`.

```shell
cd LRA

# Prepare a task's data (see datasets/ for the create_*.py scripts), then train:
python train_taskwise_kernels.py --attn sinkhorn_linear --task lra-listops
```

Available `--attn` values: `softmax`, `nystrom`, `linformer`, `informer`, `performer`,
`bigbird`, `skyformer`, `reformer`, `sinkhorn_linear`.

Per-task hyperparameters (including the `sinkhorn_linear` settings) are defined in
`LRA/lra_config.py`.

## Acknowledgements

The LRA harness builds on standard public Long Range Arena baseline implementations, and
the vision code builds on the Swin Transformer, FLatten Transformer, and PolaFormer
codebases. See `imagenet_swin/README.md` and the respective upstream repositories for
details.
