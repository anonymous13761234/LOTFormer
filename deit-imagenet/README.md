# LOTFormer — DeiT / ImageNet Experiments

DeiT (distilled Vision Transformer) with **LOTFormer** attention on ImageNet.

The self-attention in each transformer block is replaced by LOTFormer's
doubly-stochastic linear attention (two entropic optimal-transport problems against a
learnable pivot measure, solved with Sinkhorn iterations), ported from the Swin
implementation in [`../imagenet_swin/models/lot_swin.py`](../imagenet_swin/models/lot_swin.py).

## Hybrid attention

Following the ViT token layout, attention is **hybrid**:

- **Class token** and **distillation token** use ordinary **softmax attention** over the
  full token sequence — they are not part of the spatial grid, so LOTFormer's
  patch-grid mechanics (pivot transport + depthwise-conv local branch) do not apply.
- **Image patch tokens** use **LOTFormer** linear attention (Sinkhorn pivot transport)
  plus a depthwise-convolution local branch over the patch grid.

This is implemented in `models/lot_deit.py` (`LotAttention.forward`).

## Layout

```
models/lot_deit.py   LOTFormer attention + DeiT/ViT model + factory functions
main.py              ImageNet train / eval harness (torch + torchvision)
```

Model variants: `lot_deit_tiny_patch16_224`, `lot_deit_small_patch16_224`,
`lot_deit_base_patch16_224`.

## Dependencies

- Python 3.9+
- PyTorch >= 1.12
- torchvision

(No timm dependency — the model and harness are self-contained.)

## Data

Standard ImageNet folder layout:

```
imagenet/
├── train/<class>/*.jpeg
└── val/<class>/*.jpeg
```

## Train

```shell
python main.py \
    --data-path /path/to/imagenet \
    --model lot_deit_small_patch16_224 \
    --batch-size 256 --epochs 300 \
    --num-refs 32 --sink-eps 1.0 --sink-max-iter 5 \
    --output ./output
```

LOTFormer hyperparameters: `--num-refs` (pivot-measure size *r*), `--sink-eps`
(entropic regularization), `--sink-max-iter` (Sinkhorn iterations), `--kernel-size`
(local depthwise-conv kernel).

## Evaluate

```shell
python main.py --data-path /path/to/imagenet \
    --model lot_deit_small_patch16_224 --resume ./output/checkpoint.pth --eval
```

## Notes on the DeiT recipe

Both heads (class + distillation) are supervised with label-smoothed cross-entropy, so
the harness runs without a teacher network. To reproduce the exact DeiT **hard-distillation**
recipe, supply a teacher and replace the distillation-head loss in
`train_one_epoch` with the hard-label distillation objective (CE against the teacher's
top-1 prediction); strong augmentation / mixup / EMA from the original DeiT recipe can be
added there as well. The `models/lot_deit.py` model is also compatible with the official
DeiT training code: register the factory functions with `timm` and select them via
`--model`.
