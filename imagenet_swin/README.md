# LOTFormer — Vision (Swin) Experiments

This directory contains the image-classification experiments for **LOTFormer**, built on
top of the [Swin Transformer](https://github.com/microsoft/Swin-Transformer),
[FLatten Transformer](https://github.com/LeapLabTHU/FLatten-Transformer), and
[PolaFormer](https://arxiv.org/abs/2501.15061) codebases. The PolaFormer / PVT models
(`models/pola_swin.py`, `models/pola_pvt.py`) are included as **baselines**; the LOTFormer
attention is implemented in `models/lot_swin.py`.

## Dependencies

- Python 3.9
- PyTorch == 1.11.0
- torchvision == 0.12.0
- numpy
- timm == 0.4.12
- einops
- yacs

## Data preparation

The ImageNet dataset should be prepared as follows:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   └── ...
    └── ...
```

Configs for each model/variant live in `cfgs/` (e.g. `cfgs/lot_swin_t.yaml`,
`cfgs/pola_swin_t.yaml`).

## Evaluation

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --cfg <path-to-config-file> --data-path <imagenet-path> \
    --output <output-path> --eval --resume <path-to-weights>
```

## Training from scratch

See `pretrain.sh` and run:

```shell
bash pretrain.sh
```

## Acknowledgements

This code is developed on top of [Swin Transformer](https://github.com/microsoft/Swin-Transformer),
[FLatten Transformer](https://github.com/LeapLabTHU/FLatten-Transformer), and
[PolaFormer](https://github.com/ZacharyMeng/PolaFormer). Please see their repositories and
papers for the baseline implementations. The original license from the base codebase is
retained in `LICENSE`.
