# LOTFormer — fairseq NMT plug-and-play (IWSLT'14 De-En)

Plug LOTFormer's doubly-stochastic linear attention into a fairseq Transformer for
neural machine translation, mirroring the Sinkformer NMT experiment (Sander et al.,
2021). A standard `transformer_iwslt_de_en` is built, then its **non-causal**
attentions are swapped for LOTFormer attention.

## Which attentions are converted (and why)

Doubly-stochastic / low-rank optimal-transport attention aggregates over the whole key
set, so it is **not causal**. Following Sinkformers, LOTFormer is applied only to
non-causal attention:

- **Encoder self-attention** — always converted (bidirectional).
- **Encoder-decoder cross-attention** — optional, `--lot-convert encoder_cross`.
- **Decoder self-attention** — left as **softmax** (it is causal / auto-regressive).

The swap keeps the original Q/K/V/out projection weights and only changes the attention
computation (`LotMultiheadAttention.from_fairseq`).

## Files

```
lot_fairseq/
├── __init__.py                    fairseq --user-dir entry point (registers the model)
├── lot_multihead_attention.py     LotMultiheadAttention (fairseq-compatible drop-in)
├── lot_transformer.py             registers lot_transformer + lot_transformer_iwslt_de_en
└── sinkhorn_utils.py              Sinkhorn solver
```

## Dependencies

- PyTorch, `fairseq`, `sacrebleu` (for scoring). Tested against the classic
  (argparse) fairseq model-registration API used for the IWSLT'14 baseline.

## Data (IWSLT'14 De-En)

Use fairseq's standard preparation:

```shell
# from the fairseq repo
bash examples/translation/prepare-iwslt14.sh
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref iwslt14.tokenized.de-en/train \
    --validpref iwslt14.tokenized.de-en/valid \
    --testpref  iwslt14.tokenized.de-en/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 8
```

## Train

Point `--user-dir` at `lot_fairseq/` and select the LOTFormer architecture:

```shell
fairseq-train data-bin/iwslt14.tokenized.de-en \
    --user-dir /path/to/fairseq-nmt/lot_fairseq \
    --arch lot_transformer_iwslt_de_en --share-decoder-input-output-embed \
    --lot-num-refs 32 --lot-sink-eps 1.0 --lot-max-iter 5 \
    --lot-convert encoder \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/lot_iwslt
```

## Generate / score

```shell
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --user-dir /path/to/fairseq-nmt/lot_fairseq \
    --path checkpoints/lot_iwslt/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```

## LOTFormer hyperparameters

- `--lot-num-refs` — pivot-measure size *r* (landmarks per head).
- `--lot-sink-eps` — entropic regularization for the Sinkhorn transports.
- `--lot-max-iter` — number of Sinkhorn iterations.
- `--lot-learn-z` — learn the pivot prior `z` (otherwise uniform).
- `--lot-convert` — `encoder` (default) or `encoder_cross`.
