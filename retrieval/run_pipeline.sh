#!/usr/bin/env bash
# End-to-end long-context retrieval conversion pipeline.
# Runs softmax (baseline), LOTFormer, and Hedgehog through the identical recipe.
# Requires a GPU box with: transformers, datasets, peft, mteb.
set -euo pipefail

MODEL=${MODEL:-Alibaba-NLP/gte-Qwen2-1.5B-instruct}   # or gte-Qwen2-7B-instruct
POOL=${POOL:-mean}
DEV=${DEV:-cuda}

# 0) Softmax baseline (no conversion) -> Table 1 / Table 7 reference
python eval_retrieval.py --model_id "$MODEL" --attn softmax --pooling "$POOL" --device "$DEV" \
    --output results/softmax

for ATTN in lot hedgehog; do
  # 1) stage-1 attention distillation (train only the attention cores)
  python distill.py --model_id "$MODEL" --attn "$ATTN" --device "$DEV" \
      --corpus wikitext --corpus_config wikitext-103-raw-v1 --limit 2000 \
      --steps 500 --save "ckpt/${ATTN}_stage1.pt"

  # 2) stage-2 contrastive LoRA finetuning
  python finetune.py --model_id "$MODEL" --attn "$ATTN" --pooling "$POOL" --device "$DEV" \
      --stage1_cores "ckpt/${ATTN}_stage1.pt" --save "ckpt/${ATTN}_stage2"

  # 3) retrieval eval (nDCG@10) -> Tables 1 / 7
  python eval_retrieval.py --model_id "$MODEL" --attn "$ATTN" --pooling "$POOL" --device "$DEV" \
      --stage2_model "ckpt/${ATTN}_stage2" --output "results/${ATTN}"
done

# 4) efficiency vs context length -> Table 2
python bench_latency.py --device "$DEV" --heads 16 --head_dim 64 \
    --lengths 2048 4096 8192 16384 32768 --methods softmax lot hedgehog
