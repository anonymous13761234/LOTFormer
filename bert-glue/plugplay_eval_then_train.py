#!/usr/bin/env python3
"""Plug-and-play LOTFormer conversion for BERT on GLUE.

Replace BERT's softmax self-attention with LOTFormer's LinearSinkAttention (copying
Q/K/V verbatim), evaluate immediately (no training), then finetune on the task and
evaluate again. Unlike ``lot_two_stage.py`` there is no attention-distillation stage;
this measures how well LOTFormer approximates softmax out of the box and after finetuning.
"""

import argparse
import inspect
import json
import os

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, TrainingArguments, set_seed,
)

# reuse the shared LOTFormer BERT wrapper + GLUE plumbing
from lot_two_stage import (
    TASK_TO_KEYS, glue_metrics, TaskOnlyTrainer, swap_all_self_attention_with_lot,
)


def build_training_args(args, metric_for_best):
    ta_kwargs = dict(
        output_dir=args.output_dir, learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps, fp16=args.fp16, logging_steps=50,
    )
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "report_to" in sig:
        ta_kwargs["report_to"] = []
    if "warmup_ratio" in sig:
        ta_kwargs["warmup_ratio"] = args.warmup_ratio
    if "lr_scheduler_type" in sig:
        ta_kwargs["lr_scheduler_type"] = args.scheduler
    if "evaluation_strategy" in sig:
        ta_kwargs.update(evaluation_strategy="epoch", save_strategy="epoch" if "save_strategy" in sig else "steps")
    elif "eval_strategy" in sig:
        ta_kwargs.update(eval_strategy="epoch", save_strategy="epoch" if "save_strategy" in sig else "steps")
    if "load_best_model_at_end" in sig:
        ta_kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in sig:
        ta_kwargs["metric_for_best_model"] = metric_for_best
    return TrainingArguments(**ta_kwargs)


def main():
    ap = argparse.ArgumentParser(description="Plug-and-play LOTFormer attention: eval then finetune.")
    ap.add_argument("--model_id", required=True, help="e.g. JeremiahZ/bert-base-uncased-cola")
    ap.add_argument("--task_name", default="cola", choices=list(TASK_TO_KEYS.keys()))
    ap.add_argument("--output_dir", default="./lot-plugplay")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn", default="lot", choices=["lot", "hedgehog"],
                    help="attention to convert to: 'lot' (LOTFormer, ours) or 'hedgehog' (baseline)")

    # LOTFormer knobs
    ap.add_argument("--num_refs", type=int, default=32)
    ap.add_argument("--phi_features", type=int, default=None,
                    help="hedgehog baseline: feature-map size (default = head_dim)")
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--sink_eps", type=float, default=0.05)
    ap.add_argument("--attention_eps", type=float, default=1e-6)
    ap.add_argument("--learn_z", action="store_true")

    # Training knobs
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--freeze_except_attn", action="store_true",
                    help="Freeze everything except the swapped attention modules.")
    ap.add_argument("--percent_cola", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    raw = load_dataset("glue", args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if args.fp16 else None)

    s1, s2 = TASK_TO_KEYS[args.task_name]

    def preprocess(ex):
        return tokenizer(ex[s1], ex[s2], truncation=True) if s2 else tokenizer(ex[s1], truncation=True)

    cols_to_remove = [c for c in raw["train"].column_names if c != "label"]
    encoded = raw.map(preprocess, batched=True, remove_columns=cols_to_remove)

    is_reg = args.task_name == "stsb"
    feat = raw["train"].features["label"]
    if is_reg:
        num_labels = 1
    else:
        num_labels = getattr(feat, "num_classes", None)
        if num_labels is None:
            names = getattr(feat, "names", None)
            num_labels = len(names) if names is not None else 2

    cfg = AutoConfig.from_pretrained(args.model_id, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, config=cfg)
    sink_cfg = dict(attn=args.attn, num_refs=args.num_refs, max_iter=args.max_iter,
                    sink_eps=args.sink_eps, attention_eps=args.attention_eps,
                    learn_z=args.learn_z, phi_features=args.phi_features)
    swap_all_self_attention_with_lot(model, sink_cfg)

    if args.freeze_except_attn:
        for n, p in model.named_parameters():
            p.requires_grad = (".attention.self." in n)

    metric_for_best = "matthews_correlation" if args.task_name == "cola" else (
                      "combined" if args.task_name in ["mrpc", "qqp", "stsb"] else "acc")

    def compute_metrics(eval_pred):
        preds = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)
        if preds is None or labels is None:
            preds, labels = eval_pred
        yhat = np.squeeze(preds) if is_reg else np.argmax(preds, axis=-1)
        return glue_metrics(args.task_name, yhat, labels, percent_cola=args.percent_cola)

    eval_split = encoded["validation_matched"] if args.task_name == "mnli" else encoded["validation"]
    kw = {"processing_class": tokenizer} if "processing_class" in inspect.signature(
        TaskOnlyTrainer.__init__).parameters else {"tokenizer": tokenizer}
    trainer = TaskOnlyTrainer(model=model, args=build_training_args(args, metric_for_best),
                              train_dataset=encoded["train"], eval_dataset=eval_split,
                              data_collator=collator, compute_metrics=compute_metrics, **kw)

    os.makedirs(args.output_dir, exist_ok=True)
    print(">>> pre-finetune evaluation with LOTFormer attention …")
    pre = trainer.evaluate()
    print("PRE-EVAL:", pre)
    with open(os.path.join(args.output_dir, "pre_eval.json"), "w") as f:
        json.dump(pre, f, indent=2)

    print(">>> finetuning …")
    trainer.train()

    print(">>> post-finetune evaluation …")
    post = trainer.evaluate()
    print("POST-EVAL:", post)
    with open(os.path.join(args.output_dir, "post_eval.json"), "w") as f:
        json.dump(post, f, indent=2)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved converted model + metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
