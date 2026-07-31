#!/usr/bin/env python3
"""Two-stage LOTFormer conversion for BERT on GLUE (e.g., CoLA).

Stage 1 (distill): Freeze the entire pretrained model. Swap softmax self-attention for
LOTFormer's LinearSinkAttention (keeping Q/K/V verbatim) and train ONLY the LOTFormer
pivot parameters to minimize the row-wise cross-entropy between the softmax attention
weights and the row-normalized LOTFormer coupling Phi_Q @ Phi_K (softmax mimicry).

Stage 2 (finetune): Unfreeze all model params and train on the task loss (GLUE) with
AdamW, bs=8, lr=1e-5, wd=0, up to 10 epochs + early stopping.

This mirrors the two-stage Hedgehog recipe (Zhang et al., 2024), with LOTFormer's
doubly-stochastic linear attention in place of the Hedgehog feature map.
"""

import argparse
import inspect
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed,
)
from transformers.models.bert.modeling_bert import BertSelfAttention

from attention_zoo import build_attention

try:
    from transformers import EarlyStoppingCallback
    HAS_EARLY_STOP = True
except Exception:
    EarlyStoppingCallback = object  # type: ignore
    HAS_EARLY_STOP = False


# --------- BERT wrapper: swap only attention compute; keep q/k/v; expose aux loss ---------
class BertSelfAttentionLotSink(nn.Module):
    def __init__(self, hf_config, sink_cfg: dict):
        super().__init__()
        if hf_config.hidden_size % hf_config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_attention_heads = hf_config.num_attention_heads
        self.attention_head_size = hf_config.hidden_size // hf_config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hf_config.hidden_size, self.all_head_size)
        self.key = nn.Linear(hf_config.hidden_size, self.all_head_size)
        self.value = nn.Linear(hf_config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(hf_config.attention_probs_dropout_prob)

        cfg = dict(sink_cfg)
        attn_name = cfg.pop("attn", "lot")
        cfg.update({"head_dim": self.attention_head_size, "num_head": self.num_attention_heads})
        self.attn_name = attn_name
        self.attn = build_attention(attn_name, cfg)

        self.enable_distill: bool = False
        self._last_ce: Optional[torch.Tensor] = None

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False, cache_position=None, **kwargs):
        q = self._transpose_for_scores(self.query(hidden_states))
        k = self._transpose_for_scores(self.key(hidden_states))
        v = self._transpose_for_scores(self.value(hidden_states))
        ctx = self.attn(q, k, v, mask=attention_mask, distill=self.enable_distill)
        self._last_ce = self.attn._last_ce
        ctx = ctx.transpose(1, 2).contiguous().view(hidden_states.size(0), -1, self.all_head_size)
        ctx = self.dropout(ctx)
        return (ctx, None)


def swap_all_self_attention_with_lot(model, sink_cfg: dict):
    enc = model.bert.encoder
    for i in range(model.config.num_hidden_layers):
        attn = enc.layer[i].attention
        old: BertSelfAttention = attn.self
        new = BertSelfAttentionLotSink(model.config, sink_cfg)
        new.query.weight.data.copy_(old.query.weight.data); new.query.bias.data.copy_(old.query.bias.data)
        new.key.weight.data.copy_(old.key.weight.data);     new.key.bias.data.copy_(old.key.bias.data)
        new.value.weight.data.copy_(old.value.weight.data); new.value.bias.data.copy_(old.value.bias.data)
        attn.self = new


# ------------------------------ GLUE plumbing --------------------------------
TASK_TO_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def glue_metrics(task, preds, labels, percent_cola=False):
    from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
    if task == "stsb":
        pearson = float(np.corrcoef(preds.squeeze(), labels.squeeze())[0, 1])
        rp = preds.squeeze().argsort().argsort(); rt = labels.squeeze().argsort().argsort()
        spearman = float(np.corrcoef(rp, rt)[0, 1])
        return {"pearson": pearson, "spearmanr": spearman, "combined": (pearson + spearman) / 2}
    if task == "cola":
        mcc = float(matthews_corrcoef(labels, preds))
        return {"matthews_correlation": (mcc * 100.0) if percent_cola else mcc}
    if task in ["mrpc", "qqp"]:
        acc = float(accuracy_score(labels, preds)); f1 = float(f1_score(labels, preds))
        return {"acc": acc, "f1": f1, "combined": (acc + f1) / 2}
    return {"acc": float(accuracy_score(labels, preds))}


# ------------------------------ Trainers -------------------------------------
class DistillOnlyTrainer(Trainer):
    """Stage-1 trainer: uses ONLY the per-layer CE distillation term (averaged)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        for m in model.modules():
            if isinstance(m, BertSelfAttentionLotSink):
                m.enable_distill = True
        outputs = model(**inputs)
        ces = []
        for m in model.modules():
            if isinstance(m, BertSelfAttentionLotSink):
                if m._last_ce is not None:
                    ces.append(m._last_ce)
                m.enable_distill = False
                m._last_ce = None
        if not ces:
            raise RuntimeError("No distillation loss collected. Check masks/forward path.")
        loss = torch.stack(ces).mean()
        return (loss, outputs) if return_outputs else loss


class TaskOnlyTrainer(Trainer):
    """Stage-2 trainer: standard supervised task loss (no distillation)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


# ------------------------------ Utilities ------------------------------------
def build_training_args(args, *, stage: int, metric_for_best: str):
    ta_kwargs = dict(
        output_dir=os.path.join(args.output_dir, f"stage{stage}"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=50,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay if stage == 2 else 0.0,
    )
    if stage == 1:
        ta_kwargs["learning_rate"] = args.stage1_lr
        ta_kwargs["num_train_epochs"] = args.stage1_epochs
        ta_kwargs["lr_scheduler_type"] = "cosine"
        ta_kwargs["warmup_ratio"] = 0.0
    else:
        ta_kwargs["learning_rate"] = args.stage2_lr
        ta_kwargs["num_train_epochs"] = args.stage2_epochs
        ta_kwargs["lr_scheduler_type"] = args.scheduler
        ta_kwargs["warmup_ratio"] = args.warmup_ratio

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "report_to" in sig:
        ta_kwargs["report_to"] = []
    if "evaluation_strategy" in sig:
        ta_kwargs.update(evaluation_strategy="epoch",
                         save_strategy="epoch" if "save_strategy" in sig else "steps")
    elif "eval_strategy" in sig:
        ta_kwargs.update(eval_strategy="epoch",
                         save_strategy="epoch" if "save_strategy" in sig else "steps")
    if stage == 2 and "load_best_model_at_end" in sig:
        ta_kwargs["load_best_model_at_end"] = True
    if stage == 2 and "metric_for_best_model" in sig:
        ta_kwargs["metric_for_best_model"] = metric_for_best
    if stage == 2 and "label_smoothing_factor" in sig and args.label_smoothing > 0.0:
        ta_kwargs["label_smoothing_factor"] = args.label_smoothing
    return TrainingArguments(**ta_kwargs)


def main():
    ap = argparse.ArgumentParser(description="Two-stage LOTFormer conversion for GLUE (CoLA, etc.)")
    ap.add_argument("--model_id", required=True, help="e.g. JeremiahZ/bert-base-uncased-cola")
    ap.add_argument("--task_name", default="cola", choices=list(TASK_TO_KEYS.keys()))
    ap.add_argument("--output_dir", default="./lot-2stage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn", default="lot", choices=["lot", "hedgehog"],
                    help="which attention to convert to: 'lot' (LOTFormer, ours) or 'hedgehog' (baseline)")

    # LOTFormer config
    ap.add_argument("--num_refs", type=int, default=32, help="pivot-measure size r")
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--sink_eps", type=float, default=0.05)
    ap.add_argument("--attention_eps", type=float, default=1e-6)
    ap.add_argument("--learn_z", action="store_true", help="learn the pivot prior z")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--phi_features", type=int, default=None,
                    help="hedgehog baseline: number of feature-map features (default = head_dim)")

    # Stage-1 (distill) knobs
    ap.add_argument("--stage1_epochs", type=int, default=1)
    ap.add_argument("--stage1_lr", type=float, default=1e-3)

    # Stage-2 (finetune) knobs
    ap.add_argument("--stage2_epochs", type=int, default=10)
    ap.add_argument("--stage2_lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    ap.add_argument("--percent_cola", action="store_true")
    ap.add_argument("--pre_eval", action="store_true", help="Evaluate right after swap, before stage-1.")
    args = ap.parse_args()

    set_seed(args.seed)

    # Data
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

    # Model + swap
    cfg = AutoConfig.from_pretrained(args.model_id, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, config=cfg)
    sink_cfg = dict(attn=args.attn, num_refs=args.num_refs, max_iter=args.max_iter,
                    sink_eps=args.sink_eps, attention_eps=args.attention_eps,
                    learn_z=args.learn_z, temperature=args.temperature,
                    phi_features=args.phi_features)  # phi_features used by the hedgehog baseline
    swap_all_self_attention_with_lot(model, sink_cfg)

    def compute_metrics(eval_pred):
        preds = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)
        if preds is None or labels is None:
            preds, labels = eval_pred
        yhat = np.squeeze(preds) if is_reg else np.argmax(preds, axis=-1)
        return glue_metrics(args.task_name, yhat, labels, percent_cola=args.percent_cola)

    metric_for_best = "matthews_correlation" if args.task_name == "cola" else (
                      "combined" if args.task_name in ["mrpc", "qqp", "stsb"] else "acc")

    eval_split = encoded["validation_matched"] if args.task_name == "mnli" else encoded["validation"]

    def make_trainer(klass, training_args, **kw):
        if "processing_class" in inspect.signature(klass.__init__).parameters:
            kw["processing_class"] = tokenizer
        else:
            kw["tokenizer"] = tokenizer
        return klass(model=model, args=training_args, train_dataset=encoded["train"],
                     eval_dataset=eval_split, data_collator=collator, **kw)

    # Optional pre-eval (plug-and-play, right after swap)
    if args.pre_eval:
        t = make_trainer(TaskOnlyTrainer, build_training_args(args, stage=2, metric_for_best=metric_for_best),
                         compute_metrics=compute_metrics)
        print(">>> pre-distill (plug-and-play) evaluation …")
        print("PRE-EVAL:", t.evaluate())

    # Stage 1: attention distillation (train only the swapped-attention params)
    for n, p in model.named_parameters():
        p.requires_grad = (".attention.self.attn." in n)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> stage-1: attention distillation (trainable pivot params: {n_trainable}) …")
    trainer1 = make_trainer(DistillOnlyTrainer, build_training_args(args, stage=1, metric_for_best=metric_for_best))
    trainer1.train()

    # Stage 2: supervised finetuning (unfreeze everything)
    for p in model.parameters():
        p.requires_grad = True
    t2_kwargs = dict(compute_metrics=compute_metrics)
    if HAS_EARLY_STOP and "callbacks" in inspect.signature(Trainer.__init__).parameters:
        t2_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)]
    trainer2 = make_trainer(TaskOnlyTrainer, build_training_args(args, stage=2, metric_for_best=metric_for_best), **t2_kwargs)
    print(">>> stage-2: task finetuning …")
    trainer2.train()

    final_metrics = trainer2.evaluate()
    print("FINAL:", final_metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    trainer2.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved model + metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
