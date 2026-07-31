import os
import sys
import argparse
import random
import math
import json
import time
import itertools
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import redirect_stdout
from lra_config import Config
from model_wrapper import ModelForSC, ModelForSCDual
from dataset import LRADataset

# ---------------------------
# LR scheduler: warmup + cosine with floor
# ---------------------------
def build_warmup_cosine(optimizer, warmup_steps, total_steps, min_lr_ratio=0.4, start_factor=0.10):
    floor = float(min_lr_ratio)

    def lr_lambda(step):
        # linear warmup from start_factor*LR to LR
        if step < warmup_steps:
            return float(start_factor) + (1.0 - float(start_factor)) * (step + 1) / max(1, warmup_steps)
        # cosine decay with floor
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------------------
# Pretty print summary dict
# ---------------------------
def print_summary(summary, save_if_improved, model, checkpoint_path):
    summary["loss"] = np.mean(summary["loss"]) if len(summary["loss"]) else float("nan")
    summary["accu"] = np.mean(summary["accu"]) if len(summary["accu"]) else float("nan")

    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]

    summary_round = {}
    for key in summary:
        if isinstance(summary[key], str):
            summary_round[key] = summary[key]
        else:
            try:
                summary_round[key] = round(float(summary[key]), 4)
            except Exception:
                summary_round[key] = summary[key]

    print(summary_round, flush=True)

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []

# ---------------------------
# Helpers
# ---------------------------

### FIX: infinite cycling for train loader to avoid StopIteration
def make_infinite(loader):
    while True:
        for batch in loader:
            yield batch

### FIX: default total_loss_fn (identity) to replace undefined symbol
def identity_total_loss(base_loss, model):
    return base_loss

# ---------------------------
# One training/eval step with gradient accumulation + AMP
# ---------------------------
def step_LRA(model, optimizer, lr_scheduler, ds_iter, amp_scaler,
             accumu_steps, init_t, summary, component, step_idx, device, writer=None,
             clip_norm=5.0, total_loss_fn=None):  # ### FIX: default None

    if total_loss_fn is None:                   # ### FIX
        total_loss_fn = identity_total_loss     # ### FIX

    t0 = time.time()

    if component == "train":
        optimizer.zero_grad(set_to_none=True)

    # pull the next batch from the provided iterator for this component
    _, batch = next(ds_iter[component])         # iterator is managed by caller
    for key in batch:
        batch[key] = batch[key].to(device, non_blocking=True)

    if component == "train":
        outputs = {}
        partial_inputs_list = [{} for _ in range(accumu_steps)]
        for key in batch:
            chunks = torch.chunk(batch[key], accumu_steps, dim=0)
            # If batch_size is not divisible by accumu_steps, the last chunk is smaller — ok.
            for idx, inp in enumerate(chunks):
                partial_inputs_list[idx][key] = inp

        for partial_inputs in partial_inputs_list:
            if amp_scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    partial = model(**partial_inputs)

                    # --- LOSS (AMP): pass through total_loss_fn consistently
                    base_loss = partial["loss"].mean()
                    total_loss = total_loss_fn(base_loss, model)   # ### FIX
                    scaled_loss = total_loss / accumu_steps

                amp_scaler.scale(scaled_loss).backward()

                # Aggregate non-loss metrics
                for k, v in partial.items():
                    if k == "loss":
                        continue
                    try:
                        val = v.mean()
                    except Exception:
                        val = v
                    val = val / accumu_steps
                    if k not in ("accu", "logits"):
                        continue
                    if k not in outputs:
                        outputs[k] = val.detach()
                    else:
                        outputs[k] += val.detach()

                # Aggregate loss for logs
                if "loss" not in outputs:
                    outputs["loss"] = (total_loss / accumu_steps).detach()
                else:
                    outputs["loss"] += (total_loss / accumu_steps).detach()

            else:
                partial = model(**partial_inputs)

                base_loss = partial["loss"].mean()
                total_loss = total_loss_fn(base_loss, model)        # ### FIX
                scaled_loss = total_loss / accumu_steps
                scaled_loss.backward()

                for k, v in partial.items():
                    if k == "loss":
                        continue
                    try:
                        val = v.mean()
                    except Exception:
                        val = v
                    val = val / accumu_steps
                    if k not in ("accu", "logits"):
                        continue
                    if k not in outputs:
                        outputs[k] = val.detach()
                    else:
                        outputs[k] += val.detach()

                if "loss" not in outputs:
                    outputs["loss"] = (total_loss / accumu_steps).detach()
                else:
                    outputs["loss"] += (total_loss / accumu_steps).detach()

        if amp_scaler is not None:
            amp_scaler.unscale_(optimizer)

        if clip_norm and clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        if amp_scaler is not None:
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    else:
        with torch.no_grad():
            outputs = {}
            partial_inputs_list = [{} for _ in range(accumu_steps)]
            for key in batch:
                chunks = torch.chunk(batch[key], accumu_steps, dim=0)
                for idx, inp in enumerate(chunks):
                    partial_inputs_list[idx][key] = inp

            for partial_inputs in partial_inputs_list:
                partial = model(**partial_inputs)

                base_loss = partial.get("loss", None)
                if base_loss is not None:
                    base_loss = base_loss.mean() / accumu_steps
                    if "loss" not in outputs:
                        outputs["loss"] = base_loss.detach()
                    else:
                        outputs["loss"] += base_loss.detach()

                for k, v in partial.items():
                    if k in ("loss",):
                        continue
                    try:
                        val = v.mean()
                    except Exception:
                        val = v
                    val = val / accumu_steps
                    if k not in ("accu", "logits"):
                        continue
                    if k not in outputs:
                        outputs[k] = val.detach()
                    else:
                        outputs[k] += val.detach()

    t1 = time.time()
    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = float(outputs["loss"].item()) if "loss" in outputs else float("nan")
    accu = float(outputs["accu"].item()) if "accu" in outputs else float("nan")
    time_since_start = time.time() - init_t

    if step_idx % 100 == 0:
        print(
            f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, "
            f"bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}        ",
            end="\r",
            flush=True,
        )

    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)

    if writer is not None and component == "train":
        writer.add_scalar('train/loss', loss, step_idx)
        writer.add_scalar('train/accu', accu, step_idx)
        writer.add_scalar('train/lr', learning_rate, step_idx)

    return outputs

# ---------------------------
# Train loop
# ---------------------------
def train_LRA(model, optimizer, lr_scheduler, ds, amp_scaler,
              training_config, summary, writer, device):

    accumu_steps = training_config['accumu_steps']
    checkpoint_path = training_config['checkpoint_path']
    best_dev_accu = 0.0
    total_step = int(training_config["num_train_steps"])
    eval_frequency = int(training_config["eval_frequency"])

    init_t = time.time()

    ### FIX: infinite train iterator
    train_iter = enumerate(make_infinite(ds["train"]))

    model.train()
    for train_step_idx in range(total_step):
        _ = step_LRA(model, optimizer, lr_scheduler,
                     ds_iter={"train": train_iter},     # only needs 'train'
                     amp_scaler=amp_scaler,
                     accumu_steps=accumu_steps, init_t=init_t,
                     summary=summary, component='train', step_idx=train_step_idx,
                     writer=writer, device=device, clip_norm=5.0)

        if (train_step_idx + 1) % eval_frequency == 0:
            print()  # newline
            print_summary(summary["train"], False, model, checkpoint_path)

            # Fresh dev iterator each eval pass  ### FIX
            dev_iter = enumerate(ds["dev"])
            model.eval()
            for dev_step_idx in range(training_config["num_eval_steps"]):
                try:
                    _ = step_LRA(model, optimizer, lr_scheduler,
                                 ds_iter={"dev": dev_iter},   # only needs 'dev'
                                 amp_scaler=amp_scaler,
                                 accumu_steps=accumu_steps, init_t=init_t,
                                 summary=summary, component='dev', step_idx=dev_step_idx, device=device)
                except StopIteration:
                    break

            dev_accu = np.mean(summary["dev"]["accu"]) if len(summary["dev"]["accu"]) else 0.0
            if dev_accu > best_dev_accu:
                best_dev_accu = dev_accu
                if (train_step_idx + 1) > total_step * 0.2:
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
                    print('best model saved: step = ', train_step_idx, 'dev accu = ', dev_accu)

            print_summary(summary["dev"], True, model, checkpoint_path)
            model.train()

    print('total training step (k): {}'.format(total_step / 1000.0))
    print("total training time (s): {}".format(int(time.time() - init_t)))
    if torch.cuda.is_available():
        print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak'] >> 20))

# ---------------------------
# Eval loop (until StopIteration)
# ---------------------------
def eval_LRA(model, optimizer, lr_scheduler, ds, amp_scaler,
             training_config, summary, device):
    accumu_steps = training_config['accumu_steps']
    checkpoint_path = training_config['checkpoint_path']
    init_t = time.time()
    model.eval()
    # fresh test iterator each call  ### FIX
    test_iter = enumerate(ds["test"])
    try:
        for test_step_idx in itertools.count():
            _ = step_LRA(model, optimizer, lr_scheduler,
                         ds_iter={"test": test_iter},
                         amp_scaler=amp_scaler,
                         accumu_steps=accumu_steps, init_t=init_t,
                         summary=summary, component='test', step_idx=test_step_idx, device=device)
    except StopIteration:
        print_summary(summary["test"], False, model, checkpoint_path)

# ---------------------------
# Run-directory helpers (task/seed/attn/checkpoint)
# ---------------------------
def make_run_dirs(args):
    run_root = os.path.join(
        "runs",
        args.task,
        f"seed{args.random}",
        args.attn,
        args.checkpoint
    )
    dirs = {
        "root": run_root,
        "checkpoints": os.path.join(run_root, "checkpoints"),
        "logs": os.path.join(run_root, "logs"),
        "tensorboard": os.path.join(run_root, "tensorboard"),
        "preds": os.path.join(run_root, "preds"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ---------------------------
# CLI
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train eval")
    parser.add_argument("--checkpoint", type=str, default="test",
                        help="run tag; saved under runs/<task>/seed<seed>/<attn>/<checkpoint>/")
    parser.add_argument("--attn", type=str, default="softmaxQKV",
                        help="softmax, nystrom, linformer, informer, performer, bigbird, skyformer, reformer, sinkhorn_linear")
    parser.add_argument("--task", type=str, default="lra-listops",
                        help="lra-listops, lra-retrieval, lra-text, lra-pathfinder32-curv_contour_length_14")
    parser.add_argument('--random', type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use, e.g., 'cuda:0', 'cuda:1', 'cpu'")
    args = parser.parse_args()
    return args

# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.task == 'lra-pathfinder':
        args.task = 'lra-pathfinder32-curv_contour_length_14'

    # --- configs ---
    model_config = Config[args.task]["model"].copy()  # ### FIX: avoid in-place pollution
    if args.attn in Config[args.task]["extra_attn_config"]:
        model_config.update(Config[args.task]["extra_attn_config"][args.attn])
    model_config["mixed_precision"] = True   # set False during overfit debugging if needed
    model_config["attn_type"] = args.attn
    model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))
    model_config["random_seed"] = args.random

    training_config = Config[args.task]["training"].copy()  # ### FIX: avoid in-place pollution
    training_config.setdefault("min_lr_ratio", 0.4)
    training_config.setdefault("start_factor", 0.10)

    # --- per-task/run directories ---
    run_dirs = make_run_dirs(args)

    # --- logging to file ---
    log_path = os.path.join(run_dirs["logs"], "stdout.log")
    redirect_stdout(open(log_path, "w"))

    summary = {
        c: {"t": 0, "loss": [], "accu": [], "best_accu": 0, "component": c}
        for c in ["train", "dev", "test"]
    }
    writer = SummaryWriter(run_dirs["tensorboard"])

    print(json.dumps([model_config, training_config], indent=4))

    # --- seeds ---
    SEED = args.random
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ### FIX: determinism

    # --- model ---
    if args.task == "lra-retrieval":
        model = ModelForSCDual(model_config)
    else:
        model = ModelForSC(model_config)

    # --- checkpoints ---
    checkpoint_dir = run_dirs["checkpoints"]
    checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    training_config["checkpoint_path"] = checkpoint_path

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("model loaded from: " + checkpoint_path)

    model = model.to(device)
    print(model)
    print(f"parameter_size: {[w.size() for w in model.parameters()]}", flush=True)
    print(f"num_parameter: {np.sum([np.prod(w.size()) for w in model.parameters()])}", flush=True)

    # --- data (keep loaders; create iterators later where needed)  ### FIX
    ds = {
        "train": DataLoader(LRADataset(f"./datasets/{args.task}.train.pickle", True),
                            batch_size=training_config["batch_size"], drop_last=True, shuffle=True, pin_memory=True),
        "dev":   DataLoader(LRADataset(f"./datasets/{args.task}.dev.pickle", True),
                            batch_size=training_config["batch_size"], drop_last=True, shuffle=False, pin_memory=True),
        "test":  DataLoader(LRADataset(f"./datasets/{args.task}.test.pickle", False),
                            batch_size=training_config["batch_size"], drop_last=True, shuffle=False, pin_memory=True),
    }

    # --- optimizer ---
    print("lr is ...", training_config["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.0  # overfit-friendly default; change if needed
    )

    # --- scheduler: warmup + cosine with floor ---
    warmup_steps = int(training_config["warmup"])
    total_steps = int(training_config["num_train_steps"])
    lr_scheduler = build_warmup_cosine(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=training_config.get("min_lr_ratio", 0.4),
        start_factor=training_config.get("start_factor", 0.10),
    )

    # --- AMP scaler ---
    amp_scaler = torch.cuda.amp.GradScaler() if (model_config.get("mixed_precision", False) and torch.cuda.is_available()) else None

    # --- accumulation ---
    accumu_steps = model_config["bz_rate"] if "bz_rate" in model_config else 1
    print(f"accumu_steps={accumu_steps}")
    training_config['accumu_steps'] = accumu_steps

    # --- train ---
    if args.mode == 'train':
        t0 = time.time()
        train_LRA(model, optimizer, lr_scheduler, ds, amp_scaler,
                  training_config, summary, writer, device=device)

        # save brief run results after training
        results_path = os.path.join(run_dirs["root"], "results.json")
        results = {
            "task": args.task,
            "attn": args.attn,
            "seed": args.random,
            "checkpoint": checkpoint_path if os.path.exists(checkpoint_path) else None,
            "best_dev_accu": round(summary["dev"]["best_accu"], 4),
            "train_time_sec": int(time.time() - t0),
        }
        dump_json(results_path, results)

    # --- eval (best checkpoint if available) ---
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)
    eval_LRA(model, optimizer, lr_scheduler, ds, amp_scaler,
             training_config, summary, device=device)

    # write test summary
    test_results_path = os.path.join(run_dirs["root"], "test_summary.json")
    dump_json(test_results_path, {
        "mean_test_loss": round(np.mean(summary["test"]["loss"]), 4) if summary["test"]["loss"] else None,
        "mean_test_accu": round(np.mean(summary["test"]["accu"]), 4) if summary["test"]["accu"] else None,
    })

if __name__ == '__main__':
    main()
