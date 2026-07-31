"""
ImageNet training / evaluation for LOTFormer-DeiT.

A self-contained torch + torchvision harness (no timm dependency). It trains the
distilled ViT architecture; both the class head and the distillation head are
supervised with the ground-truth label (label-smoothed cross-entropy). For the
exact DeiT hard-distillation recipe, plug a teacher network into `train_one_epoch`
(see README).
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.lot_deit import build_model, MODELS


def get_args():
    p = argparse.ArgumentParser("LOTFormer-DeiT ImageNet", add_help=True)
    p.add_argument("--data-path", required=True, type=str,
                   help="ImageNet root with train/ and val/ subfolders")
    p.add_argument("--model", default="lot_deit_small_patch16_224",
                   choices=list(MODELS))
    p.add_argument("--num-classes", default=1000, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--epochs", default=300, type=int)
    p.add_argument("--warmup-epochs", default=5, type=int)
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--min-lr", default=1e-5, type=float)
    p.add_argument("--weight-decay", default=0.05, type=float)
    p.add_argument("--label-smoothing", default=0.1, type=float)
    p.add_argument("--drop-path", default=0.1, type=float)
    p.add_argument("--clip-grad", default=1.0, type=float)
    # LOTFormer hyperparameters
    p.add_argument("--num-refs", default=32, type=int, help="pivot-measure size r")
    p.add_argument("--sink-eps", default=1.0, type=float)
    p.add_argument("--sink-max-iter", default=5, type=int)
    p.add_argument("--kernel-size", default=5, type=int)
    # runtime
    p.add_argument("--workers", default=8, type=int)
    p.add_argument("--output", default="output", type=str)
    p.add_argument("--resume", default="", type=str)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--print-freq", default=50, type=int)
    return p.parse_args()


def build_loaders(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_set = datasets.ImageFolder(os.path.join(args.data_path, "train"), train_tf)
    val_set = datasets.ImageFolder(os.path.join(args.data_path, "val"), val_tf)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


def adjust_lr(optimizer, epoch, step, steps_per_epoch, args):
    """Cosine schedule with linear warmup."""
    total = args.epochs * steps_per_epoch
    warmup = args.warmup_epochs * steps_per_epoch
    cur = epoch * steps_per_epoch + step
    if cur < warmup:
        lr = args.lr * cur / max(1, warmup)
    else:
        progress = (cur - warmup) / max(1, total - warmup)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    n_classes = output.size(1)
    topk = tuple(min(k, n_classes) for k in topk)
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / target.size(0) for k in topk]


def train_one_epoch(model, loader, criterion, optimizer, epoch, args, device):
    model.train()
    steps = len(loader)
    t0 = time.time()
    for i, (images, target) in enumerate(loader):
        adjust_lr(optimizer, epoch, i, steps, args)
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)

        out = model(images)
        # distilled model returns (cls_logits, dist_logits) in training mode
        if isinstance(out, (tuple, list)):
            loss = 0.5 * (criterion(out[0], target) + criterion(out[1], target))
        else:
            loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        if i % args.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch}] [{i}/{steps}] loss {loss.item():.4f} "
                  f"lr {lr:.2e} ({(time.time()-t0)/(i+1):.2f}s/it)", flush=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1 = top5 = n = 0.0
    for images, target in loader:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        out = model(images)
        a1, a5 = accuracy(out, target, topk=(1, 5))
        bs = images.size(0)
        top1 += a1.item() * bs
        top5 += a5.item() * bs
        n += bs
    print(f"* Acc@1 {top1/n:.3f}  Acc@5 {top5/n:.3f}", flush=True)
    return top1 / n


def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | model: {args.model}")

    model = build_model(
        args.model, num_classes=args.num_classes, drop_path_rate=args.drop_path,
        num_refs=args.num_refs, sink_eps=args.sink_eps,
        sink_max_iter=args.sink_max_iter, kernel_size=args.kernel_size).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.1f}M")

    train_loader, val_loader = build_loaders(args)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch, best = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if not args.eval:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best = ckpt.get("best", 0.0)
        print(f"resumed from {args.resume} (epoch {start_epoch})")

    if args.eval:
        evaluate(model, val_loader, device)
        return

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, args, device)
        acc = evaluate(model, val_loader, device)
        best = max(best, acc)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best": best, "args": vars(args)},
                   os.path.join(args.output, "checkpoint.pth"))
        print(f"Epoch {epoch} done. acc@1 {acc:.3f} best {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
