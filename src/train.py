"""Model training.

Usage:
  python src/train.py --data data/pneumoniamnist_128.npz --epochs 15
  python src/train.py --data data/chest_xray --img-size 224 --pretrained
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from data import get_loaders
from model import build_model


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses, ys, ps = [], [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item() * len(y))
            ys.append(y.cpu().numpy())
            ps.append(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return {
        "loss": float(np.sum(losses) / len(ys)),
        "acc": float(((ps >= 0.5) == ys).mean()),
        "auc": float(roc_auc_score(ys, ps)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrained", action="store_true", help="Start from ImageNet weights (needs internet)")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_dl, val_dl, _ = get_loaders(args.data, args.img_size, args.batch_size)
    model = build_model(args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history, best_auc = [], -1.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_dl, device, criterion, optimizer)
        va = run_epoch(model, val_dl, device, criterion)
        scheduler.step()
        history.append({"epoch": ep, "train": tr, "val": va})
        print(f"Epoch {ep:02d} | train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.3f} acc {va['acc']:.3f} AUC {va['auc']:.3f} | {time.time()-t0:.0f}s")
        if va["auc"] > best_auc:
            best_auc = va["auc"]
            torch.save({"model": model.state_dict(), "img_size": args.img_size,
                        "epoch": ep, "val_auc": best_auc, "args": vars(args)}, out / "best_model.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation AUC: {best_auc:.4f}  ->  {out/'best_model.pt'}")


if __name__ == "__main__":
    main()
