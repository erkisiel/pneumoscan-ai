"""Does the uncertainty flag catch the model's mistakes?

Runs MC-dropout + TTA on the validation and test sets and reports, on the test set, the error rate of
high-confidence vs low-confidence predictions (at the screening threshold), plus a figure.

Usage: python src/evaluate_uncertainty.py --data data/pneumoniamnist_128.npz
Outputs: results/uncertainty.json, results/uncertainty.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from model import load_model
from uncertainty import MARGIN, STD_LIMIT, confidence_level, predict_with_uncertainty
import torch
from data import build_transforms


def run(model, img_size, images, n):
    tf = build_transforms(img_size, train=False)
    pts, stds = [], []
    for i, a in enumerate(images):
        img = Image.fromarray(a).convert("L")
        with torch.no_grad():
            pts.append(float(torch.softmax(model(tf(img)[None]), 1)[0, 1]))
        stds.append(predict_with_uncertainty(model, img, img_size, n=n, seed=i)[1])
    return np.array(pts), np.array(stds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="checkpoints/best_model.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    model, img_size, thr = load_model(args.ckpt)
    d = np.load(args.data)
    out = {}
    for split in ["val", "test"]:
        y = d[f"{split}_labels"].reshape(-1)
        p, s = run(model, img_size, d[f"{split}_images"], args.n)
        wrong = (p >= thr).astype(int) != y
        low = np.array([confidence_level(pi, si, thr)[0] == "low" for pi, si in zip(p, s)])
        out[split] = {
            "n": int(len(y)), "threshold": float(thr), "std_limit": STD_LIMIT, "margin": MARGIN,
            "errors": int(wrong.sum()), "low_conf": int(low.sum()),
            "errors_in_low_conf": int((wrong & low).sum()),
            "error_rate_high_conf": float(wrong[~low].mean()), "error_rate_low_conf": float(wrong[low].mean()),
            "accuracy_high_conf": float(1 - wrong[~low].mean()), "coverage_high_conf": float((~low).mean()),
            "mean_std_correct": float(s[~wrong].mean()), "mean_std_wrong": float(s[wrong].mean()),
        }
        if split == "test":
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            bins = np.linspace(0, max(s.max(), 0.2), 30)
            ax[0].hist(s[~wrong], bins, alpha=.7, label="Correct", color="#0B7A8A", density=True)
            ax[0].hist(s[wrong], bins, alpha=.7, label="Wrong", color="#C0392B", density=True)
            ax[0].axvline(STD_LIMIT, ls="--", c="k", lw=1); ax[0].set_xlabel("Prediction spread (std of 20 samples)")
            ax[0].set_ylabel("Density"); ax[0].set_title("Uncertainty of correct vs wrong predictions"); ax[0].legend()
            o = out["test"]
            bars = ax[1].bar(["High confidence", "Low confidence"],
                             [o["error_rate_high_conf"] * 100, o["error_rate_low_conf"] * 100], color=["#1E8449", "#D68910"])
            for b_, n_ in zip(bars, [o["n"] - o["low_conf"], o["low_conf"]]):
                ax[1].text(b_.get_x() + b_.get_width() / 2, b_.get_height() + .5, f"{b_.get_height():.1f}%\n(n={n_})", ha="center")
            ax[1].set_ylabel("Error rate (%)"); ax[1].set_title(f"Test error rate by confidence flag (τ={thr:.2f})")
            ax[1].set_ylim(0, max(o["error_rate_low_conf"] * 100 * 1.35, 10))
            fig.tight_layout(); fig.savefig(Path(args.out) / "uncertainty.png", dpi=160); plt.close(fig)
    (Path(args.out) / "uncertainty.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
