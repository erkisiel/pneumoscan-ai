"""Test-set evaluation + figures for the report/slides.

Usage:
  python src/evaluate.py --data data/pneumoniamnist_128.npz --ckpt checkpoints/best_model.pt
Outputs: results/ folder (metrics.json, roc.png, confusion_matrix.png,
          training_curves.png, gradcam_examples.png)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_fscore_support, accuracy_score)

from data import get_dataset, MEAN, STD, CLASSES
from gradcam import GradCAM, overlay
from model import build_model

LABELS = {
    "tr": dict(fpr="Yanlış pozitif oranı (1 - özgüllük)", tpr="Doğru pozitif oranı (duyarlılık)", rand="Rastgele",
               roc="Test kümesi ROC eğrisi", cls=["Normal", "Zatürre"], pred="Tahmin", true="Gerçek",
               cm="Karışıklık matrisi", thr="eşik", train="Eğitim", val="Doğrulama", loss="Kayıp",
               tacc="Eğitim doğruluk", vacc="Doğrulama doğruluk", vauc="Doğrulama AUC", accauc="Doğruluk / AUC",
               pn="zatürre", gc="Grad-CAM: modelin dikkat ettiği bölgeler"),
    "en": dict(fpr="False positive rate (1 - specificity)", tpr="True positive rate (sensitivity)", rand="Chance",
               roc="Test set ROC curve", cls=["Normal", "Pneumonia"], pred="Predicted", true="True",
               cm="Confusion matrix", thr="threshold", train="Train", val="Validation", loss="Loss",
               tacc="Train accuracy", vacc="Validation accuracy", vauc="Validation AUC", accauc="Accuracy / AUC",
               pn="pneumonia", gc="Grad-CAM: regions the model attends to"),
}


@torch.no_grad()
def predict(model, ds, device, bs=128):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs)
    ps, ys = [], []
    for x, y in dl:
        ps.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"threshold": float(thr), "accuracy": float(accuracy_score(y, pred)),
            "sensitivity_recall": float(rec), "specificity": float(tn / (tn + fp)),
            "precision": float(prec), "f1": float(f1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="checkpoints/best_model.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--lang", default="en", choices=["en", "tr"], help="Figure language")
    args = ap.parse_args()
    L = LABELS[args.lang]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    img_size = ckpt["img_size"]
    model = build_model(False).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    # Threshold chosen on the VALIDATION set with Youden's J (no test leakage)
    yv, pv = predict(model, get_dataset(args.data, "val", img_size), device)
    fpr, tpr, thr = roc_curve(yv, pv)
    best_thr = float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))

    test_ds = get_dataset(args.data, "test", img_size)
    yt, pt = predict(model, test_ds, device)
    res = {"n_test": int(len(yt)), "auc": float(roc_auc_score(yt, pt)),
           "at_0.5": metrics_at(yt, pt, 0.5), "at_val_threshold": metrics_at(yt, pt, best_thr),
           "best_epoch": ckpt.get("epoch"), "val_auc": ckpt.get("val_auc")}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    ckpt["threshold"] = best_thr
    torch.save(ckpt, args.ckpt)  # store the threshold for the demo app
    print(json.dumps(res, indent=2))

    # ROC curve
    f, t, _ = roc_curve(yt, pt)
    plt.figure(figsize=(5, 5))
    plt.plot(f, t, lw=2, label=f"ResNet18 (AUC = {res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label=L["rand"])
    plt.xlabel(L["fpr"]); plt.ylabel(L["tpr"])
    plt.title(L["roc"]); plt.legend(loc="lower right"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out / "roc.png", dpi=200); plt.close()

    # Confusion matrix
    m = res["at_val_threshold"]
    cmx = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cmx, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cmx[i, j], ha="center", va="center", fontsize=14,
                     color="white" if cmx[i, j] > cmx.max() / 2 else "black")
    plt.xticks([0, 1], L["cls"]); plt.yticks([0, 1], L["cls"])
    plt.xlabel(L["pred"]); plt.ylabel(L["true"]); plt.title(f"{L['cm']} ({L['thr']}={best_thr:.2f})")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=200); plt.close()

    # Training curves
    hist_path = Path(args.ckpt).parent / "history.json"
    if hist_path.exists():
        h = json.loads(hist_path.read_text())
        ep = [e["epoch"] for e in h]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(ep, [e["train"]["loss"] for e in h], label=L["train"])
        ax[0].plot(ep, [e["val"]["loss"] for e in h], label=L["val"])
        ax[0].set_title(L["loss"]); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(ep, [e["train"]["acc"] for e in h], label=L["tacc"])
        ax[1].plot(ep, [e["val"]["acc"] for e in h], label=L["vacc"])
        ax[1].plot(ep, [e["val"]["auc"] for e in h], label=L["vauc"])
        ax[1].set_title(L["accauc"]); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=200); plt.close()

    # Grad-CAM examples: 3 correctly classified images per class
    cam = GradCAM(model, model.layer4)
    rng = np.random.default_rng(0)
    picks = []
    for c in (0, 1):
        idx = np.where((yt == c) & ((pt >= best_thr).astype(int) == c))[0]
        picks += list(rng.choice(idx, size=min(3, len(idx)), replace=False))
    fig, ax = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.4))
    for k, i in enumerate(picks):
        x, y = test_ds[i]
        heat, probs = cam(x.unsqueeze(0).to(device).requires_grad_(True), class_idx=1)
        gray = ((x[0].numpy() * STD[0] + MEAN[0]).clip(0, 1) * 255).astype(np.uint8)
        ax[0, k].imshow(gray, cmap="gray"); ax[0, k].set_title(f"{L['true']}: {L['cls'][y]}", fontsize=9)
        ax[1, k].imshow(overlay(gray, heat)); ax[1, k].set_title(f"P({L['pn']})={probs[1]:.2f}", fontsize=9)
        ax[0, k].axis("off"); ax[1, k].axis("off")
    plt.suptitle(L["gc"])
    plt.tight_layout(); plt.savefig(out / "gradcam_examples.png", dpi=200); plt.close()
    print(f"Figures saved to {out}/")


if __name__ == "__main__":
    main()
