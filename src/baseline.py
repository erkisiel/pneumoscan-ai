"""Baseline: logistic regression on downsampled (32x32) raw pixels.
Used in the report to show how much the CNN adds over a classical model.
  python src/baseline.py --data data/pneumoniamnist_128.npz
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="results")
a = ap.parse_args()
d = np.load(a.data)

def feats(x):
    return np.stack([np.asarray(Image.fromarray(i).resize((32, 32)), dtype=np.float32).ravel() / 255 for i in x])

Xtr, ytr = feats(d["train_images"]), d["train_labels"].ravel()
Xte, yte = feats(d["test_images"]), d["test_labels"].ravel()
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
clf.fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
res = {"auc": float(roc_auc_score(yte, p)), "accuracy": float(accuracy_score(yte, pred)),
       "sensitivity": float(recall_score(yte, pred)), "specificity": float(recall_score(yte, pred, pos_label=0))}
Path(a.out).mkdir(exist_ok=True); (Path(a.out) / "baseline_metrics.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
