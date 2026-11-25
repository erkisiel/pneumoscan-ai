"""Prediction uncertainty: tells the physician when the model is NOT sure.

Combines two standard techniques, run on the same image:
  * Monte Carlo dropout (Gal & Ghahramani, 2016): the dropout layer before the classifier stays
    active at inference, so each forward pass samples a slightly different network.
  * Test-time augmentation (TTA): small rotations / shifts / contrast changes, the same
    perturbations used in training.
The spread (standard deviation) of the N predicted probabilities is the uncertainty.
A result is flagged LOW CONFIDENCE when the spread is large or when the mean probability lies
close to the decision threshold; such cases should always be read by a radiologist.
"""
import numpy as np
import torch
from torchvision import transforms

from data import MEAN, STD

STD_LIMIT = 0.08      # chosen on the validation set (see evaluate_uncertainty.py)
MARGIN = 0.10         # |p - threshold| below this is "borderline"


def _tta_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(8, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def _enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def predict_with_uncertainty(model, pil_img, img_size, n=20, seed=0):
    """Returns mean probability, std and the individual samples."""
    torch.manual_seed(seed)
    tf = _tta_transform(img_size)
    x = torch.stack([tf(pil_img) for _ in range(n)])
    model.eval(); _enable_dropout(model)
    try:
        p = torch.softmax(model(x), 1)[:, 1].numpy()
    finally:
        model.eval()
    return float(p.mean()), float(p.std()), p


def confidence_level(p_point, std, thr):
    """'high' or 'low' plus a short human-readable reason."""
    reasons = []
    if std > STD_LIMIT:
        reasons.append(f"predictions vary under small perturbations (±{std*100:.0f}%)")
    if abs(p_point - thr) < MARGIN:
        reasons.append("probability is close to the decision threshold")
    return ("low", "; ".join(reasons)) if reasons else ("high", "stable prediction, far from the threshold")
