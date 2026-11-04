"""Downloads the PneumoniaMNIST dataset (needs internet).
  python download_data.py --size 128
For the Kaggle version: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
download and unzip into data/chest_xray/.
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=128, choices=[28, 64, 128, 224])
a = ap.parse_args()
Path("data").mkdir(exist_ok=True)
from medmnist import PneumoniaMNIST
PneumoniaMNIST(split="train", download=True, size=a.size, root="data")
print(f"Ready: data/pneumoniamnist_{a.size}.npz")
