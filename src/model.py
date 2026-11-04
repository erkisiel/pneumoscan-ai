"""Model definition: ResNet18 adapted to single-channel (grayscale) chest X-rays."""
import torch
import torch.nn as nn
from torchvision import models


def build_model(pretrained: bool = False, num_classes: int = 2) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    # convert the 3-channel stem to 1 channel (pretrained weights are averaged)
    old = m.conv1
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        with torch.no_grad():
            m.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    return m


def load_model(ckpt_path: str, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    m = build_model(pretrained=False)
    m.load_state_dict(ckpt["model"])
    m.eval().to(device)
    return m, ckpt.get("img_size", 128), ckpt.get("threshold", 0.5)
