"""Grad-CAM: shows which image regions drive the model's decision.
Selvaraju ve ark., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, gout):
        self.grads = gout[0].detach()

    def __call__(self, x, class_idx=1):
        """x: (1,1,H,W) tensor. Returns an (H,W) heatmap in [0,1] and class probabilities."""
        self.model.zero_grad()
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)        # channel importance weights
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, probs


def overlay(gray_img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlays the heatmap on a grayscale image (H,W, 0-255); returns RGB uint8."""
    import matplotlib.cm as cm
    heat = cm.jet(cam)[..., :3]
    base = np.repeat((gray_img / 255.0)[..., None], 3, axis=2)
    return (np.clip((1 - alpha) * base + alpha * heat, 0, 1) * 255).astype(np.uint8)
