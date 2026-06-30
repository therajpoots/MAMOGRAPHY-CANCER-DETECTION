"""
Grad-CAM and Grad-CAM++ for BUS-XAINet
=======================================
Generates class activation heatmaps from CNN feature maps.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    
    Works with BUS-XAINet by hooking into the CNN feature maps.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []

        # Hook target layer (default: CNN projection layer)
        if target_layer is None:
            target_layer = model.cnn_proj
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def generate(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: (1, 1, H, W) input image tensor
            target_class: Class index for gradient computation (None = predicted class)
        
        Returns:
            Heatmap as numpy array (H, W) in [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if len(cam.shape) == 2 and cam.shape[0] > 2 and cam.shape[1] > 2:
            cam[0, :] = 0
            cam[-1, :] = 0
            cam[:, 0] = 0
            cam[:, -1] = 0

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def generate_overlay(
        self,
        input_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            input_image: Original image (H, W) or (H, W, 3) in [0, 255] uint8
            heatmap: Grad-CAM heatmap (h, w) in [0, 1]
            alpha: Blend factor
        
        Returns:
            Overlay image (H, W, 3) uint8
        """
        # Ensure input_image is uint8 and in [0, 255]
        if np.issubdtype(input_image.dtype, np.floating):
            if input_image.max() <= 1.01:
                input_image = (input_image * 255).astype(np.uint8)
            else:
                input_image = input_image.astype(np.uint8)
        elif input_image.dtype != np.uint8:
            input_image = input_image.astype(np.uint8)

        if len(input_image.shape) == 2:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

        H, W = input_image.shape[:2]

        # Resize heatmap to input size
        heatmap_resized = cv2.resize(heatmap, (W, H))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

        overlay = cv2.addWeighted(input_image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++: Improved version with pixel-level gradient weighting."""

    def generate(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients  # (1, C, H, W)
        acts = self.activations

        # Grad-CAM++ weighting
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        alpha = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + 1e-8)
        alpha = alpha * F.relu(grads)

        weights = alpha.sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        
        if len(cam.shape) == 2 and cam.shape[0] > 2 and cam.shape[1] > 2:
            cam[0, :] = 0
            cam[-1, :] = 0
            cam[:, 0] = 0
            cam[:, -1] = 0

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam
