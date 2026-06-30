"""
Transformer Attention Rollout and Visualization
===============================================
Extracts and computes attention rollout from Transformer encoder layers
to explain the model's global context focus.
"""

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class AttentionRollout:
    """
    Computes attention rollout for Vision Transformers / CNN-Transformer hybrids.
    Propagates attention weights across layers to compute a final activation map.
    """

    def __init__(self, model, head_fusion: str = "mean", discard_ratio: float = 0.9):
        """
        Args:
            model: Trained BUS-XAINet model
            head_fusion: How to fuse attention heads ('mean', 'max', or 'min')
            discard_ratio: Ratio of lowest attention values to discard (for noise reduction)
        """
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def compute_rollout(self, attention_maps: List[torch.Tensor]) -> np.ndarray:
        """
        Compute attention rollout from stored attention matrices.
        
        Args:
            attention_maps: List of attention tensors, each of shape (B, num_heads, N, N)
                            where N is the sequence length (e.g., 50: 1 CLS + 49 patches)
                            
        Returns:
            Rollout attention heatmap (H, W) corresponding to CLS token attention to patches.
        """
        # We assume batch size = 1 for visualization
        # Remove batch dim: (num_heads, N, N)
        attens = [a[0].cpu().numpy() for a in attention_maps]
        
        num_layers = len(attens)
        N = attens[0].shape[1]  # Sequence length (e.g. 50)
        
        # Initialize rollout matrix as identity
        result = np.eye(N)
        
        for attention in attens:
            # Fuse attention heads
            if self.head_fusion == "mean":
                a_fused = np.mean(attention, axis=0)
            elif self.head_fusion == "max":
                a_fused = np.max(attention, axis=0)
            elif self.head_fusion == "min":
                a_fused = np.min(attention, axis=0)
            else:
                raise ValueError(f"Unknown head fusion: {self.head_fusion}")
            
            # Account for residual connections: A = 0.5 * A + 0.5 * I
            I = np.eye(N)
            a_fused = 0.5 * a_fused + 0.5 * I
            
            # Re-normalize rows to sum to 1
            row_sums = a_fused.sum(axis=-1, keepdims=True)
            a_fused = a_fused / (row_sums + 1e-8)
            
            # Rollout multiplication: R = A * R
            result = np.matmul(a_fused, result)
            
        # CLS token attention to all other tokens
        # CLS token is at index 0, patches are at 1..N-1
        cls_attn = result[0, 1:]
        
        # Discard low values (noise reduction)
        if self.discard_ratio > 0:
            flat = cls_attn.flatten()
            indices = np.argsort(flat)
            limit = int(len(indices) * self.discard_ratio)
            threshold = flat[indices[limit]]
            cls_attn[cls_attn < threshold] = 0
            
        # Reshape to spatial dimensions
        grid_size = int(np.sqrt(len(cls_attn)))
        if grid_size * grid_size != len(cls_attn):
            raise ValueError(f"Sequence length minus CLS ({len(cls_attn)}) is not a perfect square.")
            
        heatmap = cls_attn.reshape((grid_size, grid_size))
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        return heatmap

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Run forward pass, extract attention weights, and compute rollout.
        
        Args:
            input_tensor: Input image tensor of shape (1, 1, H, W)
            
        Returns:
            Rollout attention map of shape (grid_size, grid_size)
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        attention_maps = self.model.get_attention_maps()
        if not attention_maps:
            raise ValueError("No attention maps were stored. Make sure model.forward() was run and model stores attention weights.")
            
        return self.compute_rollout(attention_maps)

    def generate_overlay(
        self,
        input_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay attention heatmap on original image."""
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
        
        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (W, H))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        
        overlay = cv2.addWeighted(input_image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay


def plot_side_by_side(
    original_img: np.ndarray,
    grad_cam_img: np.ndarray,
    rollout_img: np.ndarray,
    title: str = "Explainable AI (XAI) Comparison",
    save_path: Optional[str] = None,
):
    """
    Plots original image, Grad-CAM overlay, and Attention Rollout overlay side by side.
    
    Args:
        original_img: Original image (H, W, 3) or (H, W)
        grad_cam_img: Grad-CAM overlay image (H, W, 3)
        rollout_img: Attention Rollout overlay image (H, W, 3)
        title: Title of the figure
        save_path: Optional path to save the plot
    """
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
    grad_cam_img = cv2.cvtColor(grad_cam_img, cv2.COLOR_BGR2RGB)
    rollout_img = cv2.cvtColor(rollout_img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#1e1e1e")
    fig.suptitle(title, color="white", fontsize=16, weight="bold")
    
    images = [original_img, grad_cam_img, rollout_img]
    subtitles = ["Original BUS Image", "Grad-CAM++ (Local Features)", "Attention Rollout (Global Context)"]
    
    for ax, img, sub in zip(axes, images, subtitles):
        ax.imshow(img)
        ax.set_title(sub, color="white", fontsize=12, pad=10)
        ax.axis("off")
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300)
        plt.close()
    else:
        plt.show()
