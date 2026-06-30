"""
Integrated Gradients and Occlusion Sensitivity XAI Modules
===========================================================
Provides pixel-level attribution explanations for BUS-XAINet:
1. Integrated Gradients (IG) — path integral of gradients from baseline to input.
2. Occlusion Sensitivity — systematic masking of image patches to measure prediction drop.
"""

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class IntegratedGradients:
    """
    Computes Integrated Gradients attribution maps.
    Reference: https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model):
        self.model = model

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        steps: int = 50,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Generate Integrated Gradients attribution map.
        
        Args:
            input_tensor: (1, 1, H, W) input image tensor
            target_class: Class index for attribution (None = predicted class)
            steps: Number of steps for Riemann sum approximation
            baseline: Baseline tensor of same shape as input_tensor (None = zero tensor)
            
        Returns:
            Attribution map as numpy array (H, W)
        """
        self.model.eval()
        device = input_tensor.device
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(device)
            
        # Run forward pass to get predictions
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
                
        # Generate scaled inputs along the path: baseline + alpha * (input - baseline)
        alphas = torch.linspace(0.0, 1.0, steps=steps).to(device)
        delta = input_tensor - baseline
        
        # We will accumulate gradients
        grads_accumulated = torch.zeros_like(input_tensor).to(device)
        
        for alpha in alphas:
            # Scale input
            scaled_input = baseline + alpha * delta
            scaled_input.requires_grad_()
            
            # Forward pass
            out = self.model(scaled_input)
            
            # Calculate gradient of target class score with respect to input
            self.model.zero_grad()
            score = out[0, target_class]
            score.backward()
            
            # Accumulate gradient
            grads_accumulated += scaled_input.grad.detach()
            
        # Average the gradients and multiply by delta
        avg_grads = grads_accumulated / steps
        integrated_grad = delta * avg_grads
        
        # Convert to numpy and sum across channel dimension (which is 1 anyway)
        ig_map = integrated_grad.squeeze().cpu().numpy()
        
        # Take the absolute value or positive attributions
        ig_map = np.abs(ig_map)
        
        # Normalize
        if ig_map.max() > 0:
            ig_map = ig_map / ig_map.max()
            
        return ig_map


class OcclusionSensitivity:
    """
    Computes Occlusion Sensitivity attribution maps.
    Slides a grey/black patch over the image and measures change in target class probability.
    """

    def __init__(self, model, patch_size: int = 16, stride: int = 8):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        occlude_value: float = 0.0,
    ) -> np.ndarray:
        """
        Generate Occlusion Sensitivity map.
        
        Args:
            input_tensor: (1, 1, H, W) input image tensor
            target_class: Class index for attribution (None = predicted class)
            occlude_value: Value to pad the occluded region (default 0.0)
            
        Returns:
            Occlusion sensitivity map as numpy array (H, W)
        """
        self.model.eval()
        device = input_tensor.device
        
        _, _, H, W = input_tensor.shape
        
        # Forward pass to get original probabilities
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
                
        original_prob = probs[0, target_class].item()
        
        # Initialize sensitivity map
        sensitivity_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        
        # Slide window
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                # Clone tensor
                occluded_tensor = input_tensor.clone()
                occluded_tensor[0, 0, y : y + self.patch_size, x : x + self.patch_size] = occlude_value
                
                # Predict
                with torch.no_grad():
                    out = self.model(occluded_tensor)
                    out_probs = torch.softmax(out, dim=1)
                    occluded_prob = out_probs[0, target_class].item()
                    
                # Sensitivity is the drop in probability
                # Large drop means this region is highly important
                drop = original_prob - occluded_prob
                
                sensitivity_map[y : y + self.patch_size, x : x + self.patch_size] += drop
                count_map[y : y + self.patch_size, x : x + self.patch_size] += 1.0
                
        # Handle borders that might not have been fully covered
        non_zero = count_map > 0
        sensitivity_map[non_zero] /= count_map[non_zero]
        
        # Focus on positive drops (regions whose removal decreased probability of the target class)
        sensitivity_map = np.clip(sensitivity_map, 0, None)
        
        # Normalize
        if sensitivity_map.max() > 0:
            sensitivity_map = sensitivity_map / sensitivity_map.max()
            
        return sensitivity_map


def plot_attributions(
    original_img: np.ndarray,
    ig_map: np.ndarray,
    occlusion_map: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot Integrated Gradients and Occlusion attribution maps side by side."""
    if len(original_img.shape) == 2:
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#1e1e1e")
    fig.suptitle("Pixel-Level Attribution Explanations", color="white", fontsize=16, weight="bold")
    
    # 1. Original
    axes[0].imshow(original_img_rgb)
    axes[0].set_title("Original Image", color="white", fontsize=12)
    axes[0].axis("off")
    
    # 2. Integrated Gradients
    axes[1].imshow(original_img_rgb)
    # Overlay heatmap
    ig_resized = cv2.resize(ig_map, (original_img_rgb.shape[1], original_img_rgb.shape[0]))
    axes[1].imshow(ig_resized, cmap="hot", alpha=0.5)
    axes[1].set_title("Integrated Gradients (Pixel-level)", color="white", fontsize=12)
    axes[1].axis("off")
    
    # 3. Occlusion Sensitivity
    axes[2].imshow(original_img_rgb)
    occ_resized = cv2.resize(occlusion_map, (original_img_rgb.shape[1], original_img_rgb.shape[0]))
    axes[2].imshow(occ_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("Occlusion Sensitivity (Patch-level)", color="white", fontsize=12)
    axes[2].axis("off")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300)
        plt.close()
    else:
        plt.show()
