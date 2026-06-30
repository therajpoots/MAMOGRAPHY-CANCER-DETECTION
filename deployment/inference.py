"""
Standalone Medical Inference and Explainability Pipeline
=========================================================
Runs end-to-end inference on raw breast ultrasound images:
1. Preprocess raw image
2. Segment lesion region using Attention U-Net
3. Mask image using predicted lesion mask
4. Classify masked lesion (Benign / Malignant / Normal) using BUS-XAINet
5. Generate XAI explanations (Grad-CAM and Attention Rollout)
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import get_preprocessor
from models.unet import AttentionUNet
from models.bus_xainet import BUSXAINet
from xai.grad_cam import GradCAMPlusPlus
from xai.attention_viz import AttentionRollout, plot_side_by_side


class BUSInferencePipeline:
    """End-to-end inference pipeline for Breast Ultrasound lesion classification."""

    def __init__(
        self,
        unet_model_path: str = "outputs/models/unet_production.pt",
        classifier_model_path: str = "outputs/models/bus_xainet_production.pt",
        device: str = "auto"
    ):
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Initializing inference on: {self.device}")
        
        # Load TorchScript compiled models
        if not os.path.exists(unet_model_path):
            raise FileNotFoundError(f"U-Net model file not found: {unet_model_path}")
        if not os.path.exists(classifier_model_path):
            raise FileNotFoundError(f"Classifier model file not found: {classifier_model_path}")
            
        # Load compiled TorchScript models
        self.unet = torch.jit.load(unet_model_path, map_location=self.device)
        self.classifier = torch.jit.load(classifier_model_path, map_location=self.device)
        
        self.unet.eval()
        self.classifier.eval()
        
        # Preprocessors
        self.seg_preprocessor = get_preprocessor("segmentation")
        self.clf_preprocessor = get_preprocessor("classification")
        
        # Label mapping
        self.classes = ["Benign", "Malignant", "Normal"]

    def run_inference(self, image_path: str) -> Dict[str, Any]:
        """
        Runs the full dual-stage inference pipeline on a single image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Read raw image
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Keep original H, W
        orig_h, orig_w = raw_img.shape[:2]
        
        # --- Stage 1: Segmentation ---
        # Preprocess for segmentation (256x256)
        img_seg_pre = self.seg_preprocessor.process_image(raw_img)
        img_seg_tensor = torch.from_numpy(img_seg_pre).unsqueeze(0).unsqueeze(0).to(self.device).float()
        
        # Segment lesion
        with torch.no_grad():
            pred_mask_tensor = torch.sigmoid(self.unet(img_seg_tensor))
            
        pred_mask_np = pred_mask_tensor.squeeze().cpu().numpy()
        binary_mask_np = (pred_mask_np > 0.5).astype(np.uint8)
        
        # Post-process mask (morphological cleaning)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(binary_mask_np, cv2.MORPH_OPEN, kernel)
        
        # Resize mask back to original resolution
        orig_mask = cv2.resize(cleaned_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # --- Stage 2: Masking & Classification ---
        # Preprocess the unmasked raw image first (prevents percentile normalization and CLAHE distortion)
        img_clf_pre = self.clf_preprocessor.process_image(raw_img)
        
        # Check if lesion is detected. If normal case, U-Net might output all zeros.
        has_lesion = np.max(orig_mask) > 0
        
        # Apply mask to the preprocessed image
        if has_lesion:
            mask_clf_pre = self.clf_preprocessor.process_mask(orig_mask)
            masked_img = raw_img.copy()
            masked_img[orig_mask == 0] = 0
        else:
            mask_clf_pre = np.zeros_like(img_clf_pre)
            # Keep unmasked
            masked_img = raw_img.copy()
            
        # Convert to tensors and stack along channel dimension: shape (1, 2, H, W)
        image_tensor = torch.from_numpy(img_clf_pre).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_clf_pre).unsqueeze(0).float()
        img_clf_tensor = torch.cat([image_tensor, mask_tensor], dim=0).unsqueeze(0).to(self.device)
        
        # Classify lesion
        with torch.no_grad():
            logits = self.classifier(img_clf_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        pred_class_idx = int(np.argmax(probs))
        pred_label = self.classes[pred_class_idx]
        confidence = float(probs[pred_class_idx])
        
        return {
            "prediction": pred_label,
            "confidence": confidence,
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(3)},
            "has_lesion": has_lesion,
            "original_mask": orig_mask,
            "masked_image": masked_img,
            "clf_input_tensor": img_clf_tensor,
            "clf_input_image": img_clf_pre
        }

    def generate_explanations(
        self,
        clf_input_tensor: torch.Tensor,
        clf_input_image: np.ndarray,
        unet_mask: np.ndarray,
        save_path: str,
        target_class: Optional[int] = None
    ):
        """
        Generates explainability maps using Python models (hooks can't run on traced TorchScript).
        """
        print("Generating XAI heatmaps using Python model definitions...")
        
        # Instantiate python models for hook registration
        py_classifier = BUSXAINet(num_classes=3, pretrained=False).to(self.device)
        
        # JIT modules contain weights in identical layout, load state dict from python model if possible
        # Or load model directly (since this is run on same workspace, we load default state dict if exists)
        # For this demo/inference script, we copy state dict from self.classifier if compatible,
        # or state_dict directly since we exported it.
        try:
            py_classifier.load_state_dict(self.classifier.state_dict())
        except Exception:
            # If traced model lacks standard state dict mapping in Python, we do a parameter transfer:
            for name, param in py_classifier.named_parameters():
                try:
                    # Map traced attributes back to python weights
                    traced_param = getattr(self.classifier, name)
                    param.data.copy_(traced_param.data)
                except Exception:
                    pass
                    
        py_classifier.eval()
        
        # 1. Grad-CAM++
        gcam = GradCAMPlusPlus(py_classifier)
        heatmap_gcam = gcam.generate(clf_input_tensor, target_class=target_class)
        overlay_gcam = gcam.generate_overlay(clf_input_image, heatmap_gcam, alpha=0.4)
        gcam.remove_hooks()
        
        # 2. Attention Rollout
        rollout = AttentionRollout(py_classifier, discard_ratio=0.85)
        heatmap_roll = rollout.generate(clf_input_tensor)
        overlay_roll = rollout.generate_overlay(clf_input_image, heatmap_roll, alpha=0.4)
        
        # Convert classification input (grayscale float) to uint8 BGR for saving
        input_uint8 = (clf_input_image * 255).astype(np.uint8)
        input_bgr = cv2.cvtColor(input_uint8, cv2.COLOR_GRAY2BGR)
        
        # Plot and save
        plot_side_by_side(
            original_img=input_bgr,
            grad_cam_img=overlay_gcam,
            rollout_img=overlay_roll,
            title="BUS-XAINet Decision Explanations",
            save_path=save_path
        )
        print(f"[Saved] XAI explanation plot -> {save_path}")


def run_pipeline_demo():
    print("=== BUS-XAINet Inference Pipeline Demo ===")
    
    # Check if we have outputs/models directory and model files
    unet_path = "outputs/models/unet_production.pt"
    clf_path = "outputs/models/bus_xainet_production.pt"
    
    if not os.path.exists(unet_path) or not os.path.exists(clf_path):
        print("[ERROR] Production model files are missing. Please run export_model.py first.")
        return
        
    pipeline = BUSInferencePipeline(unet_model_path=unet_path, classifier_model_path=clf_path)
    
    # Pick a random sample image from dataset to test
    sample_dir = r"e:\CONFERENCE\unet_dataset\images"
    if not os.path.exists(sample_dir):
        print(f"[ERROR] Sample images directory {sample_dir} not found.")
        return
        
    samples = os.listdir(sample_dir)
    if not samples:
        print("[ERROR] No sample images found to run inference on.")
        return
        
    test_img_path = os.path.join(sample_dir, samples[0])
    print(f"Running end-to-end inference on sample image: {test_img_path}")
    
    results = pipeline.run_inference(test_img_path)
    
    print("\n--- Diagnostic Report ---")
    print(f"Prediction class: {results['prediction']}")
    print(f"Confidence score: {results['confidence']*100:.2f}%")
    print("Probabilities:")
    for label, prob in results["probabilities"].items():
        print(f"  - {label}: {prob*100:.2f}%")
    print(f"Lesion detected: {results['has_lesion']}")
    print("-------------------------\n")
    
    # Save segmentation mask output
    os.makedirs("outputs/xai_maps", exist_ok=True)
    mask_save_path = "outputs/xai_maps/sample_segmentation_mask.png"
    cv2.imwrite(mask_save_path, results["original_mask"])
    print(f"[Saved] Predicted mask -> {mask_save_path}")
    
    # Generate and save XAI explainability maps
    xai_save_path = "outputs/xai_maps/sample_xai_explanation.png"
    pipeline.generate_explanations(
        clf_input_tensor=results["clf_input_tensor"],
        clf_input_image=results["clf_input_image"],
        unet_mask=results["original_mask"],
        save_path=xai_save_path
    )
    
    print("=== Demo Run Completed successfully! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BUS-XAINet inference on ultrasound images.")
    parser.add_argument("--image", type=str, help="Path to input ultrasound image")
    parser.add_argument("--output", type=str, default="outputs/xai_maps/explanation.png", help="Path to save explainability plot")
    
    # If no args, run demo mode
    if len(sys.argv) == 1:
        run_pipeline_demo()
    else:
        args = parser.parse_args()
        if args.image:
            pipeline = BUSInferencePipeline()
            res = pipeline.run_inference(args.image)
            print(f"Prediction: {res['prediction']} ({res['confidence']*100:.2f}%)")
            pipeline.generate_explanations(res["clf_input_tensor"], res["clf_input_image"], res["original_mask"], args.output)
