import os
import sys
import numpy as np
import torch
import cv2
from typing import Tuple, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import get_preprocessor
from models.bus_xainet import BUSXAINet
from xai.grad_cam import GradCAMPlusPlus
from xai.attention_viz import AttentionRollout


class MedicalInferencePipeline:
    """Production-grade wrapper for U-Net Segmentation & BUS-XAINet Classification"""

    def __init__(
        self,
        unet_path: str = "outputs/models/unet_production.pt",
        classifier_path: str = "outputs/models/bus_xainet_production.pt",
        python_checkpoint_path: str = "outputs/models/xainet_fold0.pth"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AI Pipeline] Initializing inference on: {self.device}")
        
        # Paths
        self.unet_path = unet_path
        self.classifier_path = classifier_path
        self.python_checkpoint_path = python_checkpoint_path

        # Load standard TorchScript compiled models
        if not os.path.exists(self.unet_path) or not os.path.exists(self.classifier_path):
            # Fallback path if run from subfolder
            self.unet_path = os.path.join("..", unet_path)
            self.classifier_path = os.path.join("..", classifier_path)
            self.python_checkpoint_path = os.path.join("..", python_checkpoint_path)

        if not os.path.exists(self.unet_path):
            raise FileNotFoundError(f"U-Net JIT model not found: {self.unet_path}")
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"Classifier JIT model not found: {self.classifier_path}")

        self.unet = torch.jit.load(self.unet_path, map_location=self.device)
        self.classifier = torch.jit.load(self.classifier_path, map_location=self.device)
        self.unet.eval()
        self.classifier.eval()

        self.seg_preprocessor = get_preprocessor("segmentation")
        self.clf_preprocessor = get_preprocessor("classification")
        self.classes = ["Benign", "Malignant", "Normal"]

    def run(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Runs U-Net segmentation, masks image, runs classifier, and generates XAI Grad-CAM overlay.
        Saves output mask and Grad-CAM image to output_dir.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # 1. Read Raw Image
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            raise ValueError(f"Could not read image: {image_path}")

        orig_h, orig_w = raw_img.shape[:2]

        # 2. Run Segmentation
        img_seg_pre = self.seg_preprocessor.process_image(raw_img)
        img_seg_tensor = torch.from_numpy(img_seg_pre).unsqueeze(0).unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            pred_mask_tensor = torch.sigmoid(self.unet(img_seg_tensor))
        pred_mask_np = pred_mask_tensor.squeeze().cpu().numpy()
        binary_mask_np = (pred_mask_np > 0.5).astype(np.uint8)

        # Clean mask morphologically
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(binary_mask_np, cv2.MORPH_OPEN, kernel)
        orig_mask = cv2.resize(cleaned_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 3. Check for Lesion
        has_lesion = bool(np.max(orig_mask) > 0)
        lesion_diameter_px = 0.0
        lesion_area_pct = 0.0

        if has_lesion:
            # Calculate metrics
            contours, _ = cv2.findContours(orig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                _, _, w, h = cv2.boundingRect(c)
                lesion_diameter_px = float(max(w, h))
                lesion_area_pct = float(cv2.contourArea(c) / (orig_h * orig_w) * 100.0)

        # 4. Classification
        img_clf_pre = self.clf_preprocessor.process_image(raw_img)
        if has_lesion:
            mask_clf_pre = self.clf_preprocessor.process_mask(orig_mask)
        else:
            mask_clf_pre = np.zeros_like(img_clf_pre)

        image_tensor = torch.from_numpy(img_clf_pre).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_clf_pre).unsqueeze(0).float()
        img_clf_tensor = torch.cat([image_tensor, mask_tensor], dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier(img_clf_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_class_idx = int(np.argmax(probs))
        pred_label = self.classes[pred_class_idx]
        confidence = float(probs[pred_class_idx])

        # Overwrite/Set logic: if no lesion was found by U-Net, the classifier might predict Normal.
        # But if U-Net found a major lesion, classifier should lean Benign/Malignant.
        # BI-RADS assessment mapping
        if pred_label == "Normal":
            birads = "BI-RADS 1"
        elif pred_label == "Benign":
            birads = "BI-RADS 2" if confidence > 0.8 else "BI-RADS 3"
        else:  # Malignant
            if confidence > 0.85:
                birads = "BI-RADS 5"
            elif confidence > 0.6:
                birads = "BI-RADS 4C"
            else:
                birads = "BI-RADS 4B"

        # 5. Explainable AI (XAI)
        # Load Python model weights for hooks
        py_classifier = BUSXAINet(num_classes=3, pretrained=False).to(self.device)
        if os.path.exists(self.python_checkpoint_path):
            checkpoint = torch.load(self.python_checkpoint_path, map_location=self.device)
            py_classifier.load_state_dict(checkpoint["model_state_dict"])
        py_classifier.eval()

        gcam = GradCAMPlusPlus(py_classifier)
        heatmap_gcam = gcam.generate(img_clf_tensor, target_class=pred_class_idx)
        # Use the raw BGR image for overlay - resize to match classifier input spatial size
        clf_size = img_clf_pre.shape[0]  # img_clf_pre is (H, W) normalized
        raw_for_overlay = cv2.resize(raw_img, (clf_size, clf_size))
        overlay_gcam = gcam.generate_overlay(raw_for_overlay, heatmap_gcam, alpha=0.45)
        # Resize overlay back to original image dimensions for display
        overlay_gcam = cv2.resize(overlay_gcam, (orig_w, orig_h))
        gcam.remove_hooks()

        # 6. Save Images
        os.makedirs(output_dir, exist_ok=True)
        filename_base = os.path.splitext(os.path.basename(image_path))[0]

        mask_save_name = f"{filename_base}_mask.png"
        mask_save_path = os.path.join(output_dir, mask_save_name)
        cv2.imwrite(mask_save_path, orig_mask * 255)

        xai_save_name = f"{filename_base}_xai.png"
        xai_save_path = os.path.join(output_dir, xai_save_name)
        cv2.imwrite(xai_save_path, overlay_gcam)

        return {
            "prediction": pred_label,
            "confidence": confidence,
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(3)},
            "birads_rating": birads,
            "segmentation": {
                "detected": has_lesion,
                "diameter_px": lesion_diameter_px,
                "area_pct": lesion_area_pct,
                "mask_filename": mask_save_name
            },
            "xai": {
                "xai_filename": xai_save_name
            }
        }
