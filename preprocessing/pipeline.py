"""
BUS Preprocessing Pipeline
==========================
6-step pipeline for Breast Ultrasound image preprocessing:
1. Grayscale conversion
2. Artifact removal (connected component analysis)
3. CLAHE contrast enhancement
4. Percentile normalization (P2-P98)
5. Aspect-preserving resize with zero-padding
6. Mask binarization
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class BUSPreprocessor:
    """
    Breast Ultrasound preprocessing pipeline.
    
    Args:
        target_size: Output size as (H, W). Default (256, 256) for U-Net.
        clahe_clip: CLAHE clip limit. Default 2.0.
        clahe_grid: CLAHE tile grid size. Default (8, 8).
        p_low: Lower percentile for intensity clipping. Default 2.
        p_high: Upper percentile for intensity clipping. Default 98.
        remove_artifacts: Whether to remove text/label artifacts. Default True.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (8, 8),
        p_low: float = 2.0,
        p_high: float = 98.0,
        remove_artifacts: bool = True,
    ):
        self.target_size = target_size
        # Store CLAHE params instead of the cv2.CLAHE object itself — cv2.CLAHE
        # is a C++ binding and cannot be pickled by Windows multiprocessing spawn.
        self._clahe_clip = clahe_clip
        self._clahe_grid = clahe_grid
        self.p_low = p_low
        self.p_high = p_high
        self.remove_artifacts = remove_artifacts

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def remove_artifacts_cc(self, image: np.ndarray) -> np.ndarray:
        """
        Remove small bright artifacts (text labels, markers) using
        connected component analysis. Keeps only the largest bright region.
        """
        # Threshold to find bright regions
        _, binary = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels <= 1:
            return image

        # Remove small bright components (likely text/labels)
        result = image.copy()
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Small bright blobs are likely artifacts
            if area < 500:
                result[labels == i] = 0

        return result

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for local contrast enhancement.
        
        NOTE: cv2.CLAHE is recreated per-call because it cannot be pickled
        (C++ binding) — required for multiprocessing DataLoader workers.
        """
        clahe = cv2.createCLAHE(clipLimit=self._clahe_clip, tileGridSize=self._clahe_grid)
        return clahe.apply(image)

    def percentile_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Percentile-based intensity normalization.
        Clips to [P2, P98] then scales to [0, 1] float32.
        """
        p_lo = np.percentile(image, self.p_low)
        p_hi = np.percentile(image, self.p_high)

        if p_hi - p_lo < 1e-6:
            return np.zeros_like(image, dtype=np.float32)

        clipped = np.clip(image.astype(np.float32), p_lo, p_hi)
        normalized = (clipped - p_lo) / (p_hi - p_lo)
        return normalized

    def resize_with_padding(
        self, image: np.ndarray, is_mask: bool = False
    ) -> np.ndarray:
        """
        Resize image preserving aspect ratio, padding with zeros.
        Uses INTER_NEAREST for masks, INTER_LINEAR for images.
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size

        # Compute scale factor
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

        # Create canvas with zero padding
        if len(image.shape) == 3:
            canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=resized.dtype)
        else:
            canvas = np.zeros((target_h, target_w), dtype=resized.dtype)

        # Center the resized image
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        return canvas

    def binarize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Binarize mask to {0, 1}. Handles both BUSI (0/255) and
        UCLM (0/149) value ranges.
        """
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # Any non-zero pixel becomes 1
        binary = (mask > 0).astype(np.float32)
        return binary

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for a single image.
        Returns float32 array in [0, 1] with shape (target_h, target_w).
        """
        # Step 1: Grayscale
        gray = self.to_grayscale(image)

        # Step 2: Artifact removal
        if self.remove_artifacts:
            gray = self.remove_artifacts_cc(gray)

        # Step 3: CLAHE
        gray = self.apply_clahe(gray)

        # Step 4: Percentile normalization
        normalized = self.percentile_normalize(gray)

        # Step 5: Resize with padding
        resized = self.resize_with_padding(normalized, is_mask=False)

        return resized

    def process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for a mask.
        Returns float32 binary array {0, 1} with shape (target_h, target_w).
        """
        # Binarize
        binary = self.binarize_mask(mask)

        # Resize with padding (nearest neighbor)
        resized = self.resize_with_padding(binary, is_mask=True)

        return resized

    def process_pair(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process an image-mask pair together."""
        return self.process_image(image), self.process_mask(mask)


def get_preprocessor(task: str = "segmentation") -> BUSPreprocessor:
    """
    Factory function to get preprocessor with appropriate settings.
    
    Args:
        task: 'segmentation' (256x256) or 'classification' (224x224)
    """
    if task == "segmentation":
        return BUSPreprocessor(target_size=(256, 256))
    elif task == "classification":
        return BUSPreprocessor(target_size=(224, 224))
    else:
        raise ValueError(f"Unknown task: {task}. Use 'segmentation' or 'classification'.")
