"""
Data Augmentation Transforms
============================
Synchronized geometric + intensity augmentations for BUS images.
All spatial transforms are applied identically to image AND mask.
"""

import torch
import numpy as np
import cv2
import random
from typing import Tuple, Optional


class ElasticDeformation:
    """Grid-based elastic deformation for realistic tissue distortion."""

    def __init__(self, alpha: float = 30.0, sigma: float = 4.0, p: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask

        h, w = image.shape[:2]
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32),
            (0, 0),
            self.sigma,
        ) * self.alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32),
            (0, 0),
            self.sigma,
        ) * self.alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        image_out = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_out = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        return image_out, mask_out


class SynchronizedAugmentor:
    """
    Applies synchronized augmentations to image-mask pairs.
    
    Geometric transforms are applied identically to both.
    Intensity transforms are applied only to the image.
    
    Args:
        p_flip: Probability of horizontal flip
        rotation_range: Max rotation angle in degrees
        scale_range: (min_scale, max_scale) tuple
        shear_range: Max shear angle in degrees
        brightness_range: Max brightness jitter
        contrast_range: Max contrast jitter
        noise_std: Standard deviation of Gaussian noise
        gamma_range: (min_gamma, max_gamma) tuple
        elastic_alpha: Elastic deformation alpha
        elastic_sigma: Elastic deformation sigma
        p_elastic: Probability of elastic deformation
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        rotation_range: float = 10.0,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        shear_range: float = 5.0,
        brightness_range: float = 0.15,
        contrast_range: float = 0.15,
        noise_std: float = 0.01,
        gamma_range: Tuple[float, float] = (0.85, 1.15),
        elastic_alpha: float = 15.0,
        elastic_sigma: float = 4.0,
        p_elastic: float = 0.15,
    ):
        self.p_flip = p_flip
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.gamma_range = gamma_range
        self.elastic = ElasticDeformation(elastic_alpha, elastic_sigma, p_elastic)

    def _random_flip(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Horizontal flip."""
        if random.random() < self.p_flip:
            return np.fliplr(image).copy(), np.fliplr(mask).copy()
        return image, mask

    def _random_affine(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random rotation, scale, and shear."""
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        angle = random.uniform(-self.rotation_range, self.rotation_range)
        scale = random.uniform(*self.scale_range)

        M = cv2.getRotationMatrix2D(center, angle, scale)

        # Add shear
        shear = random.uniform(-self.shear_range, self.shear_range)
        shear_rad = np.deg2rad(shear)
        shear_M = np.array(
            [[1, np.tan(shear_rad), 0], [0, 1, 0]], dtype=np.float64
        )
        M = M + shear_M * 0.1  # Subtle shear

        image_out = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        mask_out = cv2.warpAffine(
            mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )

        return image_out, mask_out

    def _brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Random brightness and contrast jitter (image only)."""
        brightness = random.uniform(-self.brightness_range, self.brightness_range)
        contrast = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)

        image = image * contrast + brightness
        return np.clip(image, 0, 1)

    def _gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise (image only)."""
        noise = np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
        return np.clip(image + noise, 0, 1)

    def _gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Random gamma correction (image only)."""
        gamma = random.uniform(*self.gamma_range)
        return np.power(np.clip(image, 1e-8, 1.0), gamma).astype(np.float32)

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all augmentations to an image-mask pair.
        
        Args:
            image: Float32 array in [0, 1]
            mask: Float32 binary array {0, 1}
        
        Returns:
            Augmented (image, mask) pair
        """
        # Geometric (synchronized)
        image, mask = self._random_flip(image, mask)
        image, mask = self._random_affine(image, mask)
        image, mask = self.elastic(image, mask)

        # Intensity (image only)
        image = self._brightness_contrast(image)
        image = self._gaussian_noise(image)
        image = self._gamma_correction(image)

        # Re-binarize mask after transforms
        mask = (mask > 0.5).astype(np.float32)

        return image, mask


def get_augmentor(mode: str = "train", p_elastic: Optional[float] = None) -> Optional[SynchronizedAugmentor]:
    """
    Factory function for augmentor.
    
    Args:
        mode: 'train' returns full augmentor, 'val'/'test' returns None
        p_elastic: Custom probability for elastic deformation
    """
    if mode == "train":
        if p_elastic is not None:
            return SynchronizedAugmentor(p_elastic=p_elastic)
        return SynchronizedAugmentor()
    return None
