"""
PyTorch Dataset Classes
=======================
Dataset classes for U-Net segmentation and BUS-XAINet classification.
Includes stratified patient-aware splitting and class-balanced sampling.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Tuple, Optional, List, Dict

from preprocessing.pipeline import BUSPreprocessor, get_preprocessor
from preprocessing.augmentation import SynchronizedAugmentor, get_augmentor


_PREPROCESS_CACHE = {}

def get_preprocessed_pair(image_path: str, mask_path: str, preprocessor: BUSPreprocessor, is_classification: bool = False, label: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    cache_key = (image_path, mask_path, preprocessor.target_size, is_classification, label)
    if cache_key in _PREPROCESS_CACHE:
        return _PREPROCESS_CACHE[cache_key]
        
    image = cv2.imread(image_path)
    if is_classification:
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not load {image_path}")
            image_pre = preprocessor.process_image(image)
            if label != 2:
                mask_pre = (image_pre > 0).astype(np.float32)
            else:
                mask_pre = np.zeros_like(image_pre)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image_pre = preprocessor.process_image(image)
            if label != 2:
                mask_pre = preprocessor.process_mask(mask)
            else:
                mask_pre = np.zeros_like(image_pre)
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Could not load {image_path} or {mask_path}")
        image_pre = preprocessor.process_image(image)
        mask_pre = preprocessor.process_mask(mask)
        
    _PREPROCESS_CACHE[cache_key] = (image_pre, mask_pre)
    return image_pre, mask_pre


class BUSSegmentationDataset(Dataset):
    """
    PyTorch Dataset for U-Net segmentation training.
    
    Args:
        image_dir: Path to images directory
        mask_dir: Path to masks directory
        file_list: List of filenames to include
        preprocessor: BUSPreprocessor instance
        augmentor: SynchronizedAugmentor instance (None for val/test)
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        file_list: List[str],
        preprocessor: BUSPreprocessor,
        augmentor: Optional[SynchronizedAugmentor] = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.preprocessor = preprocessor
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.file_list[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)
        
        image, mask = get_preprocessed_pair(img_path, mask_path, self.preprocessor, is_classification=False)

        # Augment
        if self.augmentor is not None:
            image, mask = self.augmentor(image, mask)

        # Convert to tensors: (1, H, W) for single-channel
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor


class BUSClassificationDataset(Dataset):
    """
    PyTorch Dataset for BUS-XAINet classification training.
    
    Args:
        data_dir: Path to classification dataset root (with class subdirs)
        file_list: List of (filename, class_label) tuples
        preprocessor: BUSPreprocessor instance
        augmentor: SynchronizedAugmentor instance (None for val/test)
    """

    CLASS_MAP = {"benign": 0, "malignant": 1, "normal": 2}
    CLASS_NAMES = ["benign", "malignant", "normal"]

    def __init__(
        self,
        data_dir: str,
        file_list: List[Tuple[str, int]],
        preprocessor: BUSPreprocessor,
        augmentor: Optional[SynchronizedAugmentor] = None,
    ):
        self.data_dir = data_dir
        self.file_list = file_list  # [(filepath, label), ...]
        self.preprocessor = preprocessor
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.file_list[idx]

        # Extract filename to load unmasked image and mask from unet_dataset
        fname = os.path.basename(filepath)
        unet_img_dir = r"e:\CONFERENCE\unet_dataset\images"
        unet_mask_dir = r"e:\CONFERENCE\unet_dataset\masks"
        
        if not os.path.exists(unet_img_dir):
            # Fallback to relative path
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))
            unet_img_dir = os.path.join(workspace_root, "unet_dataset", "images")
            unet_mask_dir = os.path.join(workspace_root, "unet_dataset", "masks")
            
        unet_img_path = os.path.join(unet_img_dir, fname)
        unet_mask_path = os.path.join(unet_mask_dir, fname)

        image_pre, mask_pre = get_preprocessed_pair(
            filepath if not os.path.exists(unet_img_path) else unet_img_path,
            unet_mask_path,
            self.preprocessor,
            is_classification=True,
            label=label
        )

        # Augment (apply spatial transforms to image and mask simultaneously)
        if self.augmentor is not None:
            image_pre, mask_pre = self.augmentor(image_pre, mask_pre)

        # Convert to tensors and stack along channel dimension: shape (2, H, W)
        image_tensor = torch.from_numpy(image_pre).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_pre).unsqueeze(0).float()
        stacked_tensor = torch.cat([image_tensor, mask_tensor], dim=0)

        return stacked_tensor, label


def get_segmentation_file_lists(
    image_dir: str, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split segmentation dataset into train/val/test by patient prefix.
    """
    all_files = sorted(os.listdir(image_dir))

    # Extract patient prefixes for patient-aware splitting
    # BUSI: busi_benign_(N) -> busi_benign
    # UCLM: uclm_XXXX_NNN -> uclm_XXXX
    # LCIS: lcis_ILC_N -> lcis_ILC
    def get_patient(fname):
        base = os.path.splitext(fname)[0]
        if base.startswith("uclm_"):
            parts = base.rsplit("_", 1)
            return parts[0] if len(parts) > 1 else base
        return base

    patients = list(set(get_patient(f) for f in all_files))
    patient_labels = [0] * len(patients)  # Dummy labels for stratification

    # Split patients (not images) to avoid data leakage
    np.random.seed(seed)
    train_patients, temp_patients = train_test_split(
        patients, test_size=val_ratio + test_ratio, random_state=seed
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=relative_test, random_state=seed
    )

    train_patients = set(train_patients)
    val_patients = set(val_patients)
    test_patients = set(test_patients)

    train_files = [f for f in all_files if get_patient(f) in train_patients]
    val_files = [f for f in all_files if get_patient(f) in val_patients]
    test_files = [f for f in all_files if get_patient(f) in test_patients]

    return train_files, val_files, test_files


def get_classification_file_lists(
    data_dir: str, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Build file lists with labels from classification dataset directory structure.
    Stratified split by class.
    """
    all_items = []
    labels = []

    for class_name, class_idx in BUSClassificationDataset.CLASS_MAP.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith(".png"):
                filepath = os.path.join(class_dir, fname)
                all_items.append((filepath, class_idx))
                labels.append(class_idx)

    labels = np.array(labels)

    # Stratified split
    indices = np.arange(len(all_items))
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_ratio + test_ratio, random_state=seed, stratify=labels
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        random_state=seed,
        stratify=labels[temp_idx],
    )

    train_list = [all_items[i] for i in train_idx]
    val_list = [all_items[i] for i in val_idx]
    test_list = [all_items[i] for i in test_idx]

    return train_list, val_list, test_list


def get_class_weights(file_list: List[Tuple[str, int]], num_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced training."""
    labels = [label for _, label in file_list]
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)


def get_weighted_sampler(file_list: List[Tuple[str, int]], num_classes: int = 3) -> WeightedRandomSampler:
    """Create WeightedRandomSampler for class-balanced batches."""
    labels = [label for _, label in file_list]
    counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def create_segmentation_dataloaders(
    image_dir: str,
    mask_dir: str,
    batch_size: int = 16,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    p_elastic: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for segmentation.
    """
    preprocessor = get_preprocessor("segmentation")
    train_files, val_files, test_files = get_segmentation_file_lists(
        image_dir, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    train_ds = BUSSegmentationDataset(
        image_dir, mask_dir, train_files, preprocessor, get_augmentor("train", p_elastic=p_elastic)
    )
    val_ds = BUSSegmentationDataset(
        image_dir, mask_dir, val_files, preprocessor, None
    )
    test_ds = BUSSegmentationDataset(
        image_dir, mask_dir, test_files, preprocessor, None
    )

    print(f"Segmentation split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        ),
    }


def create_classification_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    p_elastic: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for classification.
    """
    preprocessor = get_preprocessor("classification")
    train_list, val_list, test_list = get_classification_file_lists(
        data_dir, seed=seed
    )

    sampler = get_weighted_sampler(train_list)

    train_ds = BUSClassificationDataset(
        data_dir, train_list, preprocessor, get_augmentor("train", p_elastic=p_elastic)
    )
    val_ds = BUSClassificationDataset(data_dir, val_list, preprocessor, None)
    test_ds = BUSClassificationDataset(data_dir, test_list, preprocessor, None)

    print(f"Classification split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        ),
    }
