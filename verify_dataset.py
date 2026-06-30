import os
import cv2
import numpy as np
import random

UNET_IMG_DIR = r"e:\CONFERENCE\unet_dataset\images"
UNET_MASK_DIR = r"e:\CONFERENCE\unet_dataset\masks"
CLASS_DIR = r"e:\CONFERENCE\classification_dataset"

def verify():
    print("=== Starting Dataset Verification ===")
    
    # 1. Check directory existences
    assert os.path.isdir(UNET_IMG_DIR), "UNet images directory missing!"
    assert os.path.isdir(UNET_MASK_DIR), "UNet masks directory missing!"
    assert os.path.isdir(CLASS_DIR), "Classification directory missing!"
    
    unet_images = os.listdir(UNET_IMG_DIR)
    unet_masks = os.listdir(UNET_MASK_DIR)
    
    # 2. Assert counts match
    assert len(unet_images) == 1533, f"Expected 1533 UNet images, got {len(unet_images)}"
    assert len(unet_masks) == 1533, f"Expected 1533 UNet masks, got {len(unet_masks)}"
    
    # Assert filenames are identical
    assert set(unet_images) == set(unet_masks), "Mismatched filenames between UNet images and masks!"
    print("[OK] UNet dataset: File counts and filenames match perfectly (1533 files).")
    
    # 3. Assert classification counts match
    benign_class = os.listdir(os.path.join(CLASS_DIR, 'benign'))
    malign_class = os.listdir(os.path.join(CLASS_DIR, 'malignant'))
    normal_class = os.listdir(os.path.join(CLASS_DIR, 'normal'))
    
    assert len(benign_class) == 681, f"Expected 681 benign images, got {len(benign_class)}"
    assert len(malign_class) == 300, f"Expected 300 malignant images, got {len(malign_class)}"
    assert len(normal_class) == 552, f"Expected 552 normal images, got {len(normal_class)}"
    
    total_class_images = len(benign_class) + len(malign_class) + len(normal_class)
    assert total_class_images == 1533, f"Total classification files should be 1533, got {total_class_images}"
    print("[OK] Classification dataset: File counts match perfectly (benign: 681, malignant: 300, normal: 552).")
    
    # 4. Check random sample masking correctness
    print("\nVerifying random sample masking logic...")
    
    # Pick a random benign image
    sample_name = random.choice(benign_class)
    img_path = os.path.join(UNET_IMG_DIR, sample_name)
    mask_path = os.path.join(UNET_MASK_DIR, sample_name)
    class_path = os.path.join(CLASS_DIR, 'benign', sample_name)
    
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    masked_img = cv2.imread(class_path)
    
    # Assert dimensions match
    assert img.shape == masked_img.shape, "Image shape mismatch!"
    assert img.shape[:2] == mask.shape[:2], "Mask shape mismatch!"
    
    # Check that pixels outside mask are blacked out
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    outside_mask_pixels = masked_img[mask_3d == 0]
    assert np.all(outside_mask_pixels == 0), "Found non-black pixels outside mask in classification dataset!"
    
    # Check that pixels inside mask match the original image
    inside_mask_pixels_original = img[mask_3d > 0]
    inside_mask_pixels_masked = masked_img[mask_3d > 0]
    assert np.all(inside_mask_pixels_original == inside_mask_pixels_masked), "Masked pixels do not match original image!"
    print(f"[OK] Masking verified for sample: {sample_name} (dimensions: {img.shape})")
    
    # 5. Check normal images (KEEP_NORMAL_UNMASKED)
    sample_normal = random.choice(normal_class)
    normal_img_path = os.path.join(CLASS_DIR, 'normal', sample_normal)
    normal_img = cv2.imread(normal_img_path)
    # Check if normal image is not completely blacked out
    assert np.max(normal_img) > 0, f"Normal image {sample_normal} is completely blacked out, but KEEP_NORMAL_UNMASKED is set to True!"
    print(f"[OK] Normal image verified: {sample_normal} is intact (not blacked out).")
    
    print("\n=== All Verifications Passed Successfully! ===")

if __name__ == "__main__":
    verify()
