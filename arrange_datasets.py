import os
import glob
import cv2
import numpy as np

# Configuration
KEEP_NORMAL_UNMASKED = True # Set to False if you want normal images to be completely blacked out

# Destination directories
UNET_IMG_DIR = r"e:\CONFERENCE\unet_dataset\images"
UNET_MASK_DIR = r"e:\CONFERENCE\unet_dataset\masks"
CLASS_DIR = r"e:\CONFERENCE\classification_dataset"

# Create directories
os.makedirs(UNET_IMG_DIR, exist_ok=True)
os.makedirs(UNET_MASK_DIR, exist_ok=True)
for c in ['benign', 'malignant', 'normal']:
    os.makedirs(os.path.join(CLASS_DIR, c), exist_ok=True)

print("Created target directories successfully.")

def process_busi_style(dataset_name, src_dir, folders_mapping):
    """
    Processes BUSI style datasets where images and masks are in the same folder.
    folders_mapping: dict mapping src_folder -> target_class
    """
    print(f"\nProcessing BUSI-style dataset: {dataset_name} at {src_dir}")
    
    for folder, target_class in folders_mapping.items():
        folder_path = os.path.join(src_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"  Warning: Folder {folder_path} not found. Skipping.")
            continue
            
        all_files = os.listdir(folder_path)
        # Find images (exclude mask files)
        images = [f for f in all_files if f.endswith('.png') and '_mask' not in f]
        print(f"  Folder '{folder}' (maps to '{target_class}'): Found {len(images)} images.")
        
        count = 0
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            
            # Base name to find masks
            base_name = os.path.splitext(img_name)[0]
            # Find all matching masks
            mask_pattern = os.path.join(folder_path, f"{base_name}_mask*.png")
            mask_files = glob.glob(mask_pattern)
            
            if not mask_files:
                print(f"    Warning: No mask found for {img_name}. Skipping.")
                continue
                
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"    Error: Could not read image {img_path}. Skipping.")
                continue
                
            # Load and combine all matching masks using logical OR
            combined_mask = None
            for mask_file in mask_files:
                mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    continue
                if combined_mask is None:
                    combined_mask = mask_img
                else:
                    combined_mask = cv2.max(combined_mask, mask_img)
            
            if combined_mask is None:
                print(f"    Error: Failed to load masks for {img_name}. Skipping.")
                continue
            
            # Ensure dimensions match
            if img.shape[:2] != combined_mask.shape[:2]:
                combined_mask = cv2.resize(combined_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Output names
            out_name = f"{dataset_name}_{img_name.replace(' ', '_')}"
            
            # Save to UNet dataset
            cv2.imwrite(os.path.join(UNET_IMG_DIR, out_name), img)
            cv2.imwrite(os.path.join(UNET_MASK_DIR, out_name), combined_mask)
            
            # Save to Classification dataset (apply masking)
            mask_3d = np.repeat(combined_mask[:, :, np.newaxis], 3, axis=2)
            if target_class == 'normal' and KEEP_NORMAL_UNMASKED:
                masked_img = img
            else:
                masked_img = np.where(mask_3d > 0, img, 0)
                
            cv2.imwrite(os.path.join(CLASS_DIR, target_class, out_name), masked_img)
            count += 1
            
        print(f"    Successfully processed {count}/{len(images)} images.")

def process_uclm_separated():
    """
    Processes bus_uclm_separated images and maps them to masks in BUS-UCLM main masks directory.
    """
    src_dir = r"e:\CONFERENCE\bus_uclm_separated"
    masks_dir = r"e:\CONFERENCE\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM\masks"
    
    print(f"\nProcessing bus_uclm_separated at {src_dir}")
    print(f"  Mapping masks from {masks_dir}")
    
    if not os.path.isdir(src_dir):
        print(f"  Error: bus_uclm_separated folder not found at {src_dir}")
        return
    if not os.path.isdir(masks_dir):
        print(f"  Error: BUS-UCLM masks directory not found at {masks_dir}")
        return
        
    folders_mapping = {
        'benign': 'benign',
        'malign': 'malignant',
        'normal': 'normal'
    }
    
    for folder, target_class in folders_mapping.items():
        folder_path = os.path.join(src_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"  Warning: Folder {folder_path} not found. Skipping.")
            continue
            
        images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        print(f"  Folder '{folder}' (maps to '{target_class}'): Found {len(images)} images.")
        
        count = 0
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            mask_path = os.path.join(masks_dir, img_name)
            
            if not os.path.exists(mask_path):
                print(f"    Warning: Mask file not found for {img_name} at {mask_path}. Skipping.")
                continue
                
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                print(f"    Error: Could not read image or mask for {img_name}. Skipping.")
                continue
                
            # Ensure dimensions match
            if img.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
            # Output name
            out_name = f"uclm_{img_name}"
            
            # Save to UNet dataset
            cv2.imwrite(os.path.join(UNET_IMG_DIR, out_name), img)
            cv2.imwrite(os.path.join(UNET_MASK_DIR, out_name), mask)
            
            # Save to Classification dataset (apply masking)
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            if target_class == 'normal' and KEEP_NORMAL_UNMASKED:
                masked_img = img
            else:
                masked_img = np.where(mask_3d > 0, img, 0)
                
            cv2.imwrite(os.path.join(CLASS_DIR, target_class, out_name), masked_img)
            count += 1
            
        print(f"    Successfully processed {count}/{len(images)} images.")

# 1. Process main BUSI
process_busi_style(
    dataset_name="busi",
    src_dir=r"e:\CONFERENCE\Dataset_BUSI_with_GT",
    folders_mapping={'benign': 'benign', 'malignant': 'malignant', 'normal': 'normal'}
)

# 2. Process LCIS from New folder
process_busi_style(
    dataset_name="lcis",
    src_dir=r"e:\CONFERENCE\New folder\Dataset_BUSI_with_GT",
    folders_mapping={'LCIS': 'benign'} # Ignore benign and malignant duplicates, map LCIS to benign
)

# 3. Process separated UCLM
process_uclm_separated()

print("\n=== Dataset Consolidation & Reorganization Completed ===")
print("Summary of files in unet_dataset:")
print(f"  Images: {len(os.listdir(UNET_IMG_DIR))} files")
print(f"  Masks:  {len(os.listdir(UNET_MASK_DIR))} files")

print("Summary of files in classification_dataset:")
for c in ['benign', 'malignant', 'normal']:
    print(f"  {c}: {len(os.listdir(os.path.join(CLASS_DIR, c)))} files")
