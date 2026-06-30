"""
U-Net Training Script
=====================
Training loop for Attention U-Net segmentation with:
- tqdm progress bars for live training progress
- Early stopping & weight decay for overfitting prevention
- Best model checkpointing
- Post-training interactive popup to verify masking quality
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from collections import defaultdict
from tqdm import tqdm
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import AttentionUNet, DiceBCELoss, dice_score, iou_score
from preprocessing.dataset import create_segmentation_dataloaders
import cv2


def verify_segmentation_visual(model, device, interactive: bool = True):
    """
    Shows a visual popup demonstrating masking predictions and asks for user approval.
    """
    print("\n[INFO] Opening visual verification windows...")
    import matplotlib.pyplot as plt
    import random
    from tkinter import messagebox, Tk
    
    model.eval()
    
    # 1. Load samples from unet_dataset (with ground truth)
    unet_img_dir = r"e:\CONFERENCE\unet_dataset\images"
    unet_mask_dir = r"e:\CONFERENCE\unet_dataset\masks"
    if not os.path.exists(unet_img_dir):
        print("[WARNING] unet_dataset images directory not found. Skipping visual demo.")
        return True
        
    all_files = sorted(os.listdir(unet_img_dir))
    random.seed(42)
    samples = random.sample(all_files, min(4, len(all_files)))
    
    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 2.5 * len(samples)), facecolor="#FFFFFF")
    fig.suptitle("U-Net Segmentation Examples (unet_dataset)", color="#2C3E50", fontsize=12, weight="bold")
    
    from preprocessing.pipeline import get_preprocessor
    prep = get_preprocessor("segmentation")
    
    for idx, fname in enumerate(samples):
        img_raw = cv2.imread(os.path.join(unet_img_dir, fname))
        mask_raw = cv2.imread(os.path.join(unet_mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        
        img_pre = prep.process_image(img_raw)
        mask_pre = prep.process_mask(mask_raw)
        
        # Run forward pass
        img_tensor = torch.from_numpy(img_pre).unsqueeze(0).unsqueeze(0).to(device).float()
        with torch.no_grad():
            pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
            
        pred_bin = (pred > 0.5).astype(np.uint8) * 255
        
        # Determine ax handles
        ax_row = axes[idx] if len(samples) > 1 else axes
        
        ax_row[0].imshow(img_pre, cmap="gray")
        ax_row[0].set_title("Original Image", color="#2C3E50", fontsize=8)
        ax_row[0].axis("off")
        
        ax_row[1].imshow(mask_pre, cmap="gray")
        ax_row[1].set_title("Ground Truth Mask", color="#2C3E50", fontsize=8)
        ax_row[1].axis("off")
        
        ax_row[2].imshow(pred_bin, cmap="gray")
        ax_row[2].set_title("Predicted Mask", color="#2C3E50", fontsize=8)
        ax_row[2].axis("off")
        
    plt.tight_layout()
    
    # 2. Visual demo on unmasked dataset (bus_uclm_separated)
    uclm_dir = r"e:\CONFERENCE\bus_uclm_separated"
    uclm_samples = []
    if os.path.exists(uclm_dir):
        for sub in ["benign", "malign", "normal"]:
            sub_dir = os.path.join(uclm_dir, sub)
            if os.path.exists(sub_dir):
                files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith(".png")]
                if files:
                    uclm_samples.append((random.choice(files), sub))
                    
    if uclm_samples:
        fig2, axes2 = plt.subplots(len(uclm_samples), 2, figsize=(8, 2.5 * len(uclm_samples)), facecolor="#FFFFFF")
        fig2.suptitle("U-Net Prediction on Unmasked Dataset (bus_uclm_separated)", color="#2C3E50", fontsize=12, weight="bold")
        for idx, (path, sub) in enumerate(uclm_samples):
            img_raw = cv2.imread(path)
            img_pre = prep.process_image(img_raw)
            img_tensor = torch.from_numpy(img_pre).unsqueeze(0).unsqueeze(0).to(device).float()
            with torch.no_grad():
                pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8) * 255
            
            ax_row2 = axes2[idx] if len(uclm_samples) > 1 else axes2
            
            ax_row2[0].imshow(img_pre, cmap="gray")
            ax_row2[0].set_title(f"Original ({sub})", color="#2C3E50", fontsize=8)
            ax_row2[0].axis("off")
            
            ax_row2[1].imshow(pred_bin, cmap="gray")
            ax_row2[1].set_title("Predicted Mask", color="#2C3E50", fontsize=8)
            ax_row2[1].axis("off")
        plt.tight_layout()
        
    if not interactive:
        out_plot_dir = r"e:\CONFERENCE\outputs\plots"
        os.makedirs(out_plot_dir, exist_ok=True)
        fig.savefig(os.path.join(out_plot_dir, "unet_verification_samples_gt.png"), dpi=300, facecolor="#FFFFFF")
        if uclm_samples:
            fig2.savefig(os.path.join(out_plot_dir, "unet_verification_samples_uclm.png"), dpi=300, facecolor="#FFFFFF")
        plt.close("all")
        print("[INFO] Non-interactive mode: Auto-approved segmentation quality. Saved plots to outputs/plots/")
        return True

    plt.show(block=False)
    
    # 3. Popup dialog asking for confirmation
    approved = False
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        ans = messagebox.askyesno(
            "U-Net Segmentation Verification",
            "U-Net training completed.\nDo you approve the predicted masks to proceed with Classifier training?",
            icon="question"
        )
        root.destroy()
        approved = ans
    except Exception:
        # Fallback to terminal input
        print("\n" + "="*50)
        print("U-Net Segmentation Training is complete.")
        print("Matplotlib windows display predicted masks.")
        ans = input("Do you approve the masking quality? (yes/no): ").strip().lower()
        print("="*50 + "\n")
        approved = ans in ["yes", "y"]
        
    plt.close("all")
    return approved


def train_unet(
    image_dir: str = r"e:\CONFERENCE\unet_dataset\images",
    mask_dir: str = r"e:\CONFERENCE\unet_dataset\masks",
    output_dir: str = r"e:\CONFERENCE\outputs",
    batch_size: int = 16,
    epochs: int = 40,  # Set to at least 40 epochs as requested
    lr: float = 3e-4,
    weight_decay: float = 1e-4, # prevent overfitting
    patience: int = 10,         # early stopping
    device: str = "auto",
    interactive: bool = True,
    num_workers: int = 0,
    p_elastic: Optional[float] = None,
):
    """Train Attention U-Net for breast ultrasound segmentation."""
    
    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training U-Net on: {device}")
    
    # Create output dirs
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    
    # Data loaders — use val_ratio=0.20 so the val set gets enough images
    # even when patient image counts vary widely across the dataset
    dataloaders = create_segmentation_dataloaders(
        image_dir, mask_dir, batch_size=batch_size, val_ratio=0.20, test_ratio=0.10,
        num_workers=num_workers, p_elastic=p_elastic
    )
    
    # Model
    model = AttentionUNet(in_channels=1, out_channels=1, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss, optimizer, scheduler
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    # Differential learning rates: encoder (pretrained ResNet34) needs smaller LR
    # than the randomly-initialized decoder to avoid destroying pretrained features
    encoder_params = list(model.encoder0.parameters()) + \
                     list(model.encoder1.parameters()) + \
                     list(model.encoder2.parameters()) + \
                     list(model.encoder3.parameters()) + \
                     list(model.encoder4.parameters())
    decoder_params = [p for p in model.parameters()
                      if not any(p is ep for ep in encoder_params)]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr * 0.1},   # pretrained: 10x smaller LR
        {"params": decoder_params, "lr": lr},          # fresh decoder: full LR
    ], weight_decay=weight_decay)
    # CosineAnnealingLR: smooth single decay avoids the LR spike of WarmRestarts
    # that was causing val dice to oscillate after every restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Mixed precision
    scaler = GradScaler("cuda", enabled=(device == "cuda"))
    
    # Training history
    history = defaultdict(list)
    best_dice = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # === Training ===
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        n_train = 0
        
        # tqdm progress bar
        pbar_train = tqdm(
            dataloaders["train"],
            desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Train]",
            bar_format="{l_bar}{bar:25}{r_bar}{bar:-25b}"
        )
        
        for images, masks in pbar_train:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast("cuda", enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            d_val = dice_score(torch.sigmoid(outputs), masks)
            iou_val = iou_score(torch.sigmoid(outputs), masks)
            
            train_loss += loss.item() * images.size(0)
            train_dice += d_val * images.size(0)
            train_iou += iou_val * images.size(0)
            n_train += images.size(0)
            
            # Live progress bar updates
            pbar_train.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d_val:.4f}")
        
        scheduler.step()
        
        train_loss /= n_train
        train_dice /= n_train
        train_iou /= n_train
        
        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        n_val = 0
        
        pbar_val = tqdm(
            dataloaders["val"],
            desc="         [Valid]",
            bar_format="{l_bar}{bar:25}{r_bar}{bar:-25b}",
            leave=False
        )
        
        with torch.no_grad():
            for images, masks in pbar_val:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                d_val = dice_score(torch.sigmoid(outputs), masks)
                iou_val = iou_score(torch.sigmoid(outputs), masks)
                
                val_loss += loss.item() * images.size(0)
                val_dice += d_val * images.size(0)
                val_iou += iou_val * images.size(0)
                n_val += images.size(0)
                
                pbar_val.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d_val:.4f}")
        
        val_loss /= n_val
        val_dice /= n_val
        val_iou /= n_val
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} Summary | "
            f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f}"
        )
        
        # Best model checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
            }, os.path.join(output_dir, "models", "unet_best.pth"))
            print(f"  -> Saved new best model checkpoint (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Save training history
    with open(os.path.join(output_dir, "unet_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nU-Net Training complete. Best validation Dice: {best_dice:.4f}")
    
    # Visual validation popup
    approved = verify_segmentation_visual(model, device, interactive=interactive)
    if not approved:
        print("[ABORTED] Masking rejected by user. Aborting training pipeline.")
        sys.exit(1)
        
    print("[SUCCESS] Masking approved. Exporting U-Net to production format...")
    from deployment.export_model import export_unet
    best_checkpoint = os.path.join(output_dir, "models", "unet_best.pth")
    export_unet(best_checkpoint, os.path.join(output_dir, "models"))
    
    print("[SUCCESS] U-Net model successfully exported. Proceeding with Classifier training.")
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Attention U-Net Segmentation Model")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--no-interactive", action="store_true", help="Bypass interactive popups and auto-approve")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--p-elastic", type=float, default=0.15, help="Elastic deformation probability (0.0 to disable)")
    args = parser.parse_args()
    
    train_unet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        interactive=not args.no_interactive,
        num_workers=args.num_workers,
        p_elastic=args.p_elastic
    )
