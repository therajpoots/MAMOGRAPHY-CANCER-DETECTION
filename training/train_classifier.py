"""
BUS-XAINet Classification Training
====================================
K-fold cross-validation training with Focal Loss,
OneCycleLR scheduling, tqdm progress bars, early stopping,
and interactive performance approval before saving checkpoints.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
from collections import defaultdict
from tqdm import tqdm
from typing import Optional
import matplotlib.pyplot as plt
from tkinter import messagebox, Tk

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bus_xainet import BUSXAINet, FocalLoss
from preprocessing.dataset import (
    BUSClassificationDataset, get_classification_file_lists,
    get_class_weights, get_weighted_sampler,
)
from preprocessing.pipeline import get_preprocessor
from preprocessing.augmentation import get_augmentor
from torch.utils.data import DataLoader
from deployment.export_model import export_bus_xainet
from plots.performance_plots import plot_confusion_matrix, plot_training_curves


def verify_classifier_visual(
    best_model,
    val_dataset,
    history,
    y_true,
    y_pred,
    y_probs,
    device,
    classes=["Benign", "Malignant", "Normal"],
    interactive: bool = True
):
    """
    Displays training curves, confusion matrix, and prediction examples for user approval.
    """
    if not interactive:
        print("[INFO] Non-interactive mode: Auto-approved classifier performance.")
        return True

    print("\n[INFO] Opening classifier verification windows...")
    
    # Set premium light style
    plt.rcParams["figure.facecolor"] = "#FFFFFF"
    plt.rcParams["axes.facecolor"] = "#F8F9FA"
    plt.rcParams["text.color"] = "#2C3E50"
    plt.rcParams["axes.labelcolor"] = "#2C3E50"
    plt.rcParams["xtick.color"] = "#34495E"
    plt.rcParams["ytick.color"] = "#34495E"
    plt.rcParams["grid.color"] = "#E2E8F0"
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    
    # 1. Figure 1: Training Curves
    epochs = len(history["train_loss"])
    epochs_range = range(1, epochs + 1)
    
    fig1, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#FFFFFF")
    fig1.suptitle("Classifier Training History (Best Fold)", color="#2C3E50", fontsize=14, weight="bold")
    
    # Loss plot
    ax_loss.plot(epochs_range, history["train_loss"], label="Train Loss", color="#D9534F", linewidth=2)
    ax_loss.plot(epochs_range, history["val_loss"], label="Val Loss", color="#D9534F", linestyle="--", linewidth=2)
    ax_loss.set_xlabel("Epochs", color="#2C3E50")
    ax_loss.set_ylabel("Loss", color="#2C3E50")
    ax_loss.set_title("Focal Loss", color="#008080")
    ax_loss.legend(facecolor="#F8F9FA", edgecolor="#CBD5E1")
    ax_loss.grid(True, linestyle=":", alpha=0.5, color="#E2E8F0")
    
    # Accuracy plot
    ax_acc.plot(epochs_range, history["train_acc"], label="Train Acc", color="#5CB85C", linewidth=2)
    ax_acc.plot(epochs_range, history["val_acc"], label="Val Acc", color="#5CB85C", linestyle="--", linewidth=2)
    ax_acc.set_xlabel("Epochs", color="#2C3E50")
    ax_acc.set_ylabel("Accuracy", color="#2C3E50")
    ax_acc.set_title("Accuracy", color="#008080")
    ax_acc.legend(facecolor="#F8F9FA", edgecolor="#CBD5E1")
    ax_acc.grid(True, linestyle=":", alpha=0.5, color="#E2E8F0")
    
    plt.tight_layout()
    plt.show(block=False)
    
    # 2. Figure 2: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    fig2, ax_cm = plt.subplots(figsize=(6, 6), facecolor="#FFFFFF")
    im = ax_cm.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    cbar = fig2.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#34495E")
    
    ax_cm.set_xticks(np.arange(len(classes)))
    ax_cm.set_yticks(np.arange(len(classes)))
    ax_cm.set_xticklabels(classes, fontsize=10, color="#2C3E50")
    ax_cm.set_yticklabels(classes, fontsize=10, color="#2C3E50")
    
    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            percentage = f"{cm_norm[i, j]*100:.1f}%"
            count = f"({int(cm[i, j])})"
            ax_cm.text(
                j, i, f"{percentage}\n{count}",
                ha="center", va="center",
                color=text_color, fontsize=10, weight="bold"
            )
            
    ax_cm.set_xlabel("Predicted Class", fontsize=11, labelpad=10, color="#2C3E50")
    ax_cm.set_ylabel("True Class", fontsize=11, labelpad=10, color="#2C3E50")
    ax_cm.set_title("Normalized Confusion Matrix (Best Fold)", fontsize=12, pad=15, weight="bold", color="#008080")
    plt.tight_layout()
    plt.show(block=False)
    
    # 3. Figure 3: Predictions Gallery (4 Validation Samples)
    random.seed(42)
    indices = list(range(len(val_dataset)))
    sample_idxs = random.sample(indices, min(4, len(indices)))
    
    fig3, axes3 = plt.subplots(len(sample_idxs), 3, figsize=(11, 2.5 * len(sample_idxs)), facecolor="#FFFFFF")
    fig3.suptitle("Classifier Predictions on Validation Samples", color="#2C3E50", fontsize=14, weight="bold")
    
    for row_idx, val_idx in enumerate(sample_idxs):
        stacked_tensor, label = val_dataset[val_idx]
        img_np = stacked_tensor[0].numpy()
        mask_np = stacked_tensor[1].numpy()
        
        # Run inference to get probabilities
        input_tensor = stacked_tensor.unsqueeze(0).to(device)
        best_model.eval()
        with torch.no_grad():
            outputs = best_model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
        pred_label_idx = np.argmax(probs)
        pred_label_name = classes[pred_label_idx]
        true_label_name = classes[label]
        
        ax_row = axes3[row_idx] if len(sample_idxs) > 1 else axes3
        
        # Plot preprocessed image
        ax_row[0].imshow(img_np, cmap="gray")
        ax_row[0].set_title(f"Image (True: {true_label_name})", color="#2C3E50", fontsize=9)
        ax_row[0].axis("off")
        
        # Plot preprocessed mask
        ax_row[1].imshow(mask_np, cmap="gray")
        ax_row[1].set_title("Lesion Mask", color="#2C3E50", fontsize=9)
        ax_row[1].axis("off")
        
        # Plot probability bar chart
        y_pos = np.arange(len(classes))
        bars = ax_row[2].barh(y_pos, probs, color=["#337AB7", "#D9534F", "#5CB85C"])
        ax_row[2].set_yticks(y_pos)
        ax_row[2].set_yticklabels(classes, color="#2C3E50", fontsize=8)
        ax_row[2].set_xlim(0, 1.05)
        ax_row[2].set_title(f"Pred: {pred_label_name} ({probs[pred_label_idx]*100:.1f}%)", color="#008080", fontsize=9)
        ax_row[2].grid(True, axis="x", linestyle=":", alpha=0.3, color="#CBD5E1")
        
        # Add labels on bars
        for bar in bars:
            width = bar.get_width()
            ax_row[2].text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%", 
                           ha="left", va="center", color="#2C3E50", fontsize=7)
            
    plt.tight_layout()
    plt.show(block=False)
    
    # 4. Ask for approval
    approved = False
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        ans = messagebox.askyesno(
            "Classifier Training Verification",
            "Classifier training completed.\nDo you approve the model performance and wish to save the production checkpoints?",
            icon="question"
        )
        root.destroy()
        approved = ans
    except Exception:
        # Fallback to terminal input
        print("\n" + "="*50)
        print("Classifier Training is complete.")
        print("Matplotlib windows display training curves, confusion matrix, and validation samples.")
        ans = input("Do you approve the classifier performance? (yes/no): ").strip().lower()
        print("="*50 + "\n")
        approved = ans in ["yes", "y"]
        
    plt.close("all")
    return approved


def train_classifier(
    data_dir: str = r"e:\CONFERENCE\classification_dataset",
    output_dir: str = r"e:\CONFERENCE\outputs",
    batch_size: int = 32,
    epochs: int = 40,  # Train for 40 epochs as requested
    lr: float = 3e-5,
    max_lr: float = 3e-4,
    n_folds: int = 5,
    patience: int = 10,  # early stopping to prevent overfitting
    device: str = "auto",
    interactive: bool = True,
    num_workers: int = 0,
    p_elastic: Optional[float] = None,
):
    """Train BUS-XAINet classifier with k-fold cross-validation."""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    
    # Get all data
    train_list, val_list, test_list = get_classification_file_lists(data_dir)
    
    # Combine train+val for k-fold; keep test held out
    all_train = train_list + val_list
    all_labels = np.array([label for _, label in all_train])
    
    # Class weights for focal loss
    class_weights = get_class_weights(all_train).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # K-Fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    all_histories = {}
    
    preprocessor = get_preprocessor("classification")
    
    # Track the absolute best overall fold model
    best_overall_f1 = -1.0
    best_overall_fold_idx = -1
    best_fold_history = None
    best_fold_val_ds = None
    best_fold_y_true = None
    best_fold_y_pred = None
    best_fold_y_probs = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_train, all_labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")
        
        fold_train = [all_train[i] for i in train_idx]
        fold_val = [all_train[i] for i in val_idx]
        
        # Datasets
        train_ds = BUSClassificationDataset(
            data_dir, fold_train, preprocessor, get_augmentor("train", p_elastic=p_elastic)
        )
        val_ds = BUSClassificationDataset(
            data_dir, fold_val, preprocessor, None
        )
        
        sampler = get_weighted_sampler(fold_train)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        )
        
        # Model
        model = BUSXAINet(num_classes=3, pretrained=True).to(device)
        
        # Loss, optimizer, scheduler
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01) # weight decay to prevent overfitting
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs,
            steps_per_epoch=len(train_loader), pct_start=0.1,
        )
        
        scaler = GradScaler("cuda", enabled=(device == "cuda"))
        
        history = defaultdict(list)
        best_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # --- Train ---
            model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []
            
            pbar_train = tqdm(
                train_loader,
                desc=f"Fold {fold+1} Epoch {epoch+1:02d}/{epochs:02d} [Train]",
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
            )
            
            for images, labels in pbar_train:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                with autocast("cuda", enabled=(device == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                train_loss += loss.item() * images.size(0)
                train_preds.extend(outputs.argmax(1).cpu().numpy())
                train_targets.extend(labels.cpu().numpy())
                
                pbar_train.set_postfix(loss=f"{loss.item():.4f}")
            
            train_loss /= len(train_ds)
            train_acc = accuracy_score(train_targets, train_preds)
            
            # --- Validate ---
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            
            pbar_val = tqdm(
                val_loader,
                desc="             [Valid]",
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                leave=False
            )
            
            with torch.no_grad():
                for images, labels in pbar_val:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    val_preds.extend(outputs.argmax(1).cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
                    
                    pbar_val.set_postfix(loss=f"{loss.item():.4f}")
            
            val_loss /= len(val_ds)
            val_acc = accuracy_score(val_targets, val_preds)
            prec, rec, f1, _ = precision_recall_fscore_support(
                val_targets, val_preds, average="macro", zero_division=0
            )
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(f1)
            history["val_precision"].append(prec)
            history["val_recall"].append(rec)
            
            print(
                f"Epoch {epoch+1:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {f1:.4f}"
            )
            
            # Save best fold model and handle early stopping
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                torch.save({
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_f1": best_f1,
                }, os.path.join(output_dir, "models", f"xainet_fold{fold}.pth"))
                print(f"  -> Saved new best fold checkpoint (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at fold {fold+1} epoch {epoch+1}")
                    break
        
        # Record results of this fold
        fold_results.append({
            "fold": fold,
            "best_f1": best_f1,
            "val_acc": val_acc,
            "val_precision": prec,
            "val_recall": rec,
        })
        all_histories[f"fold_{fold}"] = dict(history)
        print(f"\nFold {fold+1} Best validation F1: {best_f1:.4f}")
        
        # Check if this fold is the overall best
        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_overall_fold_idx = fold
            best_fold_history = dict(history)
            best_fold_val_ds = val_ds
            
            # Compute final predictions and probs of the best checkpoint
            best_checkpoint_path = os.path.join(output_dir, "models", f"xainet_fold{fold}.pth")
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            best_model = BUSXAINet(num_classes=3, pretrained=False).to(device)
            best_model.load_state_dict(checkpoint["model_state_dict"])
            best_model.eval()
            
            y_true, y_pred, y_probs = [], [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    outputs = best_model(images)
                    probs = torch.softmax(outputs, dim=1)
                    
                    y_true.extend(labels.numpy())
                    y_pred.extend(outputs.argmax(1).cpu().numpy())
                    y_probs.extend(probs.cpu().numpy())
                    
            best_fold_y_true = np.array(y_true)
            best_fold_y_pred = np.array(y_pred)
            best_fold_y_probs = np.array(y_probs)
            
    # Cross Validation Summary
    print(f"\n{'='*60}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    f1_scores = [r["best_f1"] for r in fold_results]
    print(f"Mean F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    
    # Save results
    with open(os.path.join(output_dir, "classifier_results.json"), "w") as f:
        json.dump({"fold_results": fold_results}, f, indent=2)
    with open(os.path.join(output_dir, "classifier_history.json"), "w") as f:
        json.dump(all_histories, f, indent=2)
        
    # Instantiate best model to run interactive popup
    best_model_checkpoint = os.path.join(output_dir, "models", f"xainet_fold{best_overall_fold_idx}.pth")
    checkpoint = torch.load(best_model_checkpoint, map_location=device)
    best_model = BUSXAINet(num_classes=3, pretrained=False).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    
    # Interactive Popup Approval
    approved = verify_classifier_visual(
        best_model=best_model,
        val_dataset=best_fold_val_ds,
        history=best_fold_history,
        y_true=best_fold_y_true,
        y_pred=best_fold_y_pred,
        y_probs=best_fold_y_probs,
        device=device,
        interactive=interactive
    )
    
    if not approved:
        print("[ABORTED] Classifier performance rejected by user. Checkpoints will not be serialized to production.")
        sys.exit(1)
        
    print("[SUCCESS] Classifier performance approved. Exporting best model to production formats...")
    
    # Export best classifier model to production-ready formats (TorchScript and ONNX)
    export_bus_xainet(best_model_checkpoint, os.path.join(output_dir, "models"))
    
    # Generate and save premium validation plots for walkthrough documentation
    print("[INFO] Saving premium validation plots...")
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(best_fold_y_true, best_fold_y_pred)
    plot_confusion_matrix(cm, ["Benign", "Malignant", "Normal"], os.path.join(output_dir, "plots", "confusion_matrix.png"))
    
    # 2. Training Curves
    history_for_plot = {
        "train_loss": best_fold_history["train_loss"],
        "val_loss": best_fold_history["val_loss"],
        "train_dice": best_fold_history["train_acc"],
        "val_dice": best_fold_history["val_acc"]
    }
    plot_training_curves(history_for_plot, os.path.join(output_dir, "plots", "training_curves.png"))
    
    print("[SUCCESS] All model components, checkpoints, and performance plots saved successfully!")
    return fold_results, all_histories


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BUS-XAINet 2-Channel Classifier")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--no-interactive", action="store_true", help="Bypass interactive popups and auto-approve")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--p-elastic", type=float, default=0.15, help="Elastic deformation probability (0.0 to disable)")
    args = parser.parse_args()
    
    train_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_folds=args.folds,
        patience=args.patience,
        interactive=not args.no_interactive,
        num_workers=args.num_workers,
        p_elastic=args.p_elastic
    )
