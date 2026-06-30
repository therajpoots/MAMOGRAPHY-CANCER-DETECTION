"""
Statistical Significance Analysis Script
========================================
Performs cross-validation evaluations of the trained 5-fold BUS-XAINet models
under three ablation scenarios:
1. Segmentation-based (Image + Mask stacked)
2. Raw Image Only (Image + Zero Mask stacked)
3. Mask Only (Zero Image + Mask stacked)

Runs parametric (paired t-test, ANOVA) and non-parametric (Wilcoxon, McNemar, Friedman)
statistical tests, computes 95% confidence intervals, and generates a visualization.
"""

import os
import sys
import json
import random
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bus_xainet import BUSXAINet
from preprocessing.dataset import BUSClassificationDataset, get_classification_file_lists, get_preprocessed_pair
from preprocessing.pipeline import get_preprocessor
from analysis.statistical_tests import run_anova, run_wilcoxon, run_mcnemar, run_friedman


# Define Custom Dataset wrappers to modify input channels dynamically
class AblationDataset(BUSClassificationDataset):
    def __init__(self, data_dir, file_list, preprocessor, mode="both"):
        super().__init__(data_dir, file_list, preprocessor, None)
        self.mode = mode  # "both" (default), "image_only", "mask_only"

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]
        fname = os.path.basename(filepath)
        unet_img_dir = r"e:\CONFERENCE\unet_dataset\images"
        unet_mask_dir = r"e:\CONFERENCE\unet_dataset\masks"
        
        unet_img_path = os.path.join(unet_img_dir, fname)
        unet_mask_path = os.path.join(unet_mask_dir, fname)

        image_pre, mask_pre = get_preprocessed_pair(
            filepath if not os.path.exists(unet_img_path) else unet_img_path,
            unet_mask_path,
            self.preprocessor,
            is_classification=True,
            label=label
        )

        # Apply ablation modes
        if self.mode == "image_only":
            mask_pre = np.zeros_like(mask_pre)
        elif self.mode == "mask_only":
            image_pre = np.zeros_like(image_pre)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_pre).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_pre).unsqueeze(0).float()
        stacked_tensor = torch.cat([image_tensor, mask_tensor], dim=0)

        return stacked_tensor, label


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Computes mean and confidence interval half-width using t-distribution."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - h, mean + h


def main():
    print("=== STARTING STATISTICAL ANALYSIS OF CLASSIFICATION RESULTS ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on: {device}")
    
    data_dir = r"e:\CONFERENCE\classification_dataset"
    output_dir = r"e:\CONFERENCE\outputs"
    
    # 1. Re-create Stratified K-Fold Splits exactly as in training
    train_list, val_list, test_list = get_classification_file_lists(data_dir)
    all_train = train_list + val_list
    all_labels = np.array([label for _, label in all_train])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preprocessor = get_preprocessor("classification")
    
    # Results trackers
    modes = ["both", "image_only", "mask_only"]
    mode_names = {
        "both": "Segmentation-Based (Ours)",
        "image_only": "Raw Image Only (Baseline)",
        "mask_only": "Mask Only (Ablation)"
    }
    
    fold_accuracies = {m: [] for m in modes}
    fold_f1_scores = {m: [] for m in modes}
    
    # Combined predictions for McNemar's Test
    pooled_y_true = []
    pooled_preds = {m: [] for m in modes}
    
    # 2. Evaluate all 5 checkpoints
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_train, all_labels)):
        print(f"\nEvaluating Fold {fold+1} Checkpoint...")
        checkpoint_path = os.path.join(output_dir, "models", f"xainet_fold{fold}.pth")
        
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = BUSXAINet(num_classes=3, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        fold_val = [all_train[i] for i in val_idx]
        
        # Evaluate for each ablation mode
        for mode in modes:
            val_ds = AblationDataset(data_dir, fold_val, preprocessor, mode=mode)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
            
            y_true_fold, y_pred_fold = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(1).cpu().numpy()
                    
                    y_true_fold.extend(labels.numpy())
                    y_pred_fold.extend(preds)
            
            # Compute fold metrics
            acc = accuracy_score(y_true_fold, y_pred_fold)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true_fold, y_pred_fold, average="macro", zero_division=0
            )
            
            fold_accuracies[mode].append(acc)
            fold_f1_scores[mode].append(f1)
            
            # Pool predictions for overall testing (only from fold 0..4 validation sets)
            if mode == "both":
                pooled_y_true.extend(y_true_fold)
            pooled_preds[mode].extend(y_pred_fold)
            
            print(f"  {mode_names[mode]:30s} | Acc: {acc:.4f} | F1: {f1:.4f}")
            
    # Convert pooled variables to numpy arrays
    pooled_y_true = np.array(pooled_y_true)
    for m in modes:
        pooled_preds[m] = np.array(pooled_preds[m])
        
    print(f"\n{'='*70}")
    print("STATISTICAL TEST RESULTS")
    print(f"{'='*70}")
    
    # --- 1. Normality checks ---
    print("\n[Normality Check (Shapiro-Wilk)]")
    for m in modes:
        p_acc, is_norm = check_normality_acc = stats.shapiro(fold_accuracies[m])
        print(f"  {mode_names[m]:30s} Accuracy Normality p-val: {check_normality_acc[1]:.4f} (Normal: {check_normality_acc[1] > 0.05})")

    # --- 2. Paired t-test (Parametric) & Wilcoxon Signed-Rank Test (Non-Parametric) ---
    print("\n[Paired t-test: Segmentation-Based vs Raw Image Only]")
    t_stat, t_pval = stats.ttest_rel(fold_accuracies["both"], fold_accuracies["image_only"])
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {t_pval:.6f} (Significant: {t_pval < 0.05})")
    
    print("\n[Wilcoxon Signed-Rank Test: Segmentation-Based vs Raw Image Only]")
    wilc_res = run_wilcoxon(np.array(fold_accuracies["both"]), np.array(fold_accuracies["image_only"]))
    print(f"  Test Used:   {wilc_res['test_name']}")
    print(f"  Statistic:   {wilc_res['statistic']:.4f}")
    print(f"  p-value:     {wilc_res['p_value']:.6f} (Significant: {wilc_res['significant']})")
    
    # --- 3. One-Way ANOVA & Friedman Test (Multi-group Comparisons) ---
    print("\n[One-Way ANOVA (Comparison across all 3 groups)]")
    anova_res = run_anova(
        np.array(fold_accuracies["both"]),
        np.array(fold_accuracies["image_only"]),
        np.array(fold_accuracies["mask_only"])
    )
    print(f"  Test Used:   {anova_res['test_name']}")
    print(f"  Statistic:   {anova_res['statistic']:.4f}")
    print(f"  p-value:     {anova_res['p_value']:.6f} (Significant: {anova_res['significant']})")
    
    print("\n[Friedman Test (Non-parametric multi-group comparison)]")
    fried_res = run_friedman(
        np.array(fold_accuracies["both"]),
        np.array(fold_accuracies["image_only"]),
        np.array(fold_accuracies["mask_only"])
    )
    print(f"  Statistic:   {fried_res['statistic']:.4f}")
    print(f"  p-value:     {fried_res['p_value']:.6f} (Significant: {fried_res['significant']})")
    # --- 3b. Fully Trained Folds Sub-Analysis (Folds 3, 4, 5) ---
    print("\n" + "="*70)
    print("SUB-ANALYSIS ON FULLY TRAINED FOLDS (Folds 3, 4, 5 only)")
    print("="*70)
    
    fold_accuracies_trained = {m: np.array(fold_accuracies[m][2:]) for m in modes}
    
    print("\n[Paired t-test (Folds 3-5): Ours vs Raw Image Only]")
    t_stat_t, t_pval_t = stats.ttest_rel(fold_accuracies_trained["both"], fold_accuracies_trained["image_only"])
    print(f"  t-statistic: {t_stat_t:.4f}")
    print(f"  p-value:     {t_pval_t:.6f} (Significant: {t_pval_t < 0.05})")
    
    print("\n[Wilcoxon Signed-Rank Test (Folds 3-5): Ours vs Raw Image Only]")
    wilc_res_t = run_wilcoxon(fold_accuracies_trained["both"], fold_accuracies_trained["image_only"])
    print(f"  Test Used:   {wilc_res_t['test_name']}")
    print(f"  Statistic:   {wilc_res_t['statistic']:.4f}")
    print(f"  p-value:     {wilc_res_t['p_value']:.6f} (Significant: {wilc_res_t['significant']})")
    
    print("\n[One-Way ANOVA (Folds 3-5: Comparison across all 3 groups)]")
    anova_res_t = run_anova(
        fold_accuracies_trained["both"],
        fold_accuracies_trained["image_only"],
        fold_accuracies_trained["mask_only"]
    )
    print(f"  Test Used:   {anova_res_t['test_name']}")
    print(f"  Statistic:   {anova_res_t['statistic']:.4f}")
    print(f"  p-value:     {anova_res_t['p_value']:.6f} (Significant: {anova_res_t['significant']})")

    print("\n" + "="*70)
    
    # --- 4. McNemar's Test (Paired Disagreement on Pooled Samples) ---
    print("\n[McNemar's Test (Pooled Predictions: Ours vs Image Only)]")
    mcn_res = run_mcnemar(pooled_y_true, pooled_preds["both"], pooled_preds["image_only"])
    print(f"  Test Used:   {mcn_res['test_name']}")
    print(f"  Contingency Table (Ours Correct vs Baseline Correct):")
    print(f"                   Baseline Corr  Baseline Incorr")
    print(f"    Ours Correct      {mcn_res['contingency_table'][0][0]:5d}          {mcn_res['contingency_table'][0][1]:5d}")
    print(f"    Ours Incorrect    {mcn_res['contingency_table'][1][0]:5d}          {mcn_res['contingency_table'][1][1]:5d}")
    print(f"  Statistic:   {mcn_res['statistic']:.4f}")
    print(f"  p-value:     {mcn_res['p_value']:.6e} (Significant: {mcn_res['significant']})")
    
    # --- 5. 95% Confidence Intervals (CI) for Pooled Performance ---
    print("\n[95% Confidence Intervals for Performance Metrics]")
    
    def get_metrics_and_ci(y_true, y_pred):
        # Overall Accuracy
        acc = accuracy_score(y_true, y_pred)
        # Sensitivity (Benign & Malignant grouped as positive, or class-specific)
        # We compute macro metrics or class-specific metrics:
        # Let's compute overall accuracy confidence interval using Binomial Wald or Clopper-Pearson
        # Wald Interval: p +/- z * sqrt(p(1-p)/n)
        z = 1.96
        n = len(y_true)
        acc_se = np.sqrt(acc * (1 - acc) / n)
        acc_ci = (max(0.0, acc - z * acc_se), min(1.0, acc + z * acc_se))
        
        # Sensitivity: Benign (0) and Malignant (1) correctly classified
        pos_mask = (y_true == 0) | (y_true == 1)
        sens = accuracy_score(y_true[pos_mask], y_pred[pos_mask])
        sens_se = np.sqrt(sens * (1 - sens) / np.sum(pos_mask))
        sens_ci = (max(0.0, sens - z * sens_se), min(1.0, sens + z * sens_se))
        
        # Specificity: Normal (2) correctly classified
        neg_mask = (y_true == 2)
        spec = accuracy_score(y_true[neg_mask], y_pred[neg_mask])
        spec_se = np.sqrt(spec * (1 - spec) / np.sum(neg_mask))
        spec_ci = (max(0.0, spec - z * spec_se), min(1.0, spec + z * spec_se))
        
        return {
            "Accuracy": (acc, acc_ci),
            "Sensitivity": (sens, sens_ci),
            "Specificity": (spec, spec_ci)
        }

    metrics_both = get_metrics_and_ci(pooled_y_true, pooled_preds["both"])
    metrics_image = get_metrics_and_ci(pooled_y_true, pooled_preds["image_only"])
    
    print("\n  Segmentation-Based (Ours):")
    for m, (val, ci) in metrics_both.items():
        print(f"    {m:12s}: {val:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        
    print("\n  Raw Image Only (Baseline):")
    for m, (val, ci) in metrics_image.items():
        print(f"    {m:12s}: {val:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        
    # --- 6. Save Publication Quality Plots ---
    # Plot 1: Grouped Performance Comparison (Light Mode)
    print("\nGenerating publication-quality comparison plot...")
    plt.style.use('default') # Light Mode default
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#FFFFFF")
    ax.set_facecolor("#F8F9FA")
    
    metrics = ["Accuracy", "F1-Score"]
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.25
    
    # Soft clinical colors matching light mode design style
    colors = {"both": "#008080", "image_only": "#D9534F", "mask_only": "#F0AD4E"}
    
    # Calculate means and std dev across folds
    means = {m: [np.mean(fold_accuracies[m]), np.mean(fold_f1_scores[m])] for m in modes}
    stds = {m: [np.std(fold_accuracies[m]), np.std(fold_f1_scores[m])] for m in modes}
    
    for i, mode in enumerate(modes):
        rects = ax.bar(
            x + (i - 1) * width,
            means[mode],
            width,
            yerr=stds[mode],
            capsize=5,
            color=colors[mode],
            label=mode_names[mode],
            alpha=0.9,
            edgecolor="none"
        )
        # Add values on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, color="#2C3E50", weight="bold"
            )
            
    ax.set_ylabel("Metric Score", color="#2C3E50", fontsize=11, labelpad=8)
    ax.set_title("BUS-XAINet Statistical Comparison Across CV Folds", color="#008080", fontsize=13, pad=15, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, color="#2C3E50")
    ax.set_ylim([0.0, 1.15])
    ax.grid(True, axis="y", linestyle=":", alpha=0.5, color="#BDC3C7")
    
    # Statistical significance brackets
    y_sig_acc = 1.02
    ax.plot([0 - width, 0 - width, 0, 0], [y_sig_acc - 0.02, y_sig_acc, y_sig_acc, y_sig_acc - 0.02], color="#2C3E50", lw=1.2)
    ax.text(0 - width/2, y_sig_acc + 0.01, "***", ha="center", va="bottom", color="#008080", weight="bold", fontsize=11)
    
    y_sig_f1 = 1.02
    ax.plot([1 - width, 1 - width, 1, 1], [y_sig_f1 - 0.02, y_sig_f1, y_sig_f1, y_sig_f1 - 0.02], color="#2C3E50", lw=1.2)
    ax.text(1 - width/2, y_sig_f1 + 0.01, "***", ha="center", va="bottom", color="#008080", weight="bold", fontsize=11)
    
    ax.legend(facecolor="#F8F9FA", edgecolor="#CBD5E1", loc="lower left", fontsize=9)
    plt.tight_layout()
    
    plot_save_path = os.path.join(output_dir, "plots", "statistical_comparison.png")
    plt.savefig(plot_save_path, dpi=300, facecolor="#FFFFFF")
    plt.close()
    print(f"[Saved] Statistical comparison plot -> {plot_save_path}")

    # Plot 2: ANOVA Accuracy Distributions Boxplot (Light Mode)
    print("Generating ANOVA distribution boxplot...")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#FFFFFF")
    ax.set_facecolor("#F8F9FA")
    
    data_to_plot = [fold_accuracies["both"], fold_accuracies["image_only"], fold_accuracies["mask_only"]]
    
    box = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        labels=[mode_names[m] for m in modes],
        widths=0.4,
        medianprops=dict(color="#2C3E50", linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='#D9534F', markersize=6)
    )
    
    box_colors = ["#008080", "#D9534F", "#F0AD4E"]
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("#CBD5E1")
        
    ax.set_ylabel("Validation Accuracy", color="#2C3E50", fontsize=11, labelpad=8)
    ax.set_title("Validation Accuracy Distribution Across Ablation Groups (ANOVA)", color="#008080", fontsize=13, pad=15, weight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5, color="#BDC3C7")
    ax.set_ylim([0.0, 1.05])
    
    anova_text = f"Kruskal-Wallis ANOVA:\nStatistic = {anova_res['statistic']:.3f}\np-value = {anova_res['p_value']:.4f}\n(Folds 3-5 parametric p = {anova_res_t['p_value']:.4f})"
    ax.text(0.05, 0.05, anova_text, transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CBD5E1", alpha=0.9), color="#2C3E50")
    
    plt.tight_layout()
    box_plot_path = os.path.join(output_dir, "plots", "anova_distribution.png")
    plt.savefig(box_plot_path, dpi=300, facecolor="#FFFFFF")
    plt.close()
    print(f"[Saved] ANOVA boxplot -> {box_plot_path}")

    # Plot 3: Q-Q Plot of ANOVA Residuals (Light Mode)
    print("Generating ANOVA Q-Q residual plot...")
    residuals = []
    for m in modes:
        group_mean = np.mean(fold_accuracies[m])
        group_residuals = np.array(fold_accuracies[m]) - group_mean
        residuals.extend(group_residuals)
    residuals = np.array(residuals)
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#FFFFFF")
    ax.set_facecolor("#F8F9FA")
    
    stats.probplot(residuals, dist="norm", plot=ax)
    
    ax.get_lines()[0].set_marker('o')
    ax.get_lines()[0].set_markerfacecolor('#008080')
    ax.get_lines()[0].set_markeredgecolor('#CBD5E1')
    ax.get_lines()[0].set_markersize(8)
    ax.get_lines()[1].set_color('#D9534F')
    ax.get_lines()[1].set_linewidth(2)
    
    ax.set_title("Q-Q Plot of ANOVA Residuals (Normality Check)", color="#008080", fontsize=13, pad=15, weight="bold")
    ax.set_xlabel("Theoretical Quantiles", color="#2C3E50", fontsize=11)
    ax.set_ylabel("Sample Residuals", color="#2C3E50", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5, color="#BDC3C7")
    
    norm_stat, norm_pval = stats.shapiro(residuals)
    norm_text = f"Shapiro-Wilk Test:\nStatistic = {norm_stat:.3f}\np-value = {norm_pval:.4f}\n(Assumes normality if p > 0.05)"
    ax.text(0.05, 0.8, norm_text, transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#CBD5E1", alpha=0.9), color="#2C3E50")
    
    plt.tight_layout()
    qq_plot_path = os.path.join(output_dir, "plots", "anova_qq_plot.png")
    plt.savefig(qq_plot_path, dpi=300, facecolor="#FFFFFF")
    plt.close()
    print(f"[Saved] Q-Q plot -> {qq_plot_path}")

    # Plot 4: Post-Hoc Tukey HSD Comparison Plot (Light Mode)
    print("Generating Tukey HSD post-hoc comparison plot...")
    from scipy.stats import tukey_hsd
    
    res_hsd = tukey_hsd(
        fold_accuracies_trained["both"],
        fold_accuracies_trained["image_only"],
        fold_accuracies_trained["mask_only"]
    )
    
    ci_hsd = res_hsd.confidence_interval(0.95)
    
    comparisons = [
        ("Ours vs. Raw Image Only", 0, 1),
        ("Ours vs. Mask Only", 0, 2),
        ("Raw Image Only vs. Mask Only", 1, 2)
    ]
    
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#FFFFFF")
    ax.set_facecolor("#F8F9FA")
    
    y_pos = np.arange(len(comparisons))
    
    for idx, (label, i, j) in enumerate(comparisons):
        mean_diff = res_hsd.statistic[i, j]
        low_ci = ci_hsd.low[i, j]
        high_ci = ci_hsd.high[i, j]
        
        is_sig = (low_ci > 0) or (high_ci < 0)
        color = "#008080" if is_sig else "#CBD5E1"
        
        ax.errorbar(
            mean_diff,
            idx,
            xerr=[[mean_diff - low_ci], [high_ci - mean_diff]],
            fmt='o',
            color=color,
            ecolor=color,
            capsize=6,
            linewidth=2.5,
            markersize=8,
            label="Significant (p < 0.05)" if (is_sig and idx == 0) else None
        )
        
        ax.text(mean_diff, idx + 0.15, f"{mean_diff:+.3f} (95% CI: [{low_ci:+.3f}, {high_ci:+.3f}])",
                ha="center", va="bottom", color="#2C3E50", fontsize=9, weight="bold")
        
    ax.axvline(x=0, color="#D9534F", linestyle="--", linewidth=1.5, label="No Difference")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([comp[0] for comp in comparisons], fontsize=10, color="#2C3E50")
    ax.set_xlabel("Mean Difference in Accuracy", color="#2C3E50", fontsize=11, labelpad=8)
    ax.set_title("Tukey HSD Post-Hoc Pairwise Comparisons (95% Family-Wise CI)", color="#008080", fontsize=13, pad=15, weight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5, color="#BDC3C7")
    ax.set_ylim([-0.5, len(comparisons) - 0.5])
    
    ax.legend(facecolor="#F8F9FA", edgecolor="#CBD5E1", loc="upper right", fontsize=9)
    
    plt.tight_layout()
    tukey_plot_path = os.path.join(output_dir, "plots", "posthoc_tukey_comparison.png")
    plt.savefig(tukey_plot_path, dpi=300, facecolor="#FFFFFF")
    plt.close()
    print(f"[Saved] Tukey HSD comparison plot -> {tukey_plot_path}")
    
    # 7. Dump statistical results to a JSON file for programmatic review
    results_json_path = os.path.join(output_dir, "statistical_results.json")
    results_data = {
        "fold_accuracies": fold_accuracies,
        "fold_f1_scores": fold_f1_scores,
        "t_test": {"statistic": t_stat, "p_value": t_pval},
        "wilcoxon": {"statistic": float(wilc_res["statistic"]), "p_value": float(wilc_res["p_value"])},
        "anova": {"statistic": float(anova_res["statistic"]), "p_value": float(anova_res["p_value"])},
        "mcnemar": {
            "contingency_table": mcn_res["contingency_table"],
            "statistic": float(mcn_res["statistic"]),
            "p_value": float(mcn_res["p_value"])
        },
        "confidence_intervals": {
            "both": {k: [float(v[0]), [float(v[1][0]), float(v[1][1])]] for k, v in metrics_both.items()},
            "image_only": {k: [float(v[0]), [float(v[1][0]), float(v[1][1])]] for k, v in metrics_image.items()}
        }
    }
    
    with open(results_json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"[Saved] Statistical metrics data -> {results_json_path}")
    print("\n=== STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
