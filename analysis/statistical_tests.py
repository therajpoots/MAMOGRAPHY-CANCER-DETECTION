"""
Statistical Analysis Module
===========================
Implements 6 statistical tests to evaluate the models and preprocessing pipelines:
1. One-Way ANOVA (Compare accuracy across preprocessing variants)
2. Wilcoxon Signed-Rank Test (Paired comparison across CV folds)
3. McNemar's Test (Pairwise classification disagreement)
4. DeLong's Test (Compare dependent ROC curves / AUCs)
5. Friedman Test (Compare multiple models across folds)
6. Pearson Correlation (Dice score vs. classification accuracy)

Also includes:
- Shapiro-Wilk test for normality
- Bonferroni correction helper
- Confidence intervals for AUC and metrics
"""

import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Any, Optional


# ==========================================
# 1. Shapiro-Wilk Normality Check
# ==========================================
def check_normality(data: np.ndarray) -> Tuple[float, bool]:
    """
    Shapiro-Wilk test for normality.
    Returns: (p_value, is_normal)
    """
    if len(data) < 3:
        return 1.0, True
    stat, p = stats.shapiro(data)
    return p, p > 0.05


# ==========================================
# 2. Bonferroni Correction Helper
# ==========================================
def bonferroni_correct(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Applies Bonferroni correction to a list of p-values.
    Returns: (corrected_p_values, significant_flags)
    """
    n = len(p_values)
    corrected_p = [min(1.0, p * n) for p in p_values]
    significant = [p_corr < alpha for p_corr in corrected]
    return corrected_p, significant


# ==========================================
# 3. One-Way ANOVA
# ==========================================
def run_anova(*groups: np.ndarray) -> Dict[str, Any]:
    """
    One-Way ANOVA to compare accuracy/F1 across 3+ groups (e.g. preprocessing variants).
    Checks normality first. If non-normal, runs Kruskal-Wallis.
    """
    normality_p = [check_normality(g)[0] for g in groups]
    all_normal = all(p > 0.05 for p in normality_p)
    
    if all_normal:
        stat, p = stats.f_oneway(*groups)
        test_used = "One-Way ANOVA (parametric)"
    else:
        stat, p = stats.kruskal(*groups)
        test_used = "Kruskal-Wallis H-test (non-parametric)"
        
    return {
        "test_name": test_used,
        "statistic": stat,
        "p_value": p,
        "normality_p_values": normality_p,
        "significant": p < 0.05
    }


# ==========================================
# 4. Wilcoxon Signed-Rank Test
# ==========================================
def run_wilcoxon(model_a_scores: np.ndarray, model_b_scores: np.ndarray) -> Dict[str, Any]:
    """
    Paired comparison of two models' performance across K folds.
    Checks normality of differences. If normal, runs paired t-test; else Wilcoxon.
    """
    diff = model_a_scores - model_b_scores
    p_norm, is_normal = check_normality(diff)
    
    if is_normal:
        stat, p = stats.ttest_rel(model_a_scores, model_b_scores)
        test_used = "Paired t-test (parametric)"
    else:
        stat, p = stats.wilcoxon(model_a_scores, model_b_scores)
        test_used = "Wilcoxon Signed-Rank Test (non-parametric)"
        
    return {
        "test_name": test_used,
        "statistic": stat,
        "p_value": p,
        "difference_normality_p": p_norm,
        "significant": p < 0.05
    }


# ==========================================
# 5. McNemar's Test
# ==========================================
def run_mcnemar(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, Any]:
    """
    McNemar's test to compare paired classification predictions.
    Contingency table:
                Model B Correct | Model B Incorrect
    Model A Correct     [0,0]             [0,1]
    Model A Incorrect   [1,0]             [1,1]
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    
    # Contingency cells
    n00 = np.sum(correct_a & correct_b)
    n01 = np.sum(correct_a & ~correct_b)  # A correct, B incorrect
    n10 = np.sum(~correct_a & correct_b)  # A incorrect, B correct
    n11 = np.sum(~correct_a & ~correct_b)
    
    # If off-diagonal sum is small (< 25), use binomial exact test
    off_diag_sum = n01 + n10
    if off_diag_sum < 25:
        # Binomial exact test (two-sided)
        p = 2 * stats.binom.cdf(min(n01, n10), off_diag_sum, 0.5)
        # Handle edge case where min is 0 or equal
        if off_diag_sum == 0:
            p = 1.0
        p = min(1.0, p)
        test_used = "McNemar Exact Binomial Test"
        stat = min(n01, n10)
    else:
        # Chi-square approximation with Edwards' continuity correction
        stat = ((abs(n01 - n10) - 1.0) ** 2) / (n01 + n10)
        p = stats.chi2.sf(stat, df=1)
        test_used = "McNemar Chi-Square Test (with continuity correction)"
        
    return {
        "test_name": test_used,
        "contingency_table": [[int(n00), int(n01)], [int(n10), int(n11)]],
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < 0.05
    }


# ==========================================
# 6. DeLong's Test for ROC Curves
# ==========================================
def run_delong_test(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray) -> Dict[str, Any]:
    """
    DeLong's test to compare the AUC-ROC of two models on the same dataset.
    Reference: DeLong et al., (1988). http://www.jstor.org/stable/2531595
    """
    y_true = np.array(y_true)
    y_prob_a = np.array(y_prob_a)
    y_prob_b = np.array(y_prob_b)
    
    # Check binary classification
    unique_classes = np.unique(y_true)
    if len(unique_classes) != 2:
        # For multi-class, we check macro-averaged or target-vs-rest.
        # Here we default to positive vs negative classes.
        # Binarize labels: assume the last class is positive, others negative
        pos_class = unique_classes[-1]
        y_true = (y_true == pos_class).astype(int)
        
    # Split predictions by true label
    neg_indices = (y_true == 0)
    pos_indices = (y_true == 1)
    
    m = np.sum(neg_indices)
    n = np.sum(pos_indices)
    
    if m == 0 or n == 0:
        raise ValueError("Ground truth must contain both classes to compute AUC.")
        
    x_a = y_prob_a[neg_indices]
    y_a = y_prob_a[pos_indices]
    
    x_b = y_prob_b[neg_indices]
    y_b = y_prob_b[pos_indices]
    
    # Mann-Whitney kernel function
    def psi(x, y):
        # Broadcast comparison: x (m, 1) and y (1, n)
        return (y > x).astype(float) + 0.5 * (y == x).astype(float)
        
    # Calculate structural components
    psi_a = psi(x_a[:, None], y_a[None, :])  # shape (m, n)
    v10_a = np.mean(psi_a, axis=1)  # average over positive samples (m,)
    v01_a = np.mean(psi_a, axis=0)  # average over negative samples (n,)
    
    psi_b = psi(x_b[:, None], y_b[None, :])
    v10_b = np.mean(psi_b, axis=1)
    v01_b = np.mean(psi_b, axis=0)
    
    # Compute AUCs
    auc_a = np.mean(v10_a)
    auc_b = np.mean(v10_b)
    
    # Compute covariances S10 and S01
    v10_stack = np.vstack([v10_a, v10_b])
    s10 = np.cov(v10_stack)
    
    v01_stack = np.vstack([v01_a, v01_b])
    s01 = np.cov(v01_stack)
    
    # DeLong covariance matrix: S = S10 / m + S01 / n
    S = s10 / m + s01 / n
    
    # Test statistic z = (auc_a - auc_b) / sqrt(var(auc_a - auc_b))
    # var(auc_a - auc_b) = S[0,0] + S[1,1] - 2*S[0,1]
    variance = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    
    if variance <= 0:
        z = 0.0
        p = 1.0
    else:
        z = (auc_a - auc_b) / np.sqrt(variance)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        
    # Standard errors
    se_a = np.sqrt(S[0, 0])
    se_b = np.sqrt(S[1, 1])
    
    # 95% Confidence Intervals
    ci_a = (max(0.0, auc_a - 1.96 * se_a), min(1.0, auc_a + 1.96 * se_a))
    ci_b = (max(0.0, auc_b - 1.96 * se_b), min(1.0, auc_b + 1.96 * se_b))
    
    return {
        "test_name": "DeLong's ROC Comparison Test",
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "se_a": float(se_a),
        "se_b": float(se_b),
        "ci_a": ci_a,
        "ci_b": ci_b,
        "z_statistic": float(z),
        "p_value": float(p),
        "significant": p < 0.05
    }


# ==========================================
# 7. Friedman Test
# ==========================================
def run_friedman(*model_scores: np.ndarray) -> Dict[str, Any]:
    """
    Compare 3+ models across k folds.
    Input: lists of scores (each of length K) for each model.
    """
    stat, p = stats.friedmanchisquare(*model_scores)
    return {
        "test_name": "Friedman Test (multiple dependent samples)",
        "statistic": stat,
        "p_value": p,
        "significant": p < 0.05
    }


# ==========================================
# 8. Pearson Correlation
# ==========================================
def run_correlation(metric_a: np.ndarray, metric_b: np.ndarray) -> Dict[str, Any]:
    """
    Pearson and Spearman correlations between two metrics (e.g. Dice vs. Accuracy).
    Checks normality to see if Pearson is valid.
    """
    p_norm_a, is_norm_a = check_normality(metric_a)
    p_norm_b, is_norm_b = check_normality(metric_b)
    
    # Pearson
    pearson_r, pearson_p = stats.pearsonr(metric_a, metric_b)
    # Spearman (rank correlation, non-parametric)
    spearman_r, spearman_p = stats.spearmanr(metric_a, metric_b)
    
    is_valid_pearson = is_norm_a and is_norm_b
    
    return {
        "test_name": "Pearson & Spearman Correlation",
        "pearson": {"r": pearson_r, "p_value": pearson_p, "valid": is_valid_pearson},
        "spearman": {"r": spearman_r, "p_value": spearman_p, "valid": True},
        "normality_p_a": p_norm_a,
        "normality_p_b": p_norm_b
    }


# ==========================================
# Self-Test Script
# ==========================================
if __name__ == "__main__":
    print("=== Testing Statistical Analysis Module ===")
    
    # Generate mock data
    np.random.seed(42)
    folds = 10
    
    # Mock cross-validation accuracies for 3 configurations
    config_1 = np.random.normal(loc=0.82, scale=0.03, size=folds)
    config_2 = np.random.normal(loc=0.85, scale=0.02, size=folds)
    config_3 = np.random.normal(loc=0.88, scale=0.02, size=folds)
    
    print("\n1. Testing ANOVA/Kruskal-Wallis:")
    anova_res = run_anova(config_1, config_2, config_3)
    print(f"Used: {anova_res['test_name']}")
    print(f"p-value: {anova_res['p_value']:.4f} (Significant: {anova_res['significant']})")
    
    print("\n2. Testing Wilcoxon/Paired t-test (Config 2 vs Config 3):")
    wilcoxon_res = run_wilcoxon(config_2, config_3)
    print(f"Used: {wilcoxon_res['test_name']}")
    print(f"p-value: {wilcoxon_res['p_value']:.4f} (Significant: {wilcoxon_res['significant']})")
    
    print("\n3. Testing McNemar's Test:")
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1])
    y_pred_a = np.array([1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1])
    y_pred_b = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1]) # better model
    mcnemar_res = run_mcnemar(y_true, y_pred_a, y_pred_b)
    print(f"Used: {mcnemar_res['test_name']}")
    print(f"Contingency: {mcnemar_res['contingency_table']}")
    print(f"p-value: {mcnemar_res['p_value']:.4f} (Significant: {mcnemar_res['significant']})")
    
    print("\n4. Testing DeLong's ROC Test:")
    # Mock ground truth and probability outputs
    y_true_roc = np.concatenate([np.zeros(50), np.ones(50)])
    y_prob_a = np.concatenate([np.random.beta(2, 5, 50), np.random.beta(5, 2, 50)]) # AUC ~ 0.8
    y_prob_b = y_prob_a + np.random.normal(0.05, 0.02, 100) # Slightly better, dependent
    y_prob_b = np.clip(y_prob_b, 0.0, 1.0)
    
    delong_res = run_delong_test(y_true_roc, y_prob_a, y_prob_b)
    print(f"Used: {delong_res['test_name']}")
    print(f"AUC Model A: {delong_res['auc_a']:.4f} (CI: {delong_res['ci_a'][0]:.3f} - {delong_res['ci_a'][1]:.3f})")
    print(f"AUC Model B: {delong_res['auc_b']:.4f} (CI: {delong_res['ci_b'][0]:.3f} - {delong_res['ci_b'][1]:.3f})")
    print(f"p-value: {delong_res['p_value']:.4f} (Significant: {delong_res['significant']})")
    
    print("\n5. Testing Friedman Test:")
    friedman_res = run_friedman(config_1, config_2, config_3)
    print(f"Used: {friedman_res['test_name']}")
    print(f"p-value: {friedman_res['p_value']:.4f} (Significant: {friedman_res['significant']})")
    
    print("\n6. Testing Correlation (Config 1 vs Config 2):")
    corr_res = run_correlation(config_1, config_2)
    print(f"Pearson r: {corr_res['pearson']['r']:.4f} (p={corr_res['pearson']['p_value']:.4f})")
    print(f"Spearman r: {corr_res['spearman']['r']:.4f} (p={corr_res['spearman']['p_value']:.4f})")
