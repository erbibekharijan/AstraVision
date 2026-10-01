"""
Evaluation metrics and plotting utilities for ASTRA VISION.
Computes Accuracy, Macro & Per-class Precision/Recall/F1, Confusion Matrix,
Top-2/Top-3 Accuracy, Expected Calibration Error (ECE), and Bootstrap 95% CIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE):
    ECE = sum_b (N_b / N) * |acc(b) - conf(b)|
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_bootstrap_accuracy_ci(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap 95% Confidence Interval for accuracy on small test sets.
    """
    rng = np.random.RandomState(seed)
    boot_accs = []
    n = len(labels)
    if n == 0:
        return (0.0, 0.0)

    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        acc = np.mean(predictions[indices] == labels[indices])
        boot_accs.append(acc)

    lower_pct = (1.0 - ci) / 2.0 * 100.0
    upper_pct = (1.0 + ci) / 2.0 * 100.0
    return float(np.percentile(boot_accs, lower_pct)), float(np.percentile(boot_accs, upper_pct))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    ci_lower, ci_upper = compute_bootstrap_accuracy_ci(y_pred, y_true)

    # Macro & Weighted metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(p_per_class[i]),
            "recall": float(r_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support[i]),
        }

    # Top-K accuracy
    top2_hits = 0
    top3_hits = 0
    for i in range(len(y_true)):
        top_k_indices = np.argsort(y_probs[i])[::-1]
        if y_true[i] in top_k_indices[:2]:
            top2_hits += 1
        if y_true[i] in top_k_indices[:3]:
            top3_hits += 1

    top2_acc = float(top2_hits / len(y_true)) if len(y_true) > 0 else 0.0
    top3_acc = float(top3_hits / len(y_true)) if len(y_true) > 0 else 0.0

    # Calibration error
    confidences = np.max(y_probs, axis=-1)
    ece = compute_expected_calibration_error(confidences, y_pred, y_true)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize="true")

    return {
        "accuracy": acc,
        "accuracy_ci_95": [ci_lower, ci_upper],
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "top2_accuracy": top2_acc,
        "top3_accuracy": top3_acc,
        "ece": ece,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """
    Renders and saves confusion matrix heatmap using seaborn.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))

    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Ground Truth", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
