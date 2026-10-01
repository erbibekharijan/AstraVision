#!/usr/bin/env python3
"""
Temperature scaling calibration and threshold selection script for Model A.
Fits single positive scalar temperature T on the held-out validation set
to align softmax probabilities with empirical accuracy and minimize ECE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from defence_recog.config import AppConfig, load_config
from defence_recog.data import DefenceDataset, get_eval_transforms
from defence_recog.metrics import compute_expected_calibration_error
from defence_recog.models import load_model_a


def nll_criterion(temp: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Negative Log Likelihood loss for temperature parameter.
    """
    scaled = logits / max(temp, 1e-4)
    exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    probs = exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)

    # Pick probability of ground truth label
    row_indices = np.arange(len(labels))
    truth_probs = np.clip(probs[row_indices, labels], 1e-12, 1.0)
    return -float(np.mean(np.log(truth_probs)))


def run_calibration(config_file: str = "config.yaml") -> int:
    cfg = load_config(config_file)
    device = cfg.get_device()

    print("=" * 70)
    print("ASTRA VISION — TEMPERATURE SCALING & THRESHOLD CALIBRATION")
    print(f"Device: {device}")
    print("=" * 70)

    val_csv = cfg.paths.val_split
    if not val_csv.exists():
        print(f"Error: Validation split not found at {val_csv}. Run 'make splits' first.")
        return 1

    try:
        model, id2label, label2id = load_model_a(cfg.paths.model_a_checkpoint, device=device)
    except Exception as e:
        print(f"Error loading Model A: {e}")
        return 1

    dataset = DefenceDataset(
        csv_file=val_csv,
        data_root=cfg.paths.data_dir,
        transform=get_eval_transforms(cfg.image_size),
        label2id=label2id,
    )

    print(f"Extracting validation logits for {len(dataset)} samples...")
    raw_logits_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            tensor, label, _ = dataset[i]
            tensor = tensor.unsqueeze(0).to(device)
            out = model(tensor)
            logits = out.logits.cpu().numpy()[0]
            raw_logits_list.append(logits)
            labels_list.append(label)

    raw_logits = np.array(raw_logits_list)
    labels = np.array(labels_list)

    # Compute uncalibrated metrics (T = 1.0)
    uncal_probs = F.softmax(torch.tensor(raw_logits), dim=-1).numpy()
    uncal_preds = np.argmax(uncal_probs, axis=-1)
    uncal_conf = np.max(uncal_probs, axis=-1)
    uncal_ece = compute_expected_calibration_error(uncal_conf, uncal_preds, labels)
    uncal_nll = nll_criterion(1.0, raw_logits, labels)

    print("\n[Before Calibration (T = 1.0)]:")
    print(f"  - Validation NLL: {uncal_nll:.4f}")
    print(f"  - Validation ECE: {uncal_ece:.4f}")

    # Optimize Temperature T > 0 using scipy minimize (Nelder-Mead / L-BFGS-B)
    res = minimize(
        fun=lambda t: nll_criterion(t[0], raw_logits, labels),
        x0=[1.0],
        bounds=[(0.05, 10.0)],
        method="L-BFGS-B",
    )
    optimal_temp = float(res.x[0])

    # Compute calibrated metrics
    cal_probs = F.softmax(torch.tensor(raw_logits / optimal_temp), dim=-1).numpy()
    cal_preds = np.argmax(cal_probs, axis=-1)
    cal_conf = np.max(cal_probs, axis=-1)
    cal_ece = compute_expected_calibration_error(cal_conf, cal_preds, labels)
    cal_nll = nll_criterion(optimal_temp, raw_logits, labels)

    print(f"\n[After Calibration (T = {optimal_temp:.4f})]:")
    print(f"  - Calibrated NLL: {cal_nll:.4f}")
    print(f"  - Calibrated ECE: {cal_ece:.4f} (Delta: {cal_ece - uncal_ece:+.4f})")

    # Determine uncertainty thresholds
    # Confidence threshold: 10th percentile of correct predictions
    correct_mask = (cal_preds == labels)
    if correct_mask.sum() > 0:
        correct_confs = cal_conf[correct_mask]
        suggested_conf_thresh = float(np.percentile(correct_confs, 10))
        suggested_conf_thresh = max(0.50, min(0.75, suggested_conf_thresh))
    else:
        suggested_conf_thresh = 0.60

    # Margin threshold: margin between top1 and top2
    sorted_cal_probs = np.sort(cal_probs, axis=-1)[:, ::-1]
    margins = sorted_cal_probs[:, 0] - sorted_cal_probs[:, 1]
    correct_margins = margins[correct_mask] if correct_mask.sum() > 0 else margins
    suggested_margin_thresh = float(np.percentile(correct_margins, 10))
    suggested_margin_thresh = max(0.10, min(0.25, suggested_margin_thresh))

    print("\nRecommended Operational Thresholds:")
    print(f"  - Confidence Threshold: {suggested_conf_thresh*100:.1f}%")
    print(f"  - Margin Threshold:     {suggested_margin_thresh*100:.1f}%")

    # Save to models/calibration.json
    output_path = cfg.paths.calibration_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimal_temperature": round(optimal_temp, 4),
        "uncalibrated_ece": round(uncal_ece, 4),
        "calibrated_ece": round(cal_ece, 4),
        "uncalibrated_nll": round(uncal_nll, 4),
        "calibrated_nll": round(cal_nll, 4),
        "suggested_confidence_threshold": round(suggested_conf_thresh, 3),
        "suggested_margin_threshold": round(suggested_margin_thresh, 3),
        "num_val_samples": len(labels),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nCalibration artifact saved to {output_path}")
    return 0


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    sys.exit(run_calibration(cfg_arg))
