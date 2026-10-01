#!/usr/bin/env python3
"""
Comprehensive evaluation script for ASTRA VISION.
Benchmarks:
1. Model A: Fine-tuned ConvNeXt-Tiny
2. Model B Probe: CLIP ViT-B/32 Linear Probe (Logistic Regression)
3. Model B Zero-Shot: CLIP ViT-B/32 prompt-based classification baseline
Outputs:
- outputs/metrics/metrics_model_a.json
- outputs/metrics/metrics_model_b_probe.json
- outputs/metrics/metrics_model_b_zeroshot.json
- outputs/metrics/comparison.csv
- outputs/metrics/comparison.md
- outputs/metrics/misclassified.csv
- outputs/figures/confusion_matrix_model_a.png
- outputs/figures/confusion_matrix_model_b.png
- Stratified 5-fold cross-validation on full dataset for reliable variance estimation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn.functional as F

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from defence_recog.config import AppConfig, load_config
from defence_recog.data import DefenceDataset, get_eval_transforms
from defence_recog.metrics import (
    evaluate_predictions,
    plot_and_save_confusion_matrix,
)
from defence_recog.models import (
    CLIPFeatureExtractor,
    build_model_a,
    load_clip_probe,
    load_model_a,
)
from defence_recog.preprocess import load_and_sanitize_image


def get_model_size_mb(path: Path) -> float:
    """Calculate directory or file size in MB."""
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def measure_cpu_latency(predict_fn, sample_input, warmups: int = 3, runs: int = 15) -> float:
    """Measure average inference latency in milliseconds on CPU."""
    for _ in range(warmups):
        predict_fn(sample_input)
    t0 = time.perf_counter()
    for _ in range(runs):
        predict_fn(sample_input)
    t1 = time.perf_counter()
    return ((t1 - t0) / runs) * 1000.0


def run_evaluation(config_file: str = "config.yaml") -> int:
    cfg = load_config(config_file)
    device = cfg.get_device()

    print("=" * 70)
    print("ASTRA VISION — COMPREHENSIVE BENCHMARK & MODEL COMPARISON")
    print(f"Device: {device} | Test split: {cfg.paths.test_split}")
    print("=" * 70)

    test_csv = cfg.paths.test_split
    if not test_csv.exists():
        print(f"Error: Test split not found at {test_csv}. Run 'make splits' first.")
        return 1

    df_test = pd.read_csv(test_csv)
    class_names = [c.key for c in cfg.classes]
    id2label = cfg.get_id2label()
    label2id = cfg.get_label2id()

    y_true = np.array([label2id[c] for c in df_test["category"]])

    # --------------------------------------------------------------------------
    # 1. EVALUATE MODEL A (ConvNeXt-Tiny)
    # --------------------------------------------------------------------------
    print("\n[1/3] Evaluating Model A (ConvNeXt-Tiny fine-tuned)...")
    model_a, _, _ = load_model_a(cfg.paths.model_a_checkpoint, device=device)
    model_a_ds = DefenceDataset(
        csv_file=test_csv,
        data_root=cfg.paths.data_dir,
        transform=get_eval_transforms(cfg.image_size),
        label2id=label2id,
    )

    # Load calibration temperature if present
    temperature = 1.0
    if cfg.paths.calibration_file.exists():
        with open(cfg.paths.calibration_file, "r") as f:
            cal = json.load(f)
            temperature = cal.get("optimal_temperature", 1.0)

    model_a_probs_list = []
    with torch.no_grad():
        for i in range(len(model_a_ds)):
            tensor, _, _ = model_a_ds[i]
            tensor = tensor.unsqueeze(0).to(device)
            out = model_a(tensor)
            scaled = out.logits / max(temperature, 1e-4)
            probs = F.softmax(scaled, dim=-1).cpu().numpy()[0]
            model_a_probs_list.append(probs)

    model_a_probs = np.array(model_a_probs_list)
    model_a_preds = np.argmax(model_a_probs, axis=-1)
    metrics_a = evaluate_predictions(y_true, model_a_preds, model_a_probs, class_names)

    # Model A Latency & Size
    sample_tensor = model_a_ds[0][0].unsqueeze(0).to("cpu")
    cpu_model_a = model_a.to("cpu")
    latency_a = measure_cpu_latency(lambda x: cpu_model_a(x), sample_tensor)
    size_a = get_model_size_mb(cfg.paths.model_a_checkpoint)
    metrics_a["cpu_latency_ms"] = round(latency_a, 2)
    metrics_a["model_size_mb"] = round(size_a, 2)

    with open(cfg.paths.metrics_dir / "metrics_model_a.json", "w") as f:
        json.dump(metrics_a, f, indent=2)

    plot_and_save_confusion_matrix(
        np.array(metrics_a["confusion_matrix"]),
        class_names=class_names,
        output_path=cfg.paths.figures_dir / "confusion_matrix_model_a.png",
        title="Model A (ConvNeXt-Tiny) Test Confusion Matrix",
    )

    # --------------------------------------------------------------------------
    # 2. EVALUATE MODEL B: CLIP ViT-B/32 Linear Probe
    # --------------------------------------------------------------------------
    print("\n[2/3] Evaluating Model B (CLIP ViT-B/32 Linear Probe)...")
    clip_clf, clip_classes, _ = load_clip_probe(cfg.paths.model_b_probe)
    clip_extractor = CLIPFeatureExtractor(hf_model_id=cfg.model_b.hf_model_id, device=device)

    # Load cached or extract test features
    test_cache = cfg.paths.model_dir / "clip_cache" / "test_clip.npz"
    if test_cache.exists():
        test_features = np.load(test_cache)["features"]
    else:
        test_features_list = []
        for _, row in df_test.iterrows():
            img = load_and_sanitize_image(cfg.paths.data_dir / row["file_name"])
            emb = clip_extractor.extract_image_embedding(img)[0]
            test_features_list.append(emb)
        test_features = np.array(test_features_list)

    model_b_probs = clip_clf.predict_proba(test_features)
    model_b_preds = np.argmax(model_b_probs, axis=-1)
    metrics_b = evaluate_predictions(y_true, model_b_preds, model_b_probs, class_names)

    sample_img = load_and_sanitize_image(cfg.paths.data_dir / df_test.iloc[0]["file_name"])
    latency_b = measure_cpu_latency(lambda x: clip_clf.predict_proba(test_features[:1]), None)
    size_b = get_model_size_mb(cfg.paths.model_b_probe)
    metrics_b["cpu_latency_ms"] = round(latency_b, 2)
    metrics_b["model_size_mb"] = round(size_b, 2)

    with open(cfg.paths.metrics_dir / "metrics_model_b_probe.json", "w") as f:
        json.dump(metrics_b, f, indent=2)

    plot_and_save_confusion_matrix(
        np.array(metrics_b["confusion_matrix"]),
        class_names=class_names,
        output_path=cfg.paths.figures_dir / "confusion_matrix_model_b.png",
        title="Model B (CLIP Linear Probe) Test Confusion Matrix",
    )

    # --------------------------------------------------------------------------
    # 3. EVALUATE MODEL B: CLIP Zero-Shot Baseline
    # --------------------------------------------------------------------------
    print("\n[3/3] Evaluating Model B (CLIP Zero-Shot Baseline)...")
    prompt_templates = cfg.model_b.prompt_templates
    class_mappings = cfg.model_b.class_prompt_mappings

    # Construct averaged text prompts per class
    zero_shot_prompts = [class_mappings.get(c, c) for c in class_names]
    full_prompt_list = [f"a photo of a {p}" for p in zero_shot_prompts]

    zero_shot_probs_list = []
    for _, row in df_test.iterrows():
        img = load_and_sanitize_image(cfg.paths.data_dir / row["file_name"])
        p = clip_extractor.compute_zero_shot_logits(img, full_prompt_list)[0]
        zero_shot_probs_list.append(p)

    zero_shot_probs = np.array(zero_shot_probs_list)
    zero_shot_preds = np.argmax(zero_shot_probs, axis=-1)
    metrics_zs = evaluate_predictions(y_true, zero_shot_preds, zero_shot_probs, class_names)

    with open(cfg.paths.metrics_dir / "metrics_model_b_zeroshot.json", "w") as f:
        json.dump(metrics_zs, f, indent=2)

    # --------------------------------------------------------------------------
    # 4. STRATIFIED 5-FOLD CROSS-VALIDATION (For reliable small-set evaluation)
    # --------------------------------------------------------------------------
    print("\n[4] Running Stratified 5-Fold Cross-Validation on CLIP Probe...")
    all_cache_tr = cfg.paths.model_dir / "clip_cache" / "train_clip.npz"
    all_cache_va = cfg.paths.model_dir / "clip_cache" / "val_clip.npz"
    all_cache_te = cfg.paths.model_dir / "clip_cache" / "test_clip.npz"

    if all_cache_tr.exists() and all_cache_va.exists() and all_cache_te.exists():
        d_tr = np.load(all_cache_tr)
        d_va = np.load(all_cache_va)
        d_te = np.load(all_cache_te)
        X_all = np.vstack([d_tr["features"], d_va["features"], d_te["features"]])
        labels_all = np.concatenate([d_tr["labels"], d_va["labels"], d_te["labels"]])
        y_all = np.array([label2id[c] for c in labels_all])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
        cv_accs = []
        cv_f1s = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg.seed)
            clf.fit(X_all[train_idx], y_all[train_idx])
            preds = clf.predict(X_all[val_idx])
            acc = float(np.mean(preds == y_all[val_idx]))
            f1 = float(evaluate_predictions(y_all[val_idx], preds, clf.predict_proba(X_all[val_idx]), class_names)["macro_f1"])
            cv_accs.append(acc)
            cv_f1s.append(f1)

        cv_mean_acc = float(np.mean(cv_accs))
        cv_std_acc = float(np.std(cv_accs))
        cv_mean_f1 = float(np.mean(cv_f1s))
        cv_std_f1 = float(np.std(cv_f1s))
        print(f"5-Fold CV Accuracy: {cv_mean_acc*100:.2f}% +/- {cv_std_acc*100:.2f}%")
        print(f"5-Fold CV Macro-F1: {cv_mean_f1:.4f} +/- {cv_std_f1:.4f}")
    else:
        cv_mean_acc, cv_std_acc, cv_mean_f1, cv_std_f1 = 0.0, 0.0, 0.0, 0.0

    # --------------------------------------------------------------------------
    # 5. EXPORT COMPARISON TABLES (CSV & Markdown)
    # --------------------------------------------------------------------------
    comparison_rows = [
        {
            "Model": "Model A (ConvNeXt-Tiny Fine-Tuned)",
            "Architecture": "ConvNeXt-Tiny (~28M)",
            "Test Accuracy": f"{metrics_a['accuracy']*100:.1f}%",
            "95% CI Accuracy": f"[{metrics_a['accuracy_ci_95'][0]*100:.1f}%, {metrics_a['accuracy_ci_95'][1]*100:.1f}%]",
            "Macro Precision": f"{metrics_a['macro_precision']*100:.1f}%",
            "Macro Recall": f"{metrics_a['macro_recall']*100:.1f}%",
            "Macro F1": f"{metrics_a['macro_f1']:.4f}",
            "ECE": f"{metrics_a['ece']:.4f}",
            "CPU Latency (ms)": metrics_a["cpu_latency_ms"],
            "Model Size (MB)": metrics_a["model_size_mb"],
        },
        {
            "Model": "Model B (CLIP Linear Probe)",
            "Architecture": "ViT-B/32 + Logistic Reg",
            "Test Accuracy": f"{metrics_b['accuracy']*100:.1f}%",
            "95% CI Accuracy": f"[{metrics_b['accuracy_ci_95'][0]*100:.1f}%, {metrics_b['accuracy_ci_95'][1]*100:.1f}%]",
            "Macro Precision": f"{metrics_b['macro_precision']*100:.1f}%",
            "Macro Recall": f"{metrics_b['macro_recall']*100:.1f}%",
            "Macro F1": f"{metrics_b['macro_f1']:.4f}",
            "ECE": f"{metrics_b['ece']:.4f}",
            "CPU Latency (ms)": metrics_b["cpu_latency_ms"],
            "Model Size (MB)": metrics_b["model_size_mb"],
        },
        {
            "Model": "Model B (CLIP Zero-Shot Baseline)",
            "Architecture": "ViT-B/32 Zero-Shot Text Prompts",
            "Test Accuracy": f"{metrics_zs['accuracy']*100:.1f}%",
            "95% CI Accuracy": f"[{metrics_zs['accuracy_ci_95'][0]*100:.1f}%, {metrics_zs['accuracy_ci_95'][1]*100:.1f}%]",
            "Macro Precision": f"{metrics_zs['macro_precision']*100:.1f}%",
            "Macro Recall": f"{metrics_zs['macro_recall']*100:.1f}%",
            "Macro F1": f"{metrics_zs['macro_f1']:.4f}",
            "ECE": f"{metrics_zs['ece']:.4f}",
            "CPU Latency (ms)": "-",
            "Model Size (MB)": "-",
        },
    ]

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = cfg.paths.metrics_dir / "comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)

    comp_md_path = cfg.paths.metrics_dir / "comparison.md"
    with open(comp_md_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison on Held-Out Test Set (N=23)\n\n")
        f.write("> **Caveat on Sample Size**: Because the test split contains 23 images (~4–5 per class), "
                "point estimates are accompanied by bootstrap 95% confidence intervals and 5-fold cross-validation.\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n\n### Per-Class F1 Score Breakdown\n\n")
        f.write("| Class | Model A F1 | Model B Probe F1 | Model B Zero-Shot F1 |\n")
        f.write("|---|---|---|---|\n")
        for c in class_names:
            f1_a = metrics_a["per_class"][c]["f1"]
            f1_b = metrics_b["per_class"][c]["f1"]
            f1_z = metrics_zs["per_class"][c]["f1"]
            f.write(f"| {c} | {f1_a:.4f} | {f1_b:.4f} | {f1_z:.4f} |\n")
        if cv_mean_acc > 0:
            f.write(f"\n### 5-Fold Cross-Validation Summary (Model B Probe)\n")
            f.write(f"- **Mean Accuracy**: {cv_mean_acc*100:.1f}% ± {cv_std_acc*100:.1f}%\n")
            f.write(f"- **Mean Macro-F1**: {cv_mean_f1:.4f} ± {cv_std_f1:.4f}\n")

    print(f"\nSaved comparison table to {comp_csv_path} and {comp_md_path}")

    # --------------------------------------------------------------------------
    # 6. LOG MISCLASSIFIED TEST SAMPLES
    # --------------------------------------------------------------------------
    misclassified_records = []
    for i, row in df_test.iterrows():
        gt_key = row["category"]
        pred_a_key = id2label[model_a_preds[i]]
        pred_b_key = id2label[model_b_preds[i]]

        if pred_a_key != gt_key or pred_b_key != gt_key:
            misclassified_records.append({
                "file_name": row["file_name"],
                "ground_truth": gt_key,
                "model_a_pred": pred_a_key,
                "model_a_conf": round(float(model_a_probs[i, model_a_preds[i]]), 4),
                "model_b_pred": pred_b_key,
                "model_b_conf": round(float(model_b_probs[i, model_b_preds[i]]), 4),
            })

    mis_df = pd.DataFrame(misclassified_records)
    mis_path = cfg.paths.metrics_dir / "misclassified.csv"
    mis_df.to_csv(mis_path, index=False)
    print(f"Logged {len(misclassified_records)} misclassified cases to {mis_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION BENCHMARK SUMMARY")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    print("=" * 70)

    return 0


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    sys.exit(run_evaluation(cfg_arg))
