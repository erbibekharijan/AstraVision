#!/usr/bin/env python3
"""
Train Model B: CLIP (openai/clip-vit-base-patch32) Linear Probe.
Extracts L2-normalized 512-dim image embeddings, caches them to .npy,
fits Logistic Regression with hyperparameter tuning over C on validation split,
and evaluates zero-shot classification baseline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from defence_recog.config import AppConfig, load_config
from defence_recog.models import (
    CLIPFeatureExtractor,
    save_clip_probe,
)
from defence_recog.preprocess import load_and_sanitize_image


def extract_embeddings_for_split(
    csv_file: Path,
    extractor: CLIPFeatureExtractor,
    data_dir: Path,
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts L2-normalized CLIP features for a split, saving to cache_path (.npz).
    """
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["labels"], list(data["files"])

    df = pd.read_csv(csv_file)
    features_list = []
    labels_list = []
    files_list = []

    print(f"Extracting CLIP embeddings for {len(df)} images in {csv_file.name}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rel_p = row["file_name"]
        cat = row["category"]
        full_p = data_dir / rel_p

        img = load_and_sanitize_image(full_p)
        emb = extractor.extract_image_embedding(img)  # shape (1, 512)
        features_list.append(emb[0])
        labels_list.append(cat)
        files_list.append(rel_p)

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, labels=labels, files=files_list)
    print(f"Cached {features.shape} embeddings to {cache_path}")
    return features, labels, files_list


def run_clip_probe_training(config_file: str = "config.yaml") -> int:
    cfg = load_config(config_file)
    device = cfg.get_device()

    print("=" * 70)
    print("ASTRA VISION — MODEL B (CLIP ViT-B/32) LINEAR PROBE TRAINING")
    print(f"Device: {device} | Model ID: {cfg.model_b.hf_model_id}")
    print("=" * 70)

    train_csv = cfg.paths.train_split
    val_csv = cfg.paths.val_split
    test_csv = cfg.paths.test_split
    data_dir = cfg.paths.data_dir

    if not train_csv.exists() or not val_csv.exists():
        print("Error: Train/val splits not found. Run 'make splits' first.")
        return 1

    # Instantiate CLIP extractor
    extractor = CLIPFeatureExtractor(hf_model_id=cfg.model_b.hf_model_id, device=device)

    # Embedding cache paths
    cache_dir = cfg.paths.model_dir / "clip_cache"
    train_feat, train_labels, _ = extract_embeddings_for_split(
        train_csv, extractor, data_dir, cache_dir / "train_clip.npz"
    )
    val_feat, val_labels, _ = extract_embeddings_for_split(
        val_csv, extractor, data_dir, cache_dir / "val_clip.npz"
    )
    test_feat, test_labels, _ = extract_embeddings_for_split(
        test_csv, extractor, data_dir, cache_dir / "test_clip.npz"
    )

    class_names = sorted(list(set(train_labels)))
    label2id = {c: i for i, c in enumerate(class_names)}
    id2label = {i: c for c, i in label2id.items()}

    y_train = np.array([label2id[c] for c in train_labels])
    y_val = np.array([label2id[c] for c in val_labels])
    y_test = np.array([label2id[c] for c in test_labels])

    # --------------------------------------------------------------------------
    # Tune Logistic Regression C parameter on Validation Set
    # --------------------------------------------------------------------------
    c_candidates = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    best_c = 1.0
    best_val_f1 = -1.0
    best_clf = None

    print("\nTuning Logistic Regression Regularization C on validation split:")
    for c_val in c_candidates:
        clf = LogisticRegression(
            C=c_val,
            max_iter=1000,
            class_weight="balanced",
            random_state=cfg.seed,
            solver="lbfgs",
        )
        clf.fit(train_feat, y_train)
        val_pred = clf.predict(val_feat)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        print(f"  C={c_val:<6} | Val Accuracy: {val_acc*100:.1f}% | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c_val
            best_clf = clf

    print(f"\nOptimal C selected: {best_c} (Val Macro-F1: {best_val_f1:.4f})")

    # Save trained linear probe
    save_clip_probe(
        probe_model=best_clf,
        output_file=cfg.paths.model_b_probe,
        class_names=class_names,
        id2label=id2label,
    )
    print(f"Saved CLIP linear probe model to {cfg.paths.model_b_probe}")

    # Preliminary test accuracy check
    test_pred = best_clf.predict(test_feat)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")
    print(f"Test split preliminary check -> Accuracy: {test_acc*100:.1f}%, Macro-F1: {test_f1:.4f}")

    return 0


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    sys.exit(run_clip_probe_training(cfg_arg))
