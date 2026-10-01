#!/usr/bin/env python3
"""
Training script for Model A (ConvNeXt-Tiny) using two-stage transfer learning.
Stage 1: Frozen backbone, fine-tune 5-class head.
Stage 2: Discriminative fine-tuning with cosine LR decay and label smoothing.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from defence_recog.config import AppConfig, load_config
from defence_recog.data import create_dataloaders
from defence_recog.metrics import evaluate_predictions
from defence_recog.models import (
    build_model_a,
    configure_model_a_stages,
    save_model_a,
)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, float, float, Dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        logits = outputs.logits
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.vstack(all_probs)

    metrics = evaluate_predictions(y_true, y_pred, y_probs, class_names)
    return avg_loss, metrics["accuracy"], metrics["macro_f1"], metrics


def run_training(config_file: str = "config.yaml") -> int:
    cfg = load_config(config_file)
    seed_everything(cfg.seed)
    device = cfg.get_device()

    print("=" * 70)
    print("ASTRA VISION — MODEL A (ConvNeXt-Tiny) FINE-TUNING")
    print(f"Device: {device} | Pre-trained Backbone: {cfg.model_a.hf_model_id}")
    print("=" * 70)

    # Verify split files
    train_csv = cfg.paths.train_split
    val_csv = cfg.paths.val_split
    if not train_csv.exists() or not val_csv.exists():
        print("Error: Train/val splits not found. Run 'make splits' first.")
        return 1

    class_names = [c.key for c in cfg.classes]
    id2label = cfg.get_id2label()
    label2id = cfg.get_label2id()

    # Dataloaders
    loaders = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        data_root=cfg.paths.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.model_a.batch_size,
        label2id=label2id,
    )

    # Class balance weights
    train_df = pd.read_csv(train_csv)
    counts = train_df["category"].value_counts()
    imbalance_ratio = counts.max() / counts.min()
    if imbalance_ratio > 1.5:
        class_weights = [1.0 / counts[c] for c in class_names]
        weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
        weights_t = weights_t / weights_t.sum() * len(class_names)
    else:
        weights_t = None

    criterion = nn.CrossEntropyLoss(weight=weights_t, label_smoothing=cfg.model_a.label_smoothing)

    # Build model
    print("\nLoading pre-trained ConvNeXt-Tiny weights...")
    model = build_model_a(
        hf_model_id=cfg.model_a.hf_model_id,
        num_classes=len(class_names),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    log_records = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_state_dict = None

    # --------------------------------------------------------------------------
    # STAGE 1: Train classification head only (Frozen backbone)
    # --------------------------------------------------------------------------
    print(f"\n[Stage 1] Training classification head ({cfg.model_a.epochs_stage1} epochs, lr={cfg.model_a.lr_stage1})...")
    configure_model_a_stages(model, stage=1)

    opt_stage1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.model_a.lr_stage1,
        weight_decay=cfg.model_a.weight_decay,
    )

    global_epoch = 0
    for epoch in range(1, cfg.model_a.epochs_stage1 + 1):
        global_epoch += 1
        tr_loss = train_one_epoch(model, loaders["train"], criterion, opt_stage1, device)
        val_loss, val_acc, val_f1, _ = evaluate_epoch(model, loaders["val"], criterion, device, class_names)

        log_records.append({
            "stage": 1,
            "epoch": global_epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })
        print(f"  Epoch {global_epoch:02d} (S1) - Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = global_epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --------------------------------------------------------------------------
    # STAGE 2: Discriminative fine-tuning of final stage
    # --------------------------------------------------------------------------
    print(f"\n[Stage 2] Unfreezing last stage ({cfg.model_a.epochs_stage2} epochs, lr_head={cfg.model_a.lr_stage2_head}, lr_backbone={cfg.model_a.lr_stage2_backbone})...")
    configure_model_a_stages(model, stage=2, unfreeze_blocks=cfg.model_a.unfreeze_blocks)

    # Param groups with differential learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer_stage2 = AdamW([
        {"params": backbone_params, "lr": cfg.model_a.lr_stage2_backbone},
        {"params": head_params, "lr": cfg.model_a.lr_stage2_head},
    ], weight_decay=cfg.model_a.weight_decay)

    scheduler = CosineAnnealingLR(optimizer_stage2, T_max=cfg.model_a.epochs_stage2, eta_min=1e-6)

    patience = cfg.model_a.early_stopping_patience
    no_improve_epochs = 0

    for epoch in range(1, cfg.model_a.epochs_stage2 + 1):
        global_epoch += 1
        tr_loss = train_one_epoch(model, loaders["train"], criterion, optimizer_stage2, device)
        val_loss, val_acc, val_f1, _ = evaluate_epoch(model, loaders["val"], criterion, device, class_names)
        scheduler.step()

        log_records.append({
            "stage": 2,
            "epoch": global_epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })
        print(f"  Epoch {global_epoch:02d} (S2) - Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = global_epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without validation F1 improvement.")
                break

    # Restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\nRestored best checkpoint from epoch {best_epoch} with Val Macro-F1: {best_val_f1:.4f}")

    # Save checkpoint to models/model_a/
    save_model_a(
        model=model,
        output_dir=cfg.paths.model_a_checkpoint,
        id2label=id2label,
        extra_metadata={
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "classes": class_names,
        },
    )
    print(f"Saved fine-tuned model checkpoint to {cfg.paths.model_a_checkpoint}")

    # Save training log CSV
    log_df = pd.DataFrame(log_records)
    log_csv_path = cfg.paths.metrics_dir / "train_log.csv"
    log_df.to_csv(log_csv_path, index=False)
    print(f"Saved training log to {log_csv_path}")

    # Save training curve figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss", color="royalblue", marker="o", markersize=3)
    ax1.plot(log_df["epoch"], log_df["val_loss"], label="Val Loss", color="crimson", marker="s", markersize=3)
    ax1.axvline(x=cfg.model_a.epochs_stage1 + 0.5, color="gray", linestyle="--", label="Stage 2 Unfreeze")
    ax1.set_title("Cross-Entropy Loss Curve", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(log_df["epoch"], log_df["val_accuracy"] * 100, label="Val Accuracy (%)", color="forestgreen", marker="o", markersize=3)
    ax2.plot(log_df["epoch"], log_df["val_macro_f1"] * 100, label="Val Macro-F1 (%)", color="darkorange", marker="^", markersize=3)
    ax2.axvline(x=cfg.model_a.epochs_stage1 + 0.5, color="gray", linestyle="--", label="Stage 2 Unfreeze")
    ax2.set_title("Validation Accuracy & Macro-F1", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Percentage (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_fig_path = cfg.paths.figures_dir / "training_curves.png"
    plt.savefig(curve_fig_path, dpi=200)
    plt.close()
    print(f"Saved training curves to {curve_fig_path}")

    print("\nTraining completed successfully!")
    return 0


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    sys.exit(run_training(cfg_arg))
