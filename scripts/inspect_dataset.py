#!/usr/bin/env python3
"""
Inspect dataset script for ASTRA VISION (Defence Object Recognition System).
Checks schema, missing files, image readability, corrupt files, duplicates,
class counts, imbalance ratio, image formats and dimension statistics.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from PIL import Image


def inspect_dataset(data_dir: str = "data") -> int:
    base_path = Path(data_dir)
    labels_csv = base_path / "labels.csv"
    credits_csv = base_path / "credits.csv"
    images_dir = base_path / "images"

    print("=" * 70)
    print("ASTRA VISION DATASET INSPECTION")
    print(f"Data directory: {base_path.resolve()}")
    print("=" * 70)

    fatal_errors: List[str] = []

    # 1. Check file existence
    if not labels_csv.exists():
        fatal_errors.append(f"labels.csv not found at {labels_csv}")
    if not credits_csv.exists():
        fatal_errors.append(f"credits.csv not found at {credits_csv}")
    if not images_dir.exists():
        fatal_errors.append(f"images directory not found at {images_dir}")

    if fatal_errors:
        for err in fatal_errors:
            print(f"[FATAL] {err}")
        return 1

    # 2. Schema check
    try:
        df_labels = pd.read_csv(labels_csv)
        print(f"\n[1] labels.csv schema: {list(df_labels.columns)} (rows: {len(df_labels)})")
    except Exception as e:
        print(f"[FATAL] Failed to read labels.csv: {e}")
        return 1

    try:
        df_credits = pd.read_csv(credits_csv)
        print(f"[2] credits.csv schema: {list(df_credits.columns)} (rows: {len(df_credits)})")
    except Exception as e:
        print(f"[FATAL] Failed to read credits.csv: {e}")
        return 1

    expected_label_cols = {"file_name", "category"}
    if not expected_label_cols.issubset(set(df_labels.columns)):
        fatal_errors.append(f"labels.csv missing required columns. Expected {expected_label_cols}, got {set(df_labels.columns)}")

    # 3. Duplicate checks in labels.csv
    dupe_filenames = df_labels[df_labels.duplicated(subset=["file_name"], keep=False)]
    if not dupe_filenames.empty:
        fatal_errors.append(f"Found {len(dupe_filenames)} duplicate file_name entries in labels.csv")

    # 4. Class counts & Imbalance ratio
    print("\n[3] Class Distribution:")
    class_counts = df_labels["category"].value_counts()
    for cat, count in class_counts.items():
        print(f"  - {cat:20s}: {count:3d} images ({count / len(df_labels) * 100:.1f}%)")

    max_c = class_counts.max()
    min_c = class_counts.min()
    imbalance_ratio = max_c / min_c if min_c > 0 else float("inf")
    print(f"\n  Total classes: {len(class_counts)}")
    print(f"  Total labelled images: {len(df_labels)}")
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")

    # 5. Check physical files vs CSV
    print("\n[4] Physical Files vs labels.csv Verification:")
    disk_images: Set[str] = set()
    for root, _, files in os.walk(images_dir):
        for f in files:
            full_p = Path(root) / f
            # Normalize to relative path matching labels.csv format (images/category/filename)
            rel_p = str(full_p.relative_to(base_path)).replace("\\", "/")
            disk_images.add(rel_p)

    csv_images = set(df_labels["file_name"].str.replace("\\", "/"))

    missing_on_disk = csv_images - disk_images
    if missing_on_disk:
        fatal_errors.append(f"{len(missing_on_disk)} files listed in labels.csv are missing on disk!")
        for m in sorted(list(missing_on_disk))[:5]:
            print(f"    Missing: {m}")
    else:
        print("  - All files in labels.csv exist on disk: OK")

    untracked_on_disk = disk_images - csv_images
    if untracked_on_disk:
        print(f"  - WARNING: {len(untracked_on_disk)} files found on disk with no row in labels.csv")
        for u in sorted(list(untracked_on_disk))[:5]:
            print(f"    Untracked: {u}")
    else:
        print("  - No untracked files on disk: OK")

    # 6. Image readability, size & format statistics
    print("\n[5] Image Integrity, Formats and Dimensions:")
    corrupt_files: List[Tuple[str, str]] = []
    formats: Counter = Counter()
    dimensions: List[Tuple[int, int]] = []
    aspect_ratios: List[float] = []

    for rel_path in csv_images:
        img_full_path = base_path / rel_path
        try:
            with Image.open(img_full_path) as img:
                img.verify()
            # reopen for attributes since verify closes file descriptor
            with Image.open(img_full_path) as img:
                w, h = img.size
                fmt = img.format or "UNKNOWN"
                mode = img.mode
                formats[fmt] += 1
                dimensions.append((w, h))
                aspect_ratios.append(w / h if h > 0 else 0)
                if min(w, h) < 32:
                    fatal_errors.append(f"Image {rel_path} has degenerate dimension: {w}x{h} (<32px)")
        except Exception as e:
            corrupt_files.append((rel_path, str(e)))

    if corrupt_files:
        fatal_errors.append(f"Found {len(corrupt_files)} corrupt/unreadable images!")
        for cpath, cerr in corrupt_files:
            print(f"    Corrupt: {cpath} -> {cerr}")
    else:
        print("  - All images verified readable without corruption: OK")

    print("\n  Format Distribution:")
    for fmt, count in formats.items():
        print(f"    - {fmt}: {count}")

    if dimensions:
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        min_sides = [min(d[0], d[1]) for d in dimensions]
        print("\n  Dimension Statistics (Width x Height):")
        print(f"    - Min width:  {min(widths)} px, Max width:  {max(widths)} px, Mean width:  {sum(widths)/len(widths):.1f} px")
        print(f"    - Min height: {min(heights)} px, Max height: {max(heights)} px, Mean height: {sum(heights)/len(heights):.1f} px")
        print(f"    - Min side across dataset: {min(min_sides)} px (all >= 300 px target: {min(min_sides) >= 300})")
        print(f"    - Mean aspect ratio (W/H): {sum(aspect_ratios)/len(aspect_ratios):.2f}")

    # 7. Summary & Exit status
    print("\n" + "=" * 70)
    if fatal_errors:
        print(f"INSPECTION FAILED WITH {len(fatal_errors)} FATAL ISSUE(S):")
        for err in fatal_errors:
            print(f"  [X] {err}")
        print("=" * 70)
        return 1
    else:
        print("INSPECTION PASSED: Dataset is clean, complete, and structurally verified.")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    data_dir_arg = sys.argv[1] if len(sys.argv) > 1 else "data"
    sys.exit(inspect_dataset(data_dir_arg))
