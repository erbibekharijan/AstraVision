#!/usr/bin/env python3
"""
CLI Batch Prediction script for ASTRA VISION.
Processes an entire folder of defence equipment images, applies Model A / B inference,
and outputs a structured CSV with predictions, confidence, runner-up, and uncertainty flag.
Usage:
    python scripts/batch_predict.py --input_dir data/images/aircraft --output outputs/batch_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from defence_recog.config import load_config
from defence_recog.errors import DefenceRecogError
from defence_recog.inference import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASTRA VISION — Batch Image Prediction Tool")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images to process")
    parser.add_argument("--output", type=str, default="outputs/batch_results.csv", help="Output CSV file path")
    parser.add_argument("--model", type=str, default="model_a", choices=["model_a", "model_b_probe"], help="Model architecture")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration YAML path")
    parser.add_argument("--confidence_threshold", type=float, default=None, help="Custom confidence threshold")
    return parser.parse_args()


def run_batch_predict() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return 1

    cfg = load_config(args.config)
    allowed_exts = set(cfg.upload.allowed_extensions)

    # Collect image files
    image_files: List[Path] = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in allowed_exts:
                image_files.append(p)

    if not image_files:
        print(f"No supported images found in '{input_dir}'. Supported extensions: {allowed_exts}")
        return 1

    print(f"Found {len(image_files)} images in '{input_dir}'. Loading {args.model} predictor...")
    try:
        predictor = Predictor(config=cfg, model_type=args.model)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return 1

    results = []
    print("Running batch predictions...")
    for img_path in tqdm(image_files):
        try:
            res = predictor.predict(img_path, confidence_threshold=args.confidence_threshold)
            runner_up_key = res.top_k[1].class_key if len(res.top_k) > 1 else ""
            runner_up_conf = res.top_k[1].confidence if len(res.top_k) > 1 else 0.0

            results.append({
                "file_name": str(img_path.relative_to(input_dir)),
                "full_path": str(img_path.resolve()),
                "predicted_class": res.predicted_class_key,
                "display_name": res.predicted_display_name,
                "confidence_pct": round(res.confidence * 100.0, 2),
                "runner_up_class": runner_up_key,
                "runner_up_confidence_pct": round(runner_up_conf * 100.0, 2),
                "is_uncertain": res.is_uncertain,
                "uncertainty_reasons": "; ".join(res.uncertainty_reasons),
                "status": "SUCCESS",
            })
        except Exception as e:
            results.append({
                "file_name": str(img_path.relative_to(input_dir)),
                "full_path": str(img_path.resolve()),
                "predicted_class": "ERROR",
                "display_name": "ERROR",
                "confidence_pct": 0.0,
                "runner_up_class": "",
                "runner_up_confidence_pct": 0.0,
                "is_uncertain": True,
                "uncertainty_reasons": str(e),
                "status": "FAILED",
            })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False)
    print(f"\nBatch processing complete! Processed {len(results)} images.")
    print(f"Results saved to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(run_batch_predict())
