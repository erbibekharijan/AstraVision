"""
High-level inference engine for ASTRA VISION.
Handles end-to-end preprocessing, model evaluation, temperature calibration,
top-k ranking, and uncertainty quantification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from defence_recog.config import AppConfig, load_config
from defence_recog.errors import InferenceError, ModelNotFoundError
from defence_recog.models import (
    CLIPFeatureExtractor,
    load_clip_probe,
    load_model_a,
)
from defence_recog.preprocess import preprocess_for_model


@dataclass
class TopKPrediction:
    class_id: int
    class_key: str
    display_name: str
    confidence: float  # In range [0.0, 1.0]


@dataclass
class PredictionResult:
    predicted_class_id: int
    predicted_class_key: str
    predicted_display_name: str
    confidence: float  # In range [0.0, 1.0]
    top_k: List[TopKPrediction]
    is_uncertain: bool
    uncertainty_reasons: List[str]
    models_eye_view: Image.Image
    sanitized_original: Image.Image
    raw_logits: np.ndarray
    calibrated_probs: np.ndarray
    temperature: float = 1.0
    model_name: str = "convnext_tiny"


class Predictor:
    """
    Unified predictor supporting Model A (ConvNeXt-Tiny) and Model B (CLIP Linear Probe).
    Applies temperature scaling calibration and uncertainty thresholds.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        model_type: str = "model_a",  # "model_a" or "model_b_probe"
    ) -> None:
        self.config = config or load_config()
        self.model_type = model_type
        self.device = self.config.get_device()
        self.display_names = self.config.get_display_names()
        self.temperature = 1.0

        # Load temperature scaling factor if available
        cal_path = self.config.paths.calibration_file
        if cal_path.exists():
            try:
                with open(cal_path, "r", encoding="utf-8") as f:
                    cal_data = json.load(f)
                    self.temperature = float(cal_data.get("optimal_temperature", 1.0))
            except Exception:
                self.temperature = 1.0

        # Load selected model
        if self.model_type == "model_a":
            self.model, self.id2label, self.label2id = load_model_a(
                model_dir=self.config.paths.model_a_checkpoint,
                device=self.device,
            )
            self.clip_extractor = None
            self.clip_classifier = None
        elif self.model_type == "model_b_probe":
            self.clip_classifier, self.class_names, self.id2label = load_clip_probe(
                self.config.paths.model_b_probe
            )
            self.label2id = {v: k for k, v in self.id2label.items()}
            self.clip_extractor = CLIPFeatureExtractor(
                hf_model_id=self.config.model_b.hf_model_id,
                device=self.device,
            )
            self.model = None
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Expected 'model_a' or 'model_b_probe'.")

    def predict(
        self,
        image_input: Any,
        confidence_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
    ) -> PredictionResult:
        """
        Run inference on provided image (path, bytes, or PIL Image).
        """
        conf_thresh = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.inference.confidence_threshold
        )
        margin_thresh = (
            margin_threshold
            if margin_threshold is not None
            else self.config.inference.margin_threshold
        )
        top_k_count = self.config.inference.top_k

        # 1. Preprocess
        tensor, models_eye_view, sanitized_original = preprocess_for_model(
            image_input=image_input,
            target_size=self.config.image_size,
            max_mb=self.config.upload.max_upload_mb,
            max_pixels=self.config.upload.max_image_pixels,
            min_side_px=self.config.upload.min_side_px,
        )

        # 2. Forward pass
        if self.model_type == "model_a":
            tensor = tensor.to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor)
                logits = outputs.logits  # shape (1, num_classes)
                raw_logits = logits.cpu().numpy()[0]

                # Temperature scaling calibration
                scaled_logits = logits / max(self.temperature, 1e-6)
                probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()[0]
        else:
            # Model B CLIP linear probe
            features = self.clip_extractor.extract_image_embedding(sanitized_original)
            # scikit-learn LogisticRegression predict_proba
            probs = self.clip_classifier.predict_proba(features)[0]
            # Approximate logits via log-odds
            eps = 1e-9
            raw_logits = np.log(np.clip(probs, eps, 1.0 - eps))

        # 3. Top-k ranking
        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top1_prob = float(probs[top1_idx])
        top1_key = self.id2label.get(top1_idx, f"class_{top1_idx}")
        top1_display = self.display_names.get(top1_key, top1_key)

        top2_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = top1_prob - top2_prob

        # 4. Uncertainty evaluation
        uncertainty_reasons = []
        is_uncertain = False

        if top1_prob < conf_thresh:
            is_uncertain = True
            uncertainty_reasons.append(
                f"Confidence ({top1_prob*100:.1f}%) is below the operational certainty threshold ({conf_thresh*100:.1f}%)."
            )

        if margin < margin_thresh:
            is_uncertain = True
            runner_up_key = self.id2label.get(int(sorted_indices[1]), "unknown")
            runner_up_name = self.display_names.get(runner_up_key, runner_up_key)
            uncertainty_reasons.append(
                f"Prediction margin is narrow ({margin*100:.1f}% difference with runner-up '{runner_up_name}')."
            )

        # 5. Build Top-K structures
        top_k_list: List[TopKPrediction] = []
        for rank in range(min(top_k_count, len(sorted_indices))):
            idx = int(sorted_indices[rank])
            key = self.id2label.get(idx, f"class_{idx}")
            disp = self.display_names.get(key, key)
            top_k_list.append(
                TopKPrediction(
                    class_id=idx,
                    class_key=key,
                    display_name=disp,
                    confidence=float(probs[idx]),
                )
            )

        return PredictionResult(
            predicted_class_id=top1_idx,
            predicted_class_key=top1_key,
            predicted_display_name=top1_display,
            confidence=top1_prob,
            top_k=top_k_list,
            is_uncertain=is_uncertain,
            uncertainty_reasons=uncertainty_reasons,
            models_eye_view=models_eye_view,
            sanitized_original=sanitized_original,
            raw_logits=raw_logits,
            calibrated_probs=probs,
            temperature=self.temperature,
            model_name=self.model_type,
        )
