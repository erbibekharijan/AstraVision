"""
Unit tests for inference logic using offline stub models (zero external network dependency).
Verifies:
- Softmax probability distributions sum to ~1.0
- Top-K predictions are strictly descending in confidence
- Low-confidence and narrow-margin uncertainty triggers
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from defence_recog.config import AppConfig, ClassInfo, InferenceConfig, PathsConfig
from defence_recog.inference import PredictionResult, Predictor


class DummyModel(nn.Module):
    """Stub model returning fixed logits without needing pretrained weights."""
    def __init__(self, logits: list[float]):
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        return SimpleNamespace(logits=self.logits)


def make_test_predictor(logits: list[float], conf_thresh: float = 0.60, margin_thresh: float = 0.15) -> Predictor:
    """Creates Predictor instance with injected dummy model."""
    classes = [
        ClassInfo(id=0, key="aircraft", display_name="Fighter Aircraft", description=""),
        ClassInfo(id=1, key="drone", display_name="Drone", description=""),
        ClassInfo(id=2, key="helicopter", display_name="Helicopter", description=""),
        ClassInfo(id=3, key="military-vehicle", display_name="Military Vehicle", description=""),
        ClassInfo(id=4, key="naval", display_name="Naval Vessel", description=""),
    ]

    cfg = AppConfig(
        classes=classes,
        inference=InferenceConfig(
            confidence_threshold=conf_thresh,
            margin_threshold=margin_thresh,
            top_k=3,
            device="cpu",
        ),
        paths=PathsConfig(),
    )

    predictor = object.__new__(Predictor)
    predictor.config = cfg
    predictor.model_type = "model_a"
    predictor.device = torch.device("cpu")
    predictor.display_names = cfg.get_display_names()
    predictor.id2label = cfg.get_id2label()
    predictor.label2id = cfg.get_label2id()
    predictor.temperature = 1.0
    predictor.model = DummyModel(logits)
    return predictor


def test_probabilities_sum_to_one():
    """Verify softmax probability distribution sums to 1.0."""
    predictor = make_test_predictor([5.0, 1.0, 0.5, 0.2, 0.1])
    dummy_img = Image.new("RGB", (100, 100), color=(100, 150, 200))

    result = predictor.predict(dummy_img)

    assert isinstance(result, PredictionResult)
    prob_sum = float(np.sum(result.calibrated_probs))
    assert abs(prob_sum - 1.0) < 1e-5


def test_top_k_ordering():
    """Verify top-k predictions are ordered in strictly non-increasing order of confidence."""
    predictor = make_test_predictor([1.0, 4.0, 2.5, 0.5, 0.1])
    dummy_img = Image.new("RGB", (100, 100), color=(50, 50, 50))

    result = predictor.predict(dummy_img)

    assert len(result.top_k) == 3
    # Top 1 should be class 1 (drone)
    assert result.predicted_class_id == 1
    assert result.top_k[0].class_id == 1
    # Check monotonic descent
    for i in range(len(result.top_k) - 1):
        assert result.top_k[i].confidence >= result.top_k[i + 1].confidence


def test_uncertainty_flag_low_confidence():
    """Verify uncertain flag is set when top-1 probability is below confidence threshold."""
    # Equal logits produce uniform ~0.20 probability per class
    predictor = make_test_predictor([1.0, 1.0, 1.0, 1.0, 1.0], conf_thresh=0.60)
    dummy_img = Image.new("RGB", (100, 100), color=(100, 100, 100))

    result = predictor.predict(dummy_img)

    assert result.is_uncertain is True
    assert any("below the operational certainty threshold" in r for r in result.uncertainty_reasons)


def test_uncertainty_flag_narrow_margin():
    """Verify uncertain flag is set when margin between top-1 and top-2 is below margin threshold."""
    # Logits 5.0 and 4.95 will have close probabilities
    predictor = make_test_predictor([5.0, 4.95, 0.0, 0.0, 0.0], conf_thresh=0.40, margin_thresh=0.15)
    dummy_img = Image.new("RGB", (100, 100), color=(200, 100, 50))

    result = predictor.predict(dummy_img)

    assert result.is_uncertain is True
    assert any("margin is narrow" in r for r in result.uncertainty_reasons)
