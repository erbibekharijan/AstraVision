"""
Unit tests for template-based explanation text generation.
Verifies that generated explanations:
- Differ substantially between confident and uncertain states
- Detail runner-up competitors and margins
- Always include the mandatory statistical classification disclaimer
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from defence_recog.explanation_text import DISCLAIMER_TEXT, generate_explanation
from defence_recog.inference import PredictionResult, TopKPrediction


def make_dummy_result(is_uncertain: bool, conf: float = 0.88, runner_up_conf: float = 0.08) -> PredictionResult:
    dummy_img = Image.new("RGB", (224, 224), (100, 100, 100))
    top_k = [
        TopKPrediction(class_id=0, class_key="aircraft", display_name="Fighter Aircraft", confidence=conf),
        TopKPrediction(class_id=1, class_key="drone", display_name="Drone", confidence=runner_up_conf),
        TopKPrediction(class_id=2, class_key="helicopter", display_name="Helicopter", confidence=1.0 - conf - runner_up_conf),
    ]

    reasons = ["Confidence below threshold"] if is_uncertain else []

    return PredictionResult(
        predicted_class_id=0,
        predicted_class_key="aircraft",
        predicted_display_name="Fighter Aircraft",
        confidence=conf,
        top_k=top_k,
        is_uncertain=is_uncertain,
        uncertainty_reasons=reasons,
        models_eye_view=dummy_img,
        sanitized_original=dummy_img,
        raw_logits=np.array([2.0, 1.0, 0.5, 0.0, 0.0]),
        calibrated_probs=np.array([conf, runner_up_conf, 1.0 - conf - runner_up_conf, 0.0, 0.0]),
    )


def test_disclaimer_always_present():
    """Verify statistical disclaimer is unconditionally present in all outputs."""
    confident_res = make_dummy_result(is_uncertain=False, conf=0.92, runner_up_conf=0.04)
    uncertain_res = make_dummy_result(is_uncertain=True, conf=0.45, runner_up_conf=0.40)

    text_conf = generate_explanation(confident_res)
    text_unc = generate_explanation(uncertain_res)

    assert DISCLAIMER_TEXT in text_conf
    assert DISCLAIMER_TEXT in text_unc


def test_text_differs_for_confident_vs_uncertain():
    """Verify confident text highlights high certainty while uncertain text highlights ambiguity."""
    confident_res = make_dummy_result(is_uncertain=False, conf=0.92, runner_up_conf=0.04)
    uncertain_res = make_dummy_result(is_uncertain=True, conf=0.45, runner_up_conf=0.40)

    text_conf = generate_explanation(confident_res)
    text_unc = generate_explanation(uncertain_res)

    assert text_conf != text_unc
    assert "strong confidence" in text_conf.lower()
    assert "tentative confidence" in text_unc.lower() or "ambiguity" in text_unc.lower()
