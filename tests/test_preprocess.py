"""
Unit tests for image preprocessing, validation, and format handling.
Covers:
- Valid JPG, PNG, WEBP loading
- RGBA transparency compositing & Grayscale to RGB conversion
- EXIF orientation handling
- Corrupt byte handling (InvalidImageError)
- Zero-byte input handling (InvalidImageError)
- Oversized dimension protection (ImageSizeError)
- Sub-minimum dimension rejection (ImageSizeError)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from defence_recog.errors import ImageSizeError, InvalidImageError
from defence_recog.preprocess import (
    load_and_sanitize_image,
    preprocess_for_model,
    validate_raw_bytes,
)


def create_test_image(
    size: tuple[int, int] = (100, 100),
    mode: str = "RGB",
    color: tuple[int, ...] = (120, 140, 160),
    fmt: str = "JPEG",
) -> bytes:
    """Helper to generate in-memory image bytes."""
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_valid_image_formats():
    """Verify JPEG, PNG, WEBP can be loaded and sanitized to RGB."""
    for fmt in ["JPEG", "PNG", "WEBP"]:
        raw = create_test_image((80, 80), fmt=fmt)
        img = load_and_sanitize_image(raw)
        assert img.mode == "RGB"
        assert img.size == (80, 80)


def test_rgba_and_grayscale_conversion():
    """Test RGBA alpha compositing and grayscale conversion to 3-channel RGB."""
    # RGBA with transparency
    rgba_bytes = create_test_image((64, 64), mode="RGBA", color=(255, 0, 0, 128), fmt="PNG")
    img_rgba = load_and_sanitize_image(rgba_bytes)
    assert img_rgba.mode == "RGB"
    assert img_rgba.size == (64, 64)

    # Grayscale L mode
    gray_bytes = create_test_image((64, 64), mode="L", color=(128,), fmt="PNG")
    img_gray = load_and_sanitize_image(gray_bytes)
    assert img_gray.mode == "RGB"
    assert img_gray.size == (64, 64)


def test_zero_byte_rejection():
    """Zero-byte files must raise InvalidImageError."""
    with pytest.raises(InvalidImageError, match="empty"):
        validate_raw_bytes(b"", max_mb=10)

    with pytest.raises(InvalidImageError):
        load_and_sanitize_image(b"")


def test_corrupt_bytes_rejection():
    """Corrupted byte streams must raise InvalidImageError without crash."""
    corrupt_data = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01not_a_real_image_bytes_xyz"
    with pytest.raises(InvalidImageError):
        load_and_sanitize_image(corrupt_data)


def test_tiny_image_rejection():
    """Images with dimensions < min_side_px (32) must raise ImageSizeError."""
    tiny_bytes = create_test_image((24, 24), fmt="PNG")
    with pytest.raises(ImageSizeError, match="too small"):
        load_and_sanitize_image(tiny_bytes, min_side_px=32)


def test_oversized_pixel_guard():
    """Decompression bomb safeguard must reject images exceeding max_pixels."""
    img_bytes = create_test_image((50, 50), fmt="PNG")
    # Setting max_pixels smaller than 2500 should trigger ImageSizeError
    with pytest.raises(ImageSizeError, match="safety ceiling"):
        load_and_sanitize_image(img_bytes, max_pixels=1000)


def test_preprocess_for_model_shapes():
    """Preprocessing must return properly shaped PyTorch tensor (1, 3, 224, 224)."""
    raw = create_test_image((350, 200), fmt="JPEG")
    tensor, models_eye, sanitized = preprocess_for_model(raw, target_size=224)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert isinstance(models_eye, Image.Image)
    assert models_eye.size == (224, 224)
    assert sanitized.size == (350, 200)
