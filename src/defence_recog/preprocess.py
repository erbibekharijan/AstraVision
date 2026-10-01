"""
Robust image preprocessing, sanitization and validation pipeline for ASTRA VISION.
Handles EXIF auto-rotation, color mode normalization, dimensional bounds,
decompression bomb safeguards, and generation of the model's-eye-view image.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms.functional as TF

from defence_recog.errors import ImageSizeError, InvalidImageError

# Default ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Maximum pixel threshold against decompression-bomb attacks (50 Megapixels)
MAX_PIXELS_DEFAULT = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS_DEFAULT


def validate_raw_bytes(
    file_bytes: bytes,
    max_mb: int = 10,
    filename: str = "upload",
) -> None:
    """
    Validate raw byte buffer before passing to image parser.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise InvalidImageError(f"Uploaded file '{filename}' is empty (0 bytes).")

    max_bytes = max_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise ImageSizeError(
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_mb} MB."
        )


def load_and_sanitize_image(
    image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
    max_mb: int = 10,
    max_pixels: int = MAX_PIXELS_DEFAULT,
    min_side_px: int = 32,
    allowed_extensions: Optional[list[str]] = None,
) -> Image.Image:
    """
    Safely load, verify, and normalize an image to RGB format.
    Guards against decompression bombs, corrupt payloads, and invalid dimensions.
    """
    Image.MAX_IMAGE_PIXELS = max_pixels

    if isinstance(image_input, Image.Image):
        img = image_input.copy()
    elif isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise InvalidImageError(f"Image file does not exist: {p}")
        if allowed_extensions:
            ext = p.suffix.lower()
            if ext not in allowed_extensions:
                raise InvalidImageError(
                    f"Unsupported file extension '{ext}'. Allowed: {', '.join(allowed_extensions)}"
                )
        with open(p, "rb") as f:
            raw = f.read()
        validate_raw_bytes(raw, max_mb=max_mb, filename=p.name)
        try:
            img = Image.open(io.BytesIO(raw))
        except Image.DecompressionBombError as e:
            raise ImageSizeError(f"Image dimensions exceed safety ceiling of {max_pixels:,} pixels: {e}")
        except Exception as e:
            raise InvalidImageError(f"Could not open image file '{p.name}': {e}")
    elif isinstance(image_input, (bytes, bytearray)):
        raw = bytes(image_input)
        validate_raw_bytes(raw, max_mb=max_mb)
        try:
            img = Image.open(io.BytesIO(raw))
        except Image.DecompressionBombError as e:
            raise ImageSizeError(f"Image dimensions exceed safety ceiling of {max_pixels:,} pixels: {e}")
        except Exception as e:
            raise InvalidImageError(f"Corrupted or unrecognized image byte stream: {e}")
    elif hasattr(image_input, "read"):
        raw = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        fname = getattr(image_input, "name", "stream")
        validate_raw_bytes(raw, max_mb=max_mb, filename=fname)
        try:
            img = Image.open(io.BytesIO(raw))
        except Image.DecompressionBombError as e:
            raise ImageSizeError(f"Image dimensions exceed safety ceiling of {max_pixels:,} pixels: {e}")
        except Exception as e:
            raise InvalidImageError(f"Failed to read image stream '{fname}': {e}")
    else:
        raise InvalidImageError(f"Unsupported input type for image: {type(image_input)}")

    # Verify image integrity
    try:
        # Note: verify() only checks header and blocks. It closes file, so we clone before
        img_buffer = io.BytesIO()
        img.save(img_buffer, format=img.format or "PNG")
        img_buffer.seek(0)
        verify_img = Image.open(img_buffer)
        verify_img.verify()
    except Exception as e:
        raise InvalidImageError(f"Image verification failed (corrupt file): {e}")

    # Handle multi-frame images (e.g. animated GIFs) by selecting the first frame
    if getattr(img, "is_animated", False):
        try:
            img.seek(0)
        except Exception:
            pass

    # Dimensional checks
    w, h = img.size
    total_pixels = w * h
    if total_pixels > max_pixels:
        raise ImageSizeError(
            f"Image dimensions ({w}x{h} = {total_pixels:,} pixels) exceed safety ceiling of {max_pixels:,} pixels."
        )

    if min(w, h) < min_side_px:
        raise ImageSizeError(
            f"Image is too small ({w}x{h} px). Minimum required dimension is {min_side_px} px."
        )

    # Apply EXIF rotation if present
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        # If EXIF metadata is malformed, proceed with raw orientation
        pass

    # Ensure RGB color mode
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Create a clean white background for alpha channels
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            blended = Image.alpha_composite(bg, rgba)
            img = blended.convert("RGB")
        else:
            img = img.convert("RGB")

    return img


def get_models_eye_view(img: Image.Image, target_size: int = 224) -> Image.Image:
    """
    Transform image to exact resolution and crop seen by the neural network:
    Resizes shorter side to target_size * (256/224) then performs a center crop.
    """
    w, h = img.size
    scale = 256.0 / float(target_size)
    target_shorter = int(target_size * scale)

    if w < h:
        new_w = target_shorter
        new_h = int(h * (target_shorter / w))
    else:
        new_h = target_shorter
        new_w = int(w * (target_shorter / h))

    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size

    return resized.crop((left, top, right, bottom))


def preprocess_for_model(
    image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
    target_size: int = 224,
    max_mb: int = 10,
    max_pixels: int = MAX_PIXELS_DEFAULT,
    min_side_px: int = 32,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> Tuple[torch.Tensor, Image.Image, Image.Image]:
    """
    Full preprocessing pipeline.
    Returns:
      - tensor: torch.Tensor of shape (1, 3, target_size, target_size), normalized
      - models_eye_view: PIL Image of shape (target_size, target_size)
      - sanitized_original: PIL Image of sanitized full-resolution RGB image
    """
    sanitized = load_and_sanitize_image(
        image_input=image_input,
        max_mb=max_mb,
        max_pixels=max_pixels,
        min_side_px=min_side_px,
    )

    models_eye = get_models_eye_view(sanitized, target_size=target_size)

    # Convert to Tensor and normalize
    tensor = TF.to_tensor(models_eye)  # (3, H, W) in [0, 1]
    tensor = TF.normalize(tensor, mean=mean, std=std)
    tensor = tensor.unsqueeze(0)  # (1, 3, H, W)

    return tensor, models_eye, sanitized
