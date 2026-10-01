"""
ASTRA VISION — Defence Equipment Object Recognition Package
"""

__version__ = "1.0.0"
__author__ = "ASTRA BMSIT"

from defence_recog.errors import (
    DefenceRecogError,
    InvalidImageError,
    ImageSizeError,
    ModelNotFoundError,
    InferenceError,
    ConfigurationError,
)

__all__ = [
    "DefenceRecogError",
    "InvalidImageError",
    "ImageSizeError",
    "ModelNotFoundError",
    "InferenceError",
    "ConfigurationError",
]
