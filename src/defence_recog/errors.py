"""
Custom exceptions for ASTRA VISION defence object recognition system.
"""


class DefenceRecogError(Exception):
    """Base exception for all domain errors in defence_recog."""
    pass


class InvalidImageError(DefenceRecogError):
    """Raised when an uploaded image is corrupted, zero-byte, unreadable, or invalid format."""
    pass


class ImageSizeError(DefenceRecogError):
    """Raised when image exceeds MAX_UPLOAD_MB, MAX_IMAGE_PIXELS, or is smaller than MIN_SIDE_PX."""
    pass


class ModelNotFoundError(DefenceRecogError):
    """Raised when requested model weights or checkpoints cannot be found on disk."""
    pass


class InferenceError(DefenceRecogError):
    """Raised when a failure occurs during model forward pass or prediction generation."""
    pass


class ConfigurationError(DefenceRecogError):
    """Raised when configuration values or required files are missing or malformed."""
    pass
