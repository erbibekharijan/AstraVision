"""
Persistent history and audit log for ASTRA VISION predictions.
Stores prediction records in JSONL format with thumbnail caches.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from defence_recog.inference import PredictionResult


class HistoryManager:
    """
    Manages persistent logging of predictions and thumbnail caching in outputs/history/.
    """

    def __init__(
        self,
        history_file: Union[str, Path] = "outputs/history/history.jsonl",
        thumbs_dir: Union[str, Path] = "outputs/history/thumbnails",
    ) -> None:
        self.history_file = Path(history_file)
        self.thumbs_dir = Path(thumbs_dir)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)

    def record_prediction(
        self,
        result: PredictionResult,
        source_filename: str = "upload.jpg",
    ) -> Dict[str, Any]:
        """
        Save prediction entry and save thumbnail image.
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Compute SHA256 of model's-eye view image bytes
        img_bytes = result.models_eye_view.tobytes()
        sha256_hash = hashlib.sha256(img_bytes).hexdigest()[:16]

        thumb_filename = f"{sha256_hash}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
        thumb_path = self.thumbs_dir / thumb_filename

        try:
            # Save 128x128 thumbnail
            thumb_img = result.models_eye_view.resize((128, 128), Image.Resampling.BILINEAR)
            thumb_img.save(thumb_path, format="JPEG", quality=85)
        except Exception:
            thumb_filename = ""

        entry = {
            "timestamp": timestamp,
            "filename": source_filename,
            "sha256": sha256_hash,
            "model": result.model_name,
            "predicted_class": result.predicted_class_key,
            "predicted_display": result.predicted_display_name,
            "confidence": round(result.confidence, 4),
            "is_uncertain": result.is_uncertain,
            "uncertainty_reasons": result.uncertainty_reasons,
            "top_3": [
                {
                    "class_key": item.class_key,
                    "display_name": item.display_name,
                    "confidence": round(item.confidence, 4),
                }
                for item in result.top_k[:3]
            ],
            "thumbnail_file": thumb_filename,
        }

        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def load_history(self) -> List[Dict[str, Any]]:
        """
        Load all history records, newest first.
        """
        if not self.history_file.exists():
            return []

        records = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        return list(reversed(records))

    def clear_history(self) -> None:
        """
        Purge history log and cached thumbnails.
        """
        if self.history_file.exists():
            self.history_file.unlink()

        if self.thumbs_dir.exists():
            for f in self.thumbs_dir.glob("*.jpg"):
                try:
                    f.unlink()
                except Exception:
                    pass
