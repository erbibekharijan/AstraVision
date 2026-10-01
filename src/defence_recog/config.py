"""
Configuration management for ASTRA VISION.
Implements hierarchical configuration loading with strict precedence:
Environment Variables > .env file > config.yaml defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from dotenv import load_dotenv

from defence_recog.errors import ConfigurationError

# Load environment variables from .env if present
load_dotenv(override=False)


@dataclass
class ClassInfo:
    id: int
    key: str
    display_name: str
    description: str


@dataclass
class ModelAConfig:
    name: str = "convnext_tiny"
    hf_model_id: str = "facebook/convnext-tiny-224"
    num_classes: int = 5
    batch_size: int = 8
    epochs_stage1: int = 8
    lr_stage1: float = 0.001
    epochs_stage2: int = 12
    lr_stage2_backbone: float = 0.0001
    lr_stage2_head: float = 0.001
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    unfreeze_blocks: int = 2
    early_stopping_patience: int = 5


@dataclass
class ModelBConfig:
    name: str = "clip_vit_base"
    hf_model_id: str = "openai/clip-vit-base-patch32"
    prompt_templates: List[str] = field(default_factory=lambda: [
        "a photo of a military {label}",
        "a photo of a defence {label}",
        "a military {label} in operation",
        "a photo of a {label}",
    ])
    class_prompt_mappings: Dict[str, str] = field(default_factory=lambda: {
        "aircraft": "military fighter aircraft",
        "drone": "unmanned military aerial drone or UAV",
        "helicopter": "military combat helicopter",
        "military-vehicle": "military armoured combat vehicle or tank",
        "naval": "military naval warship or vessel",
    })


@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.60
    margin_threshold: float = 0.15
    top_k: int = 3
    device: str = "auto"


@dataclass
class UploadConfig:
    allowed_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
    max_upload_mb: int = 10
    max_image_pixels: int = 50_000_000
    min_side_px: int = 32


@dataclass
class PathsConfig:
    data_dir: Path = Path("data")
    images_dir: Path = Path("data/images")
    labels_csv: Path = Path("data/labels.csv")
    credits_csv: Path = Path("data/credits.csv")
    splits_dir: Path = Path("data/splits")
    train_split: Path = Path("data/splits/train.csv")
    val_split: Path = Path("data/splits/val.csv")
    test_split: Path = Path("data/splits/test.csv")
    model_dir: Path = Path("models")
    model_a_checkpoint: Path = Path("models/model_a")
    model_b_probe: Path = Path("models/clip_logistic_regression.joblib")
    calibration_file: Path = Path("models/calibration.json")
    outputs_dir: Path = Path("outputs")
    metrics_dir: Path = Path("outputs/metrics")
    figures_dir: Path = Path("outputs/figures")
    history_dir: Path = Path("outputs/history")
    history_file: Path = Path("outputs/history/history.jsonl")


@dataclass
class AppConfig:
    project_name: str = "ASTRA VISION"
    version: str = "1.0.0"
    description: str = "AI-Based Defence Equipment Object Recognition System"
    seed: int = 42
    image_size: int = 224
    classes: List[ClassInfo] = field(default_factory=list)
    model_a: ModelAConfig = field(default_factory=ModelAConfig)
    model_b: ModelBConfig = field(default_factory=ModelBConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def get_device(self) -> torch.device:
        dev_str = self.inference.device.lower().strip()
        if dev_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        elif dev_str in ("cuda", "gpu") and torch.cuda.is_available():
            return torch.device("cuda")
        elif dev_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_id2label(self) -> Dict[int, str]:
        return {c.id: c.key for c in self.classes}

    def get_label2id(self) -> Dict[str, int]:
        return {c.key: c.id for c in self.classes}

    def get_display_names(self) -> Dict[str, str]:
        return {c.key: c.display_name for c in self.classes}


def load_config(config_file: Optional[str] = None) -> AppConfig:
    """
    Load configuration from config.yaml, applied with environment overrides.
    """
    if config_file is None:
        # Default locations to search
        candidates = [Path("config.yaml"), Path(__file__).resolve().parent.parent.parent / "config.yaml"]
        for c in candidates:
            if c.exists():
                config_file = str(c)
                break

    raw_yaml: Dict[str, Any] = {}
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                raw_yaml = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to parse config file {config_file}: {e}")

    # Build ClassInfo objects
    classes_list: List[ClassInfo] = []
    classes_data = raw_yaml.get("classes", [])
    if classes_data:
        for item in classes_data:
            classes_list.append(
                ClassInfo(
                    id=item.get("id", 0),
                    key=item.get("key", ""),
                    display_name=item.get("display_name", ""),
                    description=item.get("description", ""),
                )
            )
    else:
        # Fallback default classes
        defaults = [
            (0, "aircraft", "Fighter Aircraft", "Fixed-wing combat aircraft"),
            (1, "drone", "Unmanned Aerial Vehicle (UAV / Drone)", "Autonomous or remotely piloted aircraft"),
            (2, "helicopter", "Military Helicopter", "Rotary-wing combat or utility helicopter"),
            (3, "military-vehicle", "Armoured & Military Vehicle", "Tanks and armoured ground vehicles"),
            (4, "naval", "Naval Vessel / Warship", "Surface and subsurface warships"),
        ]
        for cid, key, name, desc in defaults:
            classes_list.append(ClassInfo(id=cid, key=key, display_name=name, description=desc))

    # Model A config
    raw_a = raw_yaml.get("model_a", {})
    model_a = ModelAConfig(
        name=raw_a.get("name", "convnext_tiny"),
        hf_model_id=raw_a.get("hf_model_id", "facebook/convnext-tiny-224"),
        num_classes=len(classes_list),
        batch_size=raw_a.get("batch_size", 8),
        epochs_stage1=raw_a.get("epochs_stage1", 8),
        lr_stage1=raw_a.get("lr_stage1", 0.001),
        epochs_stage2=raw_a.get("epochs_stage2", 12),
        lr_stage2_backbone=raw_a.get("lr_stage2_backbone", 0.0001),
        lr_stage2_head=raw_a.get("lr_stage2_head", 0.001),
        weight_decay=raw_a.get("weight_decay", 0.01),
        label_smoothing=raw_a.get("label_smoothing", 0.1),
        unfreeze_blocks=raw_a.get("unfreeze_blocks", 2),
        early_stopping_patience=raw_a.get("early_stopping_patience", 5),
    )

    # Model B config
    raw_b = raw_yaml.get("model_b", {})
    model_b = ModelBConfig(
        name=raw_b.get("name", "clip_vit_base"),
        hf_model_id=raw_b.get("hf_model_id", "openai/clip-vit-base-patch32"),
        prompt_templates=raw_b.get("prompt_templates", ModelBConfig().prompt_templates),
        class_prompt_mappings=raw_b.get("class_prompt_mappings", ModelBConfig().class_prompt_mappings),
    )

    # Inference config with ENV override precedence
    raw_inf = raw_yaml.get("inference", {})
    conf_thresh = raw_inf.get("confidence_threshold", 0.60)
    if os.getenv("CONFIDENCE_THRESHOLD"):
        try:
            conf_thresh = float(os.getenv("CONFIDENCE_THRESHOLD", conf_thresh))
        except ValueError:
            pass

    device_pref = os.getenv("DEVICE", raw_inf.get("device", "auto"))

    inference = InferenceConfig(
        confidence_threshold=conf_thresh,
        margin_threshold=raw_inf.get("margin_threshold", 0.15),
        top_k=raw_inf.get("top_k", 3),
        device=device_pref,
    )

    # Upload config with ENV override precedence
    raw_up = raw_yaml.get("upload", {})
    max_mb = raw_up.get("max_upload_mb", 10)
    if os.getenv("MAX_UPLOAD_MB"):
        try:
            max_mb = int(os.getenv("MAX_UPLOAD_MB", max_mb))
        except ValueError:
            pass

    upload = UploadConfig(
        allowed_extensions=raw_up.get("allowed_extensions", [".jpg", ".jpeg", ".png", ".webp", ".bmp"]),
        max_upload_mb=max_mb,
        max_image_pixels=raw_up.get("max_image_pixels", 50_000_000),
        min_side_px=raw_up.get("min_side_px", 32),
    )

    # Paths config with ENV override precedence
    raw_p = raw_yaml.get("paths", {})
    data_dir = Path(os.getenv("DATA_DIR", raw_p.get("data_dir", "data")))
    model_dir = Path(os.getenv("MODEL_DIR", raw_p.get("model_dir", "models")))
    outputs_dir = Path(os.getenv("OUTPUT_DIR", raw_p.get("outputs_dir", "outputs")))

    paths = PathsConfig(
        data_dir=data_dir,
        images_dir=data_dir / "images",
        labels_csv=data_dir / "labels.csv",
        credits_csv=data_dir / "credits.csv",
        splits_dir=data_dir / "splits",
        train_split=data_dir / "splits" / "train.csv",
        val_split=data_dir / "splits" / "val.csv",
        test_split=data_dir / "splits" / "test.csv",
        model_dir=model_dir,
        model_a_checkpoint=model_dir / "model_a",
        model_b_probe=model_dir / "clip_logistic_regression.joblib",
        calibration_file=model_dir / "calibration.json",
        outputs_dir=outputs_dir,
        metrics_dir=outputs_dir / "metrics",
        figures_dir=outputs_dir / "figures",
        history_dir=outputs_dir / "history",
        history_file=outputs_dir / "history" / "history.jsonl",
    )

    # Ensure required runtime folders exist
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.history_dir.mkdir(parents=True, exist_ok=True)

    proj = raw_yaml.get("project", {})

    return AppConfig(
        project_name=proj.get("name", "ASTRA VISION"),
        version=proj.get("version", "1.0.0"),
        description=proj.get("description", "AI-Based Defence Equipment Object Recognition System"),
        seed=raw_yaml.get("seed", 42),
        image_size=raw_yaml.get("image_size", 224),
        classes=classes_list,
        model_a=model_a,
        model_b=model_b,
        inference=inference,
        upload=upload,
        paths=paths,
    )
