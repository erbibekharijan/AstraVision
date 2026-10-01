"""
Model architecture definitions and builder utilities for ASTRA VISION.
Primary Model A: ConvNeXt-Tiny (facebook/convnext-tiny-224) fine-tuned for defence classification.
Comparison Model B: CLIP ViT-B/32 (openai/clip-vit-base-patch32) zero-shot and linear probe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    CLIPModel,
    CLIPProcessor,
    ConvNextForImageClassification,
)

from defence_recog.errors import ModelNotFoundError


# ==============================================================================
# Model A: ConvNeXt-Tiny Transfer Learning
# ==============================================================================

def build_model_a(
    hf_model_id: str = "facebook/convnext-tiny-224",
    num_classes: int = 5,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
) -> ConvNextForImageClassification:
    """
    Instantiate ConvNeXt-Tiny with a freshly initialized 5-class classification head.
    Pre-trained ImageNet-1k weights are loaded into the convolutional backbone.
    """
    if id2label is None:
        id2label = {i: f"class_{i}" for i in range(num_classes)}
    if label2id is None:
        label2id = {v: k for k, v in id2label.items()}

    # Convert keys to int for id2label and string for labels
    id2label_int = {int(k): str(v) for k, v in id2label.items()}
    label2id_str = {str(k): int(v) for k, v in label2id.items()}

    model = AutoModelForImageClassification.from_pretrained(
        hf_model_id,
        num_labels=num_classes,
        id2label=id2label_int,
        label2id=label2id_str,
        ignore_mismatched_sizes=True,
    )
    return model


def configure_model_a_stages(
    model: ConvNextForImageClassification,
    stage: int = 1,
    unfreeze_blocks: int = 2,
) -> None:
    """
    Configure parameter freezing for two-stage transfer learning:
    Stage 1: Freeze entire convolutional backbone, train head (classifier) only.
    Stage 2: Unfreeze last stage / last blocks for discriminative feature fine-tuning.
    """
    if stage == 1:
        # Freeze entire backbone
        for param in model.convnext.parameters():
            param.requires_grad = False
        # Ensure classifier is trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif stage == 2:
        # Keep earlier stages frozen
        for param in model.convnext.embeddings.parameters():
            param.requires_grad = False

        # Freeze earlier stages in encoder
        stages = model.convnext.encoder.stages
        total_stages = len(stages)
        # Stages are typically 4: [0, 1, 2, 3]
        for i in range(max(0, total_stages - 1)):
            for param in stages[i].parameters():
                param.requires_grad = False

        # Unfreeze the last stage
        last_stage = stages[-1]
        for param in last_stage.parameters():
            param.requires_grad = True

        # Classifier remains trainable
        for param in model.classifier.parameters():
            param.requires_grad = True


def save_model_a(
    model: ConvNextForImageClassification,
    output_dir: Union[str, Path],
    id2label: Dict[int, str],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save fine-tuned Model A weights, config, and class mappings to directory.
    """
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(p))

    # Save canonical class mapping file
    class_map_file = p / "class_names.json"
    with open(class_map_file, "w", encoding="utf-8") as f:
        json.dump({
            "id2label": {int(k): v for k, v in id2label.items()},
            "label2id": {v: int(k) for k, v in id2label.items()},
            "metadata": extra_metadata or {},
        }, f, indent=2)


def load_model_a(
    model_dir: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[ConvNextForImageClassification, Dict[int, str], Dict[str, int]]:
    """
    Load fine-tuned Model A from local checkpoint directory.
    """
    p = Path(model_dir)
    if not p.exists() or not (p / "config.json").exists():
        raise ModelNotFoundError(
            f"Model A checkpoint not found at '{p}'. Run 'make train' or 'python scripts/train.py' first."
        )

    model = AutoModelForImageClassification.from_pretrained(str(p))
    if device is not None:
        model = model.to(device)
    model.eval()

    # Load class mapping
    class_map_file = p / "class_names.json"
    if class_map_file.exists():
        with open(class_map_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            id2label = {int(k): v for k, v in data.get("id2label", {}).items()}
            label2id = {str(k): int(v) for k, v in data.get("label2id", {}).items()}
    else:
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        label2id = {str(k): int(v) for k, v in model.config.label2id.items()}

    return model, id2label, label2id


# ==============================================================================
# Model B: CLIP ViT-B/32 Linear Probe & Zero-Shot
# ==============================================================================

class CLIPFeatureExtractor:
    """
    Feature extractor utilizing frozen CLIP ViT-B/32 vision tower.
    Produces L2-normalized image embeddings (512-dim).
    """

    def __init__(
        self,
        hf_model_id: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model_id = hf_model_id
        self.processor = CLIPProcessor.from_pretrained(hf_model_id)
        self.model = CLIPModel.from_pretrained(hf_model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_image_embedding(self, image_input: Any) -> np.ndarray:
        """
        Extract L2-normalized embedding for a single image or batch of images.
        """
        inputs = self.processor(images=image_input, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            image_features = outputs.pooler_output
        elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            image_features = outputs.image_embeds
        elif isinstance(outputs, (tuple, list)):
            image_features = outputs[0]
        else:
            image_features = outputs
        # Normalize embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    @torch.no_grad()
    def compute_zero_shot_logits(
        self,
        image_input: Any,
        prompt_texts: List[str],
    ) -> np.ndarray:
        """
        Compute zero-shot cosine similarities against provided text prompts.
        """
        inputs = self.processor(
            text=prompt_texts,
            images=image_input,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        # image-text similarity logits
        logits_per_image = outputs.logits_per_image  # shape: (batch_size, num_prompts)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs


def save_clip_probe(
    probe_model: Any,
    output_file: Union[str, Path],
    class_names: List[str],
    id2label: Dict[int, str],
) -> None:
    """
    Save trained scikit-learn Logistic Regression linear probe with joblib.
    """
    p = Path(output_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": probe_model,
        "class_names": class_names,
        "id2label": id2label,
        "label2id": {v: k for k, v in id2label.items()},
    }
    joblib.dump(payload, str(p))


def load_clip_probe(
    probe_file: Union[str, Path],
) -> Tuple[Any, List[str], Dict[int, str]]:
    """
    Load trained CLIP linear probe model.
    """
    p = Path(probe_file)
    if not p.exists():
        raise ModelNotFoundError(
            f"CLIP linear probe not found at '{p}'. Run 'make probe' or 'python scripts/train_clip_probe.py' first."
        )
    data = joblib.load(str(p))
    return data["classifier"], data["class_names"], data["id2label"]
