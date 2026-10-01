"""
ASTRA VISION — AI-Based Defence Equipment Object Recognition System
Interactive Web Application built with Streamlit.
Tabs: Classify | Batch Processing | Audit History | Model Performance | About & Credits
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# Ensure src is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from defence_recog.config import AppConfig, load_config
from defence_recog.errors import DefenceRecogError, ImageSizeError, InvalidImageError, ModelNotFoundError
from defence_recog.explain import GradCAMExplainer
from defence_recog.explanation_text import generate_explanation
from defence_recog.history import HistoryManager
from defence_recog.inference import PredictionResult, Predictor
from defence_recog.preprocess import load_and_sanitize_image

# Optional YOLOv8 for bonus multi-object detection mode
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ==============================================================================
# PAGE CONFIGURATION & ASTRA BRANDING
# ==============================================================================

st.set_page_config(
    page_title="ASTRA VISION — Defence Equipment Recognition",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for ASTRA Brand Identity (Charcoal #1B1F23, Signal Gold #B8860B)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, .astra-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #1B1F23;
    }

    .main-header {
        background: linear-gradient(135deg, #1B1F23 0%, #2D3748 100%);
        padding: 1.6rem 2.0rem;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.12);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .main-header h1 {
        color: #FFFFFF;
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #E2E8F0;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    .gold-badge {
        background-color: #B8860B;
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
    }

    .prediction-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }

    .class-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B1F23;
        margin: 0 0 0.5rem 0;
    }

    .uncertain-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        color: #92400E;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    .certain-banner {
        background-color: #ECFDF5;
        border-left: 5px solid #059669;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        color: #065F46;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    .disclaimer-footer {
        text-align: center;
        padding: 1.8rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid #E2E8F0;
        color: #64748B;
        font-size: 0.85rem;
    }

    .stat-metric {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        text-align: center;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.0rem;
        padding: 10px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# RESOURCE CACHING & INITIALIZATION
# ==============================================================================

@st.cache_resource(show_spinner="Initializing ASTRA VISION Engine...")
def get_app_config() -> AppConfig:
    return load_config(str(BASE_DIR / "config.yaml"))


@st.cache_resource(show_spinner="Loading Deep Learning Vision Model...")
def get_predictor() -> Optional[Predictor]:
    cfg = get_app_config()
    try:
        return Predictor(config=cfg)
    except ModelNotFoundError:
        return None
    except Exception as e:
        st.error(f"Failed to load predictor for '{model_type}': {e}")
        return None


@st.cache_resource(show_spinner="Loading YOLOv8 Detector...")
def get_yolo_detector():
    if not YOLO_AVAILABLE:
        return None
    try:
        return YOLO("yolov8n.pt")
    except Exception:
        return None


# ==============================================================================
# SIDEBAR
# ==============================================================================

def render_sidebar(cfg: AppConfig) -> Tuple[float, float, bool, bool, bool]:
    # Display ASTRA Logo
    logo_path = BASE_DIR / "assets" / "branding" / "astra-logo.png"
    if not logo_path.exists():
        logo_path = BASE_DIR / "branding" / "astra-logo.png"

    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)
    else:
        st.sidebar.markdown("## 🛡️ **ASTRA VISION**")

    st.sidebar.markdown(
        """
        <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 1.2rem;">
        BMS Institute of Technology & Management<br>
        <strong>Defence Object Recognition System</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ **Inference Settings**")

    st.sidebar.markdown("**Active Classifier**  \nModel A: ConvNeXt-Tiny (Fine-Tuned)")

    # Threshold sliders
    conf_default = cfg.inference.confidence_threshold
    cal_file = cfg.paths.calibration_file
    if cal_file.exists():
        try:
            with open(cal_file, "r") as f:
                cdata = json.load(f)
                conf_default = float(cdata.get("suggested_confidence_threshold", conf_default))
        except Exception:
            pass

    confidence_thresh = st.sidebar.slider(
        "Certainty Threshold (%)",
        min_value=30,
        max_value=95,
        value=int(conf_default * 100),
        step=5,
        help="Predictions below this confidence level trigger an uncertain warning banner.",
    ) / 100.0

    margin_thresh = st.sidebar.slider(
        "Min Margin Threshold (%)",
        min_value=5,
        max_value=30,
        value=int(cfg.inference.margin_threshold * 100),
        step=1,
        help="If the difference between Top-1 and Top-2 is smaller than this margin, the prediction is flagged uncertain.",
    ) / 100.0

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 **Explainability & Visuals**")

    show_gradcam = st.sidebar.checkbox(
        "Grad-CAM Attention Heatmap",
        value=True,
        help="Visualizes convolutional activation map highlighting features that drove prediction.",
    )

    show_models_eye = st.sidebar.checkbox(
        "Model's-Eye View (224×224)",
        value=False,
        help="Examine the exact cropped and normalized tensor view processed by the neural network.",
    )

    enable_yolo = st.sidebar.checkbox(
        "Detect Multi-Object (Experimental)",
        value=False,
        help="Uses YOLOv8n to draw bounding boxes, then crops each target for Model A classification.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        **System Status**:
        - PyTorch: `{torch.__version__}`
        - Execution Device: `{cfg.get_device()}`
        - Classes: `5 Defence Targets`
        """
    )

    return confidence_thresh, margin_thresh, show_gradcam, show_models_eye, enable_yolo


# ==============================================================================
# TAB 1: CLASSIFY (SINGLE IMAGE)
# ==============================================================================

def render_classify_tab(
    cfg: AppConfig,
    predictor: Optional[Predictor],
    conf_thresh: float,
    margin_thresh: float,
    show_gradcam: bool,
    show_models_eye: bool,
    enable_yolo: bool,
    history_mgr: HistoryManager,
) -> None:
    st.markdown("### 🎯 Single Target Recognition")

    if predictor is None:
        st.warning(
            "⚠️ **Model A weights were not found.**\n\n"
            f"Please run the training pipeline to generate the weights:\n"
            f"Run `make train` or `python scripts/train.py`.\n"
        )
        return

    # Sample images selector for instant testing without file upload
    test_csv = cfg.paths.test_split
    sample_options = ["None (Upload my own image)"]
    sample_files: Dict[str, Path] = {}

    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        for _, row in df_test.head(10).iterrows():
            rel_p = row["file_name"]
            cat = row["category"]
            fname = Path(rel_p).name
            label = f"Sample: {fname} [{cat}]"
            sample_options.append(label)
            sample_files[label] = cfg.paths.data_dir / rel_p

    col_select, col_upload = st.columns([1, 2])
    with col_select:
        selected_sample = st.selectbox(
            "🧪 Try a Sample Image (Test Split):",
            options=sample_options,
            index=0,
            help="Select one of the held-out test images to evaluate immediately.",
        )

    with col_upload:
        uploaded_file = st.file_uploader(
            "Or upload an image file (JPG, PNG, WEBP, BMP):",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help=f"Maximum allowed file size: {cfg.upload.max_upload_mb} MB.",
        )

    image_source = None
    source_filename = "upload.jpg"

    if selected_sample != "None (Upload my own image)":
        image_source = sample_files[selected_sample]
        source_filename = image_source.name
    elif uploaded_file is not None:
        image_source = uploaded_file
        source_filename = uploaded_file.name

    if image_source is None:
        st.info("👆 Select a sample image above or upload an image to begin classification.")
        return

    # Process and classify
    try:
        with st.spinner("Analyzing target imagery..."):
            result = predictor.predict(
                image_input=image_source,
                confidence_threshold=conf_thresh,
                margin_threshold=margin_thresh,
            )
            # Log to persistent history
            history_mgr.record_prediction(result, source_filename=source_filename)

    except (InvalidImageError, ImageSizeError) as e:
        st.error(f"❌ **Input Validation Error:** {e}")
        return
    except Exception as e:
        st.error(f"❌ **Inference Error:** An unexpected error occurred: {e}")
        return

    # Visual Layout: Left column = Images, Right column = Results
    col_left, col_right = st.columns([1.1, 1.3], gap="large")

    with col_left:
        # Check experimental multi-object detection mode
        yolo_detector = get_yolo_detector() if enable_yolo else None
        if enable_yolo and yolo_detector is not None:
            st.markdown("#### 🔭 Multi-Object Detection (Experimental)")
            st.caption("COCO-pretrained YOLOv8 draws bounding boxes; Model A classifies each cropped object.")
            # Run YOLO
            img_cv = np.array(result.sanitized_original)
            yolo_res = yolo_detector(img_cv, verbose=False)[0]
            boxes = yolo_res.boxes

            if len(boxes) > 0:
                drawn_img = result.sanitized_original.copy()
                from PIL import ImageDraw
                draw = ImageDraw.Draw(drawn_img)
                detected_crops_info = []

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    # Crop object
                    crop = result.sanitized_original.crop((x1, y1, x2, y2))
                    if crop.width >= 20 and crop.height >= 20:
                        crop_res = predictor.predict(crop)
                        draw.rectangle([x1, y1, x2, y2], outline="#B8860B", width=3)
                        label_txt = f"{crop_res.predicted_display_name} ({crop_res.confidence*100:.0f}%)"
                        draw.text((x1 + 4, max(0, y1 - 15)), label_txt, fill="#B8860B")
                        detected_crops_info.append(label_txt)

                st.image(drawn_img, caption="Multi-Object Detections with Model A Classification", use_container_width=True)
                st.caption(f"Detected {len(boxes)} object(s): " + ", ".join(detected_crops_info))
            else:
                st.info("No generic objects detected by YOLOv8. Displaying standard image view.")
                st.image(result.sanitized_original, caption=f"Target: {source_filename}", use_container_width=True)
        else:
            st.markdown("#### 📷 Target Imagery")
            st.image(result.sanitized_original, caption=f"File: {source_filename}", use_container_width=True)

        # Grad-CAM overlay if requested and using Model A
        if show_gradcam and predictor.model is not None:
            st.markdown("#### 🔥 Grad-CAM Saliency Attention")
            explainer = GradCAMExplainer(predictor.model, device=predictor.device)
            # Preprocess tensor for CAM
            cam_tensor = result.models_eye_view
            cam_t = torch.tensor(np.array(cam_tensor).transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0)
            # Normalize with ImageNet
            for c_i, (m, s) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
                cam_t[:, c_i] = (cam_t[:, c_i] - m) / s

            cam_map = explainer.generate_cam(cam_t, target_class_id=result.predicted_class_id)
            overlay = explainer.overlay_heatmap(result.sanitized_original, cam_map, alpha=0.55)
            st.image(overlay, caption="Grad-CAM Feature Activation Map", use_container_width=True)
            st.caption("⚠️ *Heatmaps represent statistical feature correlation in the final stage, not human reasoning.*")

    with col_right:
        st.markdown("#### 📊 Classification Intelligence")

        # Uncertainty Banner
        if result.is_uncertain:
            reasons_html = "<br>• ".join([""] + result.uncertainty_reasons)
            st.markdown(
                f"""
                <div class="uncertain-banner">
                <strong>⚠️ UNCERTAIN PREDICTION</strong>{reasons_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="certain-banner">
                <strong>✅ HIGH-CONFIDENCE CLASSIFICATION</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Prominent Result Card
        conf_pct = result.confidence * 100.0
        st.markdown(
            f"""
            <div class="prediction-card">
                <span class="gold-badge">{result.model_name.upper()}</span>
                <div class="class-header" style="margin-top: 0.5rem;">{result.predicted_display_name}</div>
                <div style="font-size: 1.15rem; color: #475569; margin-bottom: 0.8rem;">
                    Confidence Score: <strong>{conf_pct:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(result.confidence)

        # Plain-English Explanation
        st.markdown("##### 📝 Analytical Assessment")
        explanation = generate_explanation(result)
        st.info(explanation)

        # Top-3 Predictions Chart
        st.markdown("##### 🏆 Top Candidate Rankings")
        chart_data = pd.DataFrame([
            {
                "Class": p.display_name,
                "Confidence (%)": round(p.confidence * 100.0, 1),
            }
            for p in result.top_k
        ])

        chart = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadius=4, color="#B8860B")
            .encode(
                x=alt.X("Confidence (%):Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Class:N", sort="-x", title=""),
                tooltip=["Class", "Confidence (%)"],
            )
            .properties(height=140)
        )
        st.altair_chart(chart, use_container_width=True)

        # Model's-eye view expander
        if show_models_eye:
            with st.expander("👁️ Model's-Eye View (224×224 RGB)", expanded=True):
                st.image(
                    result.models_eye_view,
                    caption="Preprocessed resolution fed into neural network",
                    width=224,
                )

        with st.expander("🛠️ Raw Logits & Calibration Telemetry"):
            st.json({
                "model_name": result.model_name,
                "temperature": result.temperature,
                "predicted_index": result.predicted_class_id,
                "calibrated_probabilities": {
                    predictor.id2label.get(i, f"c_{i}"): round(float(result.calibrated_probs[i]), 4)
                    for i in range(len(result.calibrated_probs))
                },
            })


# ==============================================================================
# TAB 2: BATCH PROCESSING
# ==============================================================================

def render_batch_tab(cfg: AppConfig, predictor: Optional[Predictor]) -> None:
    st.markdown("### 📦 Batch Image Processing")
    st.caption("Upload multiple equipment images simultaneously to generate structured tabular intelligence.")

    if predictor is None:
        st.warning("Please train Model A before running batch processing.")
        return

    uploaded_files = st.file_uploader(
        "Upload a batch of images:",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload 2 or more defence equipment images to run automated batch classification.")
        return

    st.write(f"Loaded **{len(uploaded_files)}** images for processing.")

    if st.button("🚀 Process Batch", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        t0 = time.time()
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Processing ({idx+1}/{len(uploaded_files)}): {file.name}")
            try:
                res = predictor.predict(file)
                runner_up = res.top_k[1].display_name if len(res.top_k) > 1 else ""
                runner_up_conf = res.top_k[1].confidence if len(res.top_k) > 1 else 0.0

                results.append({
                    "Filename": file.name,
                    "Predicted Class": res.predicted_display_name,
                    "Confidence (%)": round(res.confidence * 100.0, 1),
                    "Runner-Up": runner_up,
                    "Runner-Up Conf (%)": round(runner_up_conf * 100.0, 1),
                    "Uncertain?": "⚠️ Yes" if res.is_uncertain else "✅ No",
                    "Uncertainty Detail": "; ".join(res.uncertainty_reasons),
                })
            except Exception as e:
                results.append({
                    "Filename": file.name,
                    "Predicted Class": "ERROR",
                    "Confidence (%)": 0.0,
                    "Runner-Up": "",
                    "Runner-Up Conf (%)": 0.0,
                    "Uncertain?": "❌ Failed",
                    "Uncertainty Detail": str(e),
                })
            progress_bar.progress((idx + 1) / len(uploaded_files))

        elapsed = time.time() - t0
        status_text.success(f"Processed {len(uploaded_files)} images in {elapsed:.2f} seconds ({elapsed/len(uploaded_files)*1000:.1f} ms/image).")

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        # CSV Download Button
        csv_data = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Batch Results (CSV)",
            data=csv_data,
            file_name=f"batch_results_{int(time.time())}.csv",
            mime="text/csv",
        )


# ==============================================================================
# TAB 3: AUDIT HISTORY
# ==============================================================================

def render_history_tab(cfg: AppConfig, history_mgr: HistoryManager) -> None:
    st.markdown("### 🗃️ Persistent Prediction History & Gallery")
    st.caption("Audit log of predictions persisted across application restarts with thumbnail indexing.")

    history = history_mgr.load_history()

    if not history:
        st.info("No prediction history recorded yet. Make a classification in the 'Classify' tab to populate this gallery.")
        return

    # Filter controls
    col_f1, col_f2, col_clear = st.columns([1.5, 1.5, 1.2])
    with col_f1:
        classes_present = ["All"] + sorted(list({h.get("predicted_display", "") for h in history}))
        filter_class = st.selectbox("Filter by Class:", options=classes_present)
    with col_f2:
        filter_uncertain = st.selectbox("Filter by Certainty:", options=["All", "Certain Only", "Uncertain Only"])
    with col_clear:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear History", help="Delete all audit records and thumbnails"):
            history_mgr.clear_history()
            st.rerun()

    # Apply filters
    filtered = history
    if filter_class != "All":
        filtered = [h for h in filtered if h.get("predicted_display") == filter_class]
    if filter_uncertain == "Certain Only":
        filtered = [h for h in filtered if not h.get("is_uncertain", False)]
    elif filter_uncertain == "Uncertain Only":
        filtered = [h for h in filtered if h.get("is_uncertain", False)]

    st.write(f"Showing **{len(filtered)}** of **{len(history)}** logged events.")

    # Gallery display in 3-column grid
    cols_per_row = 3
    for i in range(0, len(filtered), cols_per_row):
        row_items = filtered[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for c, item in zip(cols, row_items):
            with c:
                st.markdown(
                    f"""
                    <div class="stat-metric">
                        <strong>{item.get('predicted_display')}</strong><br>
                        Confidence: <strong>{item.get('confidence', 0)*100:.1f}%</strong><br>
                        <small>{'⚠️ Uncertain' if item.get('is_uncertain') else '✅ Confident'} · {item.get('model')}</small><br>
                        <small style="color: #94A3B8;">{item.get('timestamp')[:19].replace('T', ' ')}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                thumb_file = item.get("thumbnail_file")
                if thumb_file:
                    thumb_path = history_mgr.thumbs_dir / thumb_file
                    if thumb_path.exists():
                        st.image(str(thumb_path), use_container_width=True)


# ==============================================================================
# TAB 4: MODEL PERFORMANCE & METRICS
# ==============================================================================

def render_performance_tab(cfg: AppConfig) -> None:
    st.markdown("### 📈 Model A Performance")
    st.caption("Model A evaluation on the held-out test split (N=23).")

    comp_csv = cfg.paths.metrics_dir / "comparison.csv"
    cm_a = cfg.paths.figures_dir / "confusion_matrix_model_a.png"
    curves = cfg.paths.figures_dir / "training_curves.png"
    misclassified_csv = cfg.paths.metrics_dir / "misclassified.csv"

    if not comp_csv.exists():
        st.warning(
            "⚠️ **Benchmark metrics not yet computed.**\n\n"
            "Run the evaluation pipeline: `make evaluate` or `python scripts/evaluate.py` to generate the comparison suite."
        )
        return

    # Comparison Table
    df_comp = pd.read_csv(comp_csv)
    df_comp = df_comp[df_comp["Model"].str.startswith("Model A")]
    st.markdown("#### 🏆 Model A Test Results")
    st.dataframe(df_comp, use_container_width=True)

    st.markdown(
        "> **Note on Sample Size**: Held-out test set contains 23 images (~4–5 per class). "
        "Confidence intervals are generated via 1,000 bootstrap iterations to transparently capture sampling variance."
    )

    # Confusion Matrices
    st.markdown("---")
    st.markdown("#### 🔲 Model A Test Confusion Matrix")
    if cm_a.exists():
        st.image(str(cm_a), caption="Model A (ConvNeXt-Tiny) Confusion Matrix", use_container_width=True)
    else:
        st.info("Model A confusion matrix not available.")

    # Training Curves
    if curves.exists():
        st.markdown("---")
        st.markdown("#### 📉 Model A Two-Stage Training Curves")
        st.image(str(curves), caption="Cross-Entropy Loss and Validation Accuracy / Macro-F1 across Epochs", use_container_width=True)

    # Misclassified test cases
    if misclassified_csv.exists():
        df_mis = pd.read_csv(misclassified_csv)
        st.markdown("---")
        st.markdown(f"#### 🔍 Error Analysis: Misclassified Test Images ({len(df_mis)} cases)")
        if len(df_mis) > 0:
            st.dataframe(df_mis, use_container_width=True)
        else:
            st.success("Zero test set misclassifications! All test samples correctly identified.")


# ==============================================================================
# TAB 5: ABOUT & CREDITS
# ==============================================================================

def render_about_tab(cfg: AppConfig) -> None:
    st.markdown("### ℹ️ About ASTRA VISION & Dataset Attribution")

    col_desc, col_tech = st.columns([1.2, 1.0], gap="large")

    with col_desc:
        st.markdown(
            """
            #### 🛡️ Project Overview
            **ASTRA VISION** is an AI-based defence object recognition prototype developed for 
            **Challenge 02** of the **ASTRA Software Recruitment Challenge** at **BMSIT**.

            The system accepts imagery of military equipment, classifies the platform into one of five defence categories, 
            quantifies prediction certainty, and generates plain-English explanations with Grad-CAM feature attention maps.

            #### 🎯 The 5 Target Categories
            1. **Fighter Aircraft**: Fixed-wing combat, patrol, and multi-role airframes.
            2. **UAV / Drone**: Remotely piloted and autonomous aerial vehicles.
            3. **Military Helicopter**: Rotary-wing combat, attack, and utility helicopters.
            4. **Armoured & Military Vehicle**: Main battle tanks, infantry vehicles, and tactical ground platforms.
            5. **Naval Vessel**: Surface and subsurface warships, carriers, destroyers, and patrol boats.

            #### ⚠️ Limitations & Responsible AI Notice
            - **Closed-Set Taxonomy**: The system only operates across 5 defence classes. Unrelated imagery (e.g. civilian cars, animals) will be forced into one of the 5 classes. The certainty threshold mitigates this by flagging low-probability classifications.
            - **Dataset Scale**: Trained on ~150 curated open-source Wikimedia Commons images (~30 images/class).
            - **Educational Use**: Intended solely for academic, engineering evaluation and recruitment demonstration. Not for operational or safety-critical deployment.
            """
        )

    with col_tech:
        st.markdown(
            """
            #### 🛠️ Technology Stack
            | Component | Technology | Rationale |
            |---|---|---|
            | **Model A** | `facebook/convnext-tiny-224` | Modern CNN backbone with strong inductive bias on small datasets |
            | **Explainability** | Grad-CAM (`grad-cam`) | Saliency attention maps showing feature correlation |
            | **Framework** | PyTorch & Hugging Face | Reliable CPU/GPU inference with uniform API |
            | **Application** | Streamlit | Rapid interactive web dashboard |
            | **Preprocessing** | Pillow & Torchvision | EXIF normalization, RGB sanitization |

            #### 📐 Architecture Pipeline
            `Input Image` ➔ `Sanitize & EXIF Fix` ➔ `Model's-Eye View (224×224)` ➔ `ConvNeXt-Tiny` ➔ `Temperature Calibration` ➔ `Top-K Ranking` ➔ `Grad-CAM Overlay` ➔ `User Interface`
            """
        )

    st.markdown("---")
    st.markdown("#### 📜 Dataset Credits & Attribution (Mandatory Third-Party Licences)")
    st.markdown(
        "All starter images were sourced from **Wikimedia Commons** under open public licenses "
        "(Creative Commons CC BY-SA 4.0, CC BY 4.0, CC0 1.0, and Public Domain). The original dataset "
        "catalogue (`credits.csv`) is preserved unmodified below:"
    )

    credits_path = cfg.paths.credits_csv
    if credits_path.exists():
        df_credits = pd.read_csv(credits_path)
        st.dataframe(df_credits, use_container_width=True, height=350)
    else:
        st.info("Credits file not found at data/credits.csv.")


# ==============================================================================
# MAIN APPLICATION ROUTING
# ==============================================================================

def main() -> None:
    cfg = get_app_config()
    history_mgr = HistoryManager(
        history_file=cfg.paths.history_file,
        thumbs_dir=cfg.paths.history_dir / "thumbnails",
    )

    # Header
    st.markdown(
        f"""
        <div class="main-header">
            <div>
                <h1>🛡️ {cfg.project_name}</h1>
                <p>AI-Based Defence Equipment Object Recognition & Saliency Intelligence</p>
            </div>
            <div>
                <span class="gold-badge">ASTRA · BMSIT</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar settings
    conf_thresh, margin_thresh, show_gradcam, show_models_eye, enable_yolo = render_sidebar(cfg)

    # Initialize active predictor
    predictor = get_predictor()

    # Top Tabs
    tab_classify, tab_batch, tab_history, tab_perf, tab_about = st.tabs([
        "🎯 Classify",
        "📦 Batch",
        "🗃️ History",
        "📈 Model Performance",
        "ℹ️ About & Credits",
    ])

    with tab_classify:
        render_classify_tab(
            cfg=cfg,
            predictor=predictor,
            conf_thresh=conf_thresh,
            margin_thresh=margin_thresh,
            show_gradcam=show_gradcam,
            show_models_eye=show_models_eye,
            enable_yolo=enable_yolo,
            history_mgr=history_mgr,
        )

    with tab_batch:
        render_batch_tab(cfg, predictor)

    with tab_history:
        render_history_tab(cfg, history_mgr)

    with tab_perf:
        render_performance_tab(cfg)

    with tab_about:
        render_about_tab(cfg)

    # Persistent Footer
    st.markdown(
        """
        <div class="disclaimer-footer">
        <strong>ASTRA VISION</strong> · Educational Prototype · Recruitment Build Challenge 2026–27<br>
        <em>Predictions are statistical machine learning outputs and can be incorrect. Not for operational, targeting, or safety-critical use.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
