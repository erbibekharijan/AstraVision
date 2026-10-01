# Product Requirements Document (PRD) — ASTRA VISION

> **Status:** Candidate Stub  
> **Challenge:** 02 — AI-Based Defence Object Recognition System  
> **Team / Candidate:** ASTRA Recruitment Build Challenge (2026–27)

---

## 1. Overview
This project adheres to the product requirements for **Challenge 02: ASTRA VISION**, an AI-based defence equipment object recognition system designed to classify military platforms from visual imagery with calibrated confidence and transparent explainability.

## 2. Core Functional Requirements
1. **Target Imagery Ingestion**: Accept image inputs via upload or sample selector (JPEG, PNG, WEBP, BMP).
2. **Transfer Learning Recognition**: Classify equipment into 5 defence categories (Fighter Aircraft, UAV / Drone, Military Helicopter, Armoured & Military Vehicle, Naval Vessel).
3. **Calibrated Confidence**: Softmax output probability calibrated via post-hoc temperature scaling.
4. **Uncertainty Flagging**: Active detection and amber warning for out-of-distribution, degraded, or ambiguous inputs.
5. **Analytical Explanation**: Plain-English, deterministic explanation generated from prediction dynamics without external LLM dependencies.
6. **Explainability**: Convolutional activation heatmaps (Grad-CAM) visualizing spatial regions of correlation.

---
*Official club PRD updates will be integrated here upon release.*
