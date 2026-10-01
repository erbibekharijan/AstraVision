# Model Comparison on Held-Out Test Set (N=23)

> **Caveat on Sample Size**: Because the test split contains 23 images (~4–5 per class), point estimates are accompanied by bootstrap 95% confidence intervals and 5-fold cross-validation.

| Model                              | Architecture                    | Test Accuracy   | 95% CI Accuracy   | Macro Precision   | Macro Recall   |   Macro F1 |    ECE | CPU Latency (ms)   | Model Size (MB)   |
|:-----------------------------------|:--------------------------------|:----------------|:------------------|:------------------|:---------------|-----------:|-------:|:-------------------|:------------------|
| Model A (ConvNeXt-Tiny Fine-Tuned) | ConvNeXt-Tiny (~28M)            | 87.0%           | [69.6%, 100.0%]   | 90.0%             | 87.0%          |     0.8632 | 0.1295 | 33.13              | 106.16            |
| Model B (CLIP Linear Probe)        | ViT-B/32 + Logistic Reg         | 87.0%           | [69.6%, 100.0%]   | 90.3%             | 87.0%          |     0.8659 | 0.1104 | 0.07               | 0.01              |
| Model B (CLIP Zero-Shot Baseline)  | ViT-B/32 Zero-Shot Text Prompts | 87.0%           | [69.6%, 100.0%]   | 87.7%             | 87.0%          |     0.8652 | 0.0923 | -                  | -                 |

### Per-Class F1 Score Breakdown

| Class | Model A F1 | Model B Probe F1 | Model B Zero-Shot F1 |
|---|---|---|---|
| aircraft | 0.9091 | 0.8333 | 0.8889 |
| drone | 0.8571 | 0.8571 | 0.7500 |
| helicopter | 1.0000 | 1.0000 | 0.8889 |
| military-vehicle | 0.7500 | 0.7500 | 0.9091 |
| naval | 0.8000 | 0.8889 | 0.8889 |

### 5-Fold Cross-Validation Summary (Model B Probe)
- **Mean Accuracy**: 92.7% ± 4.9%
- **Mean Macro-F1**: 0.9251 ± 0.0498
