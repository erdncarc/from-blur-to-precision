<div align="center">

# ✈️ from Blur to Precision

### Improving Military Aircraft Classification and Detection via Super-Resolution

*low resolution shouldn't mean low confidence*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00a3e0.svg)

</div>

---

## Overview

Aerial surveillance and defense applications need accurate recognition and localization of military aircraft, but low-resolution imagery and arbitrary object orientations degrade model performance. This project investigates SR as a quality-enhancement step that restores fine spatial detail before it reaches the recognition models.

The work is organized as a three-stage pipeline, each stage isolating a different recognition task and comparing a **baseline** (degraded / original-resolution input) against an **SR-enhanced** counterpart under identical training conditions. By holding annotations and hyperparameters fixed and varying only image quality, every comparison attributes the change in accuracy directly to Super-Resolution.

---

## Key Results

Across all three stages, SR produces a clear and reproducible improvement. The numbers below are the test-set figures reported for each model.

### Stage 1 — Classification (MobileNetV2)

| Metric    | Baseline (original crops) | SR-enhanced (EDSR ×4) |
|-----------|:-------------------------:|:---------------------:|
| Accuracy  | 0.400                     | **0.932**             |
| Precision | 0.371                     | **0.933**             |
| Recall    | 0.361                     | **0.931**             |
| F1-score  | 0.351                     | **0.931**             |

### Stage 2 — Horizontal Detection (YOLOv8n)

| Metric          | Baseline (4× downsampled) | SR-enhanced (ESRGAN) |
|-----------------|:-------------------------:|:--------------------:|
| Precision       | 0.62                      | **0.92**             |
| Recall          | 0.61                      | **0.93**             |
| mAP@0.5         | 0.64                      | **0.96**             |
| mAP@0.5:0.95    | 0.39                      | **0.71**             |

### Stage 3 — Oriented Detection (YOLOv8n-OBB)

| Metric              | Baseline (original) | SR-enhanced (ESRGAN) |
|---------------------|:-------------------:|:--------------------:|
| Precision           | 46.22%              | **74.10%**           |
| Recall              | 58.37%              | **72.85%**           |
| Exact-match acc.    | 34.15%              | **56.10%**           |
| Object-count MSE    | 5.90                | **2.70**             |

---

## Pipeline

### Stage 1 — Aircraft Classification

Individual aircraft are isolated from full aerial frames using the horizontal bounding-box annotations, giving an object-level classification task free of background clutter. Crops are padded to a square, resized, and normalized with ImageNet statistics, then split into stratified train / validation / test sets.

Two pipelines are trained for a direct comparison. The baseline feeds original crops resized to 64×64 into a MobileNetV2 (ImageNet-pretrained, frozen backbone, fine-tuned head). The SR pipeline first enhances each crop with the **EDSR ×4** model, then resizes to 224×224 before the same classifier. Both train for 30 epochs with cross-entropy loss and the Adam optimizer, selecting the best checkpoint by validation accuracy.

### Stage 2 — Horizontal Detection

To probe SR on detection, original images are degraded by 4× bicubic downsampling and then restored to their original dimensions with the **ESRGAN (RRDB)** model. Both the degraded and restored image sets keep the original YOLOv8 horizontal annotations, so only image quality differs. Two YOLOv8n detectors are trained for 50 epochs with the AdamW optimizer and cosine learning-rate scheduling, using identical augmentation (horizontal flips, color jitter, mosaic). Evaluation is on a held-out set with original-resolution annotations.

### Stage 3 — Oriented Detection

The final stage tackles rotation-aware recognition. Polygonal OBB annotations are parsed and converted to YOLOv8-OBB format `(cls, x1, y1, …, x4, y4)` in normalized coordinates. Only native 800×800 images are kept for resolution consistency, and a validation split is created by stratified sampling to preserve class balance. A YOLOv8n-OBB detector — initialized from VisDrone-OBB weights — is fine-tuned for 30 epochs (AdamW, cosine LR) on both the original inputs and their ESRGAN-enhanced versions. Beyond detecting presence, the model regresses the vertices of a rotated rectangle, enabling flight-line angle analysis and instance counting.

---

## Dataset

This project uses the **Military Aircraft Recognition** dataset ([Bilel, 2023](https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset)): 3,842 aerial images and over 22,000 annotated instances spanning 20 aircraft classes, with both horizontal and oriented bounding boxes. Most images are 800×800 pixels, which simplifies standardization.

Exploratory analysis showed meaningful class imbalance (classes such as A2, A13, and A16 dominate) and that bounding boxes mostly fall between 80–160 px in width and height — a finding that motivated the 64×64 crop resolution used in classification.

The dataset is **not** included in this repository. Download it from the Kaggle link above and point the notebook to its location.

---

## Repository Structure

```
form-blur-to-precision/
├── src/
│   └── main.ipynb                  # Main code: preprocessing, training, evaluation for all 3 stages
│
└── model/
    ├── RRDB_ESRGAN_x4.pth          # Primary / Super-Resolution model checkpoint
    └── weights/
        ├── classification/
        │   ├── before_supres.pth   # MobileNetV2 trained on original crops
        │   └── after_supres.pth    # MobileNetV2 trained on EDSR-enhanced crops
        │
        ├── horizontal/
        │   ├── before_supres.pth   # YOLOv8n trained on downsampled images
        │   └── after_supres.pth    # YOLOv8n trained on ESRGAN-enhanced images
        │
        └── oriented/
            ├── before_supres.pth   # YOLOv8n-OBB trained on original images
            └── after_supres.pth    # YOLOv8n-OBB trained on ESRGAN-enhanced images
```

Each task folder under `weights/` holds the matched pair used for its baseline-vs-SR comparison: `before_supres.pth` is the model trained without Super-Resolution, and `after_supres.pth` is the SR-enhanced counterpart.

---

## Usage

All experiments live in `src/main.ipynb`. Open it in Colab or Jupyter and run the cells top to bottom. The notebook is organized by stage, so each block can be run independently once the dataset path and weights are configured:

1. Set the dataset path to your local copy of the Military Aircraft Recognition dataset.
2. Run the **Stage 1** cells to reproduce classification (with and without SR).
3. Run **Stage 2** for horizontal detection on downsampled vs SR-enhanced images.
4. Run **Stage 3** for oriented detection on original vs SR-enhanced inputs.

To skip training and run inference directly, load the corresponding checkpoint from `model/weights/<task>/` instead of retraining.

---

## Evaluation Strategy

Each stage is judged with task-appropriate metrics:

- **Classification** — accuracy, precision, recall, F1-score, and confusion matrices.
- **Horizontal detection** — precision, recall, mAP@0.5, mAP@0.5:0.95, and the count of correctly detected instances.
- **Oriented detection** — precision, recall, exact-match accuracy, and the MSE of predicted object counts.

---

## Future Work

Several directions could push the results further. **Content-aware SR** (e.g. GAN-based attention) would concentrate detail restoration on aircraft regions rather than wasting it on background. A **multi-scale late-fusion ensemble** could combine low-res and SR-enhanced predictions to capture complementary signal. **Domain-specific SR**, fine-tuned on aircraft patches rather than general imagery, would better restore features like winglets and tails. Finally, **adversarial or synthetic blur/noise augmentation** during SR training could make the detectors more robust to real-world degradation.

---

## Authors

| Name | Student ID |
|------|-----------|
| Erdinç Arıcı | 2210356035 |
| Gazi Kağan Soysal | 2210356050 |
| Emirhan Utku | 2210765029 |

Each team member was responsible for one stage of the pipeline.
