# Thermal Image Anomaly Detection

A YOLOv8x pipeline for detecting normal and anomalous regions in thermal images.

**Computer vision · Object detection · Ultralytics**

Course project for **ITE4052 Computer Vision, Hanyang University ERICA**, Sep–Dec 2025.

[Pipeline implementation](main.py) · [Dependencies](requirements.txt)

## Overview

The project converts COCO annotations into YOLO labels, fine-tunes a detector, and evaluates predictions on a separate test set. The implementation combines data preparation, training, and evaluation in one script.

| Component | Implementation |
| :--- | :--- |
| Detector | YOLOv8x by default |
| Classes | `normal` and `anomaly` |
| Training/validation split | 90/10, stratified by the presence of anomaly annotations |
| Input size | 640 pixels |
| Optimizer | SGD; initial learning rate 0.0008 |
| Augmentation | Small color and geometric changes; mosaic and mixup disabled |
| Test summary | mAP50–95, precision, recall, and a weighted project score |

## Pipeline

```mermaid
flowchart LR
    D["Thermal images + COCO annotations"] --> C["Convert boxes to YOLO labels"]
    C --> S["Training / validation split"]
    S --> T["Fine-tune YOLOv8x"]
    T --> W["Trained checkpoint"]
    W --> E["Separate test images and annotations"]
    E --> R["Metrics CSV"]
```

The augmentation settings are deliberately modest. They are a design choice for this project; the repository does not include an ablation showing that they outperform alternative settings.

## Reported Results

The original README contained both validation tables and separate test summaries. They are separated below to avoid treating them as the same evaluation.

### Validation

The original report describes 300 validation images, with 150 images per class.

| Reported run | Overall mAP50 | Overall mAP50–95 |
| :--- | ---: | ---: |
| Seed 0 | 0.995 | 0.971 |
| Seed 1 | 0.991 | 0.971 |
| Seed 2 | 0.995 | 0.972 |

### Test

| Reported run | mAP50–95 | Precision | Recall | Project score |
| :--- | ---: | ---: | ---: | ---: |
| Seed 0 | 0.7942 | 0.9687 | 0.6809 | 0.8003 |
| Seed 1 | 0.7929 | 0.9616 | 0.6776 | 0.7982 |

The project score is `0.8 × mAP50–95 + 0.1 × precision + 0.1 × recall`; it is a project-specific summary, not a standard detection metric. A complete Seed 2 test summary was not provided in the original README.

These are historical reported values. Original logs, data, and checkpoints are not included, so their provenance cannot be audited here. The current CLI seeds Python, NumPy, and PyTorch, but does not pass `args.seed` to Ultralytics' trainer. The run labels therefore should not be taken as verified independent trainer seeds.

## Data Layout

Prepare the following files beneath the `WORKSPACE` directory configured near the top of [main.py](main.py):

```text
/data/workspace/
├── images/
│   ├── train/
│   └── test/
├── train.json
└── test.json
```

Both JSON files must contain COCO-style `images`, `annotations`, and `categories`. Category order is mapped to numeric class IDs; keep it consistent with the generated class names `normal` (0) and `anomaly` (1).

The source images and annotations are not distributed in this repository. Change `WORKSPACE` if your data is stored elsewhere.

## Getting Started

```bash
git clone https://github.com/yunakinn/thermal-image-anomaly-detection.git
cd thermal-image-anomaly-detection
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Train

```bash
python main.py --mode train --model yolov8x.pt --epochs 30 --seed 0
```

Training converts the annotations and creates a 90/10 split with a fixed split seed of 42. Ultralytics handles the training run and checkpoint directory; use the checkpoint path printed by that run.

### Evaluate

```bash
python main.py --mode test \
  --weights /path/to/training-run/weights/best.pt \
  --seed 0
```

The test command prepares a separate dataset configuration and writes:

```text
WORKSPACE/yolo_test_predictions/test_metrics.csv
```

Each invocation replaces that summary file. Save a copy elsewhere if comparing multiple runs. Training and test preparation also recreate their respective generated YOLO dataset directories.

## Repository Guide

| File | Purpose |
| :--- | :--- |
| [main.py](main.py) | Annotation conversion, splitting, training, and evaluation |
| [requirements.txt](requirements.txt) | Runtime dependencies |
| [LICENSE](LICENSE) | Repository license |

## Limitations

This is a two-class course project on a specific thermal dataset. Performance on other sensors or environments has not been established. The reported validation and test scores describe different splits and should not be compared as interchangeable results.

