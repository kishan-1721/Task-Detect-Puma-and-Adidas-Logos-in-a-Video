---

# Puma & Adidas Logo Detection — README

**Project:** Logo Detection in Video (Puma & Adidas)
**Author:** Kishan Patel

---

## 📌 Summary

This repository contains the code, dataset configuration, training notebook, and inference script for detecting **Puma** and **Adidas** logos in video frames using a custom-trained YOLO-based object detection model.
The model was trained on a GPU environment (Google Colab with CUDA). Inference results are generated on a sample video due to submission size limits.

---

## 📑 Table of Contents

* [Project Overview](#project-overview)
* [Dataset Details](#dataset-details)
* [Preprocessing & Augmentation](#preprocessing--augmentation)
* [Train / Val / Test Split](#train--val--test-split)
* [Model Recommendation & Training Setup](#model-recommendation--training-setup)
* [Training Steps (Colab)](#training-steps-colab)
* [Inference (Video) & CSV Output](#inference-video--csv-output)
* [Repository Structure](#repository-structure)
* [Notes & Limitations](#notes--limitations)
* [How to Reproduce (Quick Commands)](#how-to-reproduce-quick-commands)
* [Contact](#contact)

---

## 📝 Project Overview

The goal of this project is to:

* Detect Puma and Adidas logos in video frames
* Annotate each frame with bounding boxes
* Generate a labeled output video
* Export detections into a structured CSV file

This project uses a **custom YOLO model** trained on an augmented dataset of 352 images. Due to file size constraints, inference and outputs were generated on a smaller sample video.

---

## 📂 Dataset Details

* **Base images:** ~130 manually collected + Roboflow augmented
* **Final dataset size:** 352 images
* **Annotation format:** YOLO (normalized x_center, y_center, width, height)
* **Classes:**

  1. puma
  2. adidas

### ✔ Preprocessing applied (Roboflow)

* Auto-orientation (EXIF rotation fix)
* Resize to **512×512 (stretched)**

### ✔ Augmentations applied

* 50% horizontal flip
* 50% vertical flip
* Random 90° rotations (0°, 90°, 180°, 270°)
* Random crop (0–20%)
* Random rotation (−15° to +15°)

---

## 🔧 Preprocessing Notes

* Images resized to 512×512 for consistency
* Augmentation improves robustness to orientation and partial visibility
* Very small logos were minimized due to detection difficulty on limited data

---

## 🔀 Train / Val / Test Split

| Split      | Images |
| ---------- | ------ |
| Train      | 333    |
| Validation | 10     |
| Test       | 9      |

All split folders contain corresponding `images/` and `labels/`.

---

## 🧠 Model Recommendation & Training Setup

* **Model family used:** YOLO (Ultralytics) — *yolo11n*
* **Why YOLO?**

  * Fast
  * Good for small-object detection
  * Easy video inference pipeline
* **Training hardware:** Google Colab GPU (T4)

### Suggested Hyperparameters

| Parameter  | Value              |
| ---------- | ------------------ |
| Epochs     | 300                |
| Image size | 640                |
| Batch size | 16                 |
| Optimizer  | Default (SGD/Adam) |

### Example `data.yaml`

```yaml
train: ../data/images/train
val: ../data/images/valid
test: ../data/images/test

nc: 2
names: ["puma", "adidas"]
```

---

## 🚀 Training Steps (Colab)

1. Enable GPU runtime
2. Install dependencies:

```bash
pip install ultralytics roboflow opencv-python-headless pandas
```

3. Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Train the model:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # pretrained weights

results = model.train(
    data="/content/Brand_Logo_Detection-3/data.yaml",
    epochs=300,
    imgsz=640
)
```

5. Best weights will be saved under:

```
runs/detect/train/weights/best.pt
```

---

## 🎥 Inference (Video) & CSV Output

### Goal

* Process each frame
* Detect Puma/Adidas
* Save annotated video
* Save CSV with:

| frame_no | brand | confidence | x1 | y1 | x2 | y2 |

### Example Script (provided in repo)

```bash
python scripts/detect_video.py \
    --weights runs/detect/train/weights/best.pt \
    --source inputs/sample_video.mp4 \
    --output output/output_labeled_video.mp4 \
    --csv output/detections.csv
```

A sample video and its inference result are included in the `inputs/` and `output/` directories.

---

## 📁 Repository Structure

```
Main-Directory/
├── data/                      # Dataset (train/valid/test)
├── models/                    # Saved model weights (best.pt)
├── runs/                      # YOLO training logs + metrics
├── scripts/
│   ├── Model_Train.ipynb      # Training and evaluation notebook
├── inputs/                    # Sample input videos
├── output/                    # Output videos + CSV results
├── README.md                  # Documentation
├── requirements.txt           # Dependencies
├── data.yaml                  # YOLO dataset config
└── yolo11n.pt                 # Pretrained YOLO weights
```

---

## ⚠️ Notes & Limitations

* 512×512 *stretched* resize may distort logos
* Very small logos remain challenging without high-resolution data
* Only a **sample video** is included due to submission limitations

---

## 💡 How to Reproduce (Quick Commands)

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train model

```bash
python scripts/train.py --data data.yaml --epochs 300 --imgsz 640 --batch 16
```

### 3️⃣ Run inference

```bash
python scripts/detect_video.py \
    --weights models/best.pt \
    --source inputs/sample_video.mp4 \
    --output output/output_labeled_video.mp4 \
    --csv output/detections.csv
```

---

## 📩 Contact

For support, notebook access, or clarifications:

**Kishan Patel**

---
