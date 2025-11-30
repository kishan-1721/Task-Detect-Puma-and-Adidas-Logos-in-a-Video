# Puma & Adidas Logo Detection — README

*Project:* Logo detection in video (Puma & Adidas)

*Author:* Kishan Patel

*Summary*
This repository contains the code, dataset details, training configuration, and inference scripts used to train a custom object-detection model for detecting Puma and Adidas logos in video frames. The model was trained using a GPU (Google Colab with CUDA) and inference was tested on a sample video (full-size video had size issues; submitted results are for the sample video).

---

## Table of Contents

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

## Project Overview

Detect Puma and Adidas logos in each frame of a video, annotate detections with bounding boxes and class labels, save a labeled video, and export a CSV listing every detection (frame, class, confidence, bounding box coordinates).

This work was completed with the following constraints and choices: small custom dataset, augmentation applied to increase image count, and training performed on Google Colab GPU (CUDA). Due to video size constraints during submission, a sample video was used for demonstrating the inference pipeline and outputs.

---

## Dataset Details

* *Original source images (manually collected + Roboflow exports):* 130 images
* *Final dataset size after augmentation:* 333 images
* *Annotation format:* YOLO-style (YOLOv11-compatible; class x_center y_center width height normalized format)
* *Classes:*

  1. puma
  2. adidas

*Key Roboflow pre-processing applied to each image:*

* Auto-orientation of pixel data (EXIF orientation stripped)
* Resize to *512×512* (stretch)

*Augmentations applied to generate additional images:*

* 50% probability horizontal flip
* 50% probability vertical flip
* One of the following 90° rotations chosen equally: none, clockwise, counter-clockwise, upside-down
* Random crop between 0% and 20% of the image
* Random rotation between *-15°* and *+15°*

Roboflow (or equivalent) was used to perform augmentation and export the final dataset in YOLO format.

---

## Preprocessing Notes

* Images were resized by stretching to 512×512 during preprocessing. This keeps a consistent input resolution for training.
* Roboflow's augmentation produced varied samples that mimic real-world variation (orientation, partial crops, rotation), improving generalization.
* Very small logos (extremely tiny bounding boxes) were avoided where possible, since detection performance drops for extremely small objects with limited examples.

---

## Train / Val / Test Split

The final dataset of *352* images was split as follows:

* *Train:*  → *333 images*
* *Validation:*  → *10 images*
* *Test:*  → *9 images*

> Splits are ready in data/ with corresponding images/ and labels/ subfolders for each split.

---

## Model Recommendation & Training Setup

*Recommended model family:* YOLO (Ultralytics) — yolov11n or yolov8s or yolov8n for speed/efficiency. YOLOv11 format annotations are compatible when exported in standard YOLO-format.

*Why YOLO?*

* Fast to train and infer
* Works well for logos (small objects with clear shapes)
* Built-in utilities for video inference and export

*Hardware used for training:* Google Colab with GPU (CUDA enabled). Use a Colab GPU runtime (Tesla T4 / P100 / V100 where available).

*Suggested hyperparameters (example):*

* epochs: 300 (adjust depending on validation mAP)
* imgsz: 640 (YOLO will internally resize from 512 input; higher shapes may improve accuracy at cost of speed)
* batch: 16 (Colab GPU memory dependent)
* optimizer: Adam / SGD (default from Ultralytics is usually fine)

**Example data.yaml**

yaml
train: ../data/images/train
val: ../data/images/valid
test: ../data/images/test
nc: 2
names: ["puma", "adidas"]


---

## Training Steps (Colab)

1. Create a Colab notebook and select *GPU* runtime.
2. Install dependencies (example uses Ultralytics YOLO and common libs):

bash
pip install ultralytics roboflow opencv-python-headless pandas


3. Upload dataset to Colab or mount Google Drive (recommended for persistence):

python
from google.colab import drive
drive.mount('/content/drive')


4. Start training (example using Ultralytics YOLO API):

python
model = YOLO("yolo11n.pt")

results = model.train(
    data=r"/content/Brand_Logo_Detection-3/data.yaml",
    epochs=300,
    imgsz=640,
    device=device  # auto-selected GPU/CPU
)


5. Monitor training metrics (loss, mAP at 0.5) and adjust epochs if necessary. Save the best model weights (e.g. runs/detect/train/weights/best.pt).

> If you use a different YOLO implementation (v11), follow its training CLI/API. The repository contains scripts/Model_Train.ipynb with a ready-to-run example for Colab.

---

## Inference (Video) & CSV Output

*Goal:* Run model on video frames and generate a labeled video + CSV with detection details.

*Sample inference workflow (high-level):*

1. Download / place the sample video in inputs/.
2. Load the trained model weights.
3. Iterate frames, run model.predict on each frame.
4. For each detection, write a line into detections.csv with columns:

   * frame_no, brand, confidence, x1, y1, x2, y2
5. Draw bounding boxes + labels onto frames and write frames to output/output_labeled_video.mp4.

*Note:* Due to video size limitations during submission, only a *sample video* was processed and included in output/.

A ready-to-run script scripts/detect_video.py is included in this repo and expects:

bash
python scripts/detect_video.py --weights runs/detect/train/weights/best.pt --source inputs/sample_video.mp4 --output output/output_labeled_video.mp4 --csv output/detections.csv


---

## Repository Structure


..Main-Directory/
├── data/                    # dataset with train/valid/test images and labels
│   ├── train/
│   ├── valid/
│   └── test/
├── models/                  # saved weights (best.pt)
├── runs/                    # all run time model data and performance metrix
├── scripts/
│   ├── Model_Train.ipynb    # training and testing code
├── inputs/                  # sample input video(s)
├── output/                  # result video + CSV
├── README.md
├── requirements.txt
└── data.yaml
└── yolo11n.pt

## Notes & Limitations

* The dataset was augmented and stretched to 512×512; stretching may distort logos slightly — consider maintaining aspect ratio in production.
* Extremely small logos are hard to detect reliably without many examples or higher-resolution inputs.
* Only a sample video was processed due to submission size constraints; full video inference instructions are included in the repo.

---

## How to Reproduce (Quick Commands)

1. Install requirements:

bash
pip install -r requirements.txt


2. Train (Colab / local):

bash
python scripts/train.py --data data.yaml --epochs 300 --imgsz 640 --batch 16


3. Inference on sample video:

bash
python scripts/Model_Train.ipynb --weights models/best.pt --source inputs/sample_video.mp4 --output output/output_labeled_video.mp4 --csv output/detections.csv


---

## Contact

If you need help reproducing the results or want the Colab notebook used for training, contact:

* *Kishan Patel*

---

Thank you — this README documents the dataset preparation, augmentation, training and inference steps used to produce the submitted sample video and CSV.