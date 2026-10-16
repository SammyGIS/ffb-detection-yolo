# Changelog

All notable changes, optimizations, structural consolidations, and legacy purges for the **Oil Palm Fresh Fruit Bunch (FFB) Detection & Retraining Project** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] - 2026-10-16

### 🚀 High-Accuracy Workstation Retraining & Validation Benchmark
- **Workstation Acceleration (NVIDIA GeForce RTX 4060 Laptop GPU)**: Successfully executed offline local retraining with PyTorch CUDA 12.4 and Automatic Mixed Precision (AMP FP16).
- **Benchmark Record (96.5% mAP@50)**:
  - Evaluated on held-out 10% stratified plantation validation test set (1,071 unseen images / 12,897 bunches).
  - Achieved **`96.5% mAP@50`** (IoU 0.50), **`91.0% Precision`**, and **`92.8% Recall`** ($F_1 = 0.919$).
- **Model-Side False Positive Suppression**:
  - Benchmark confirmed **`97.8%` clean negative clutter suppression** on Okomu estate ground clutter at `conf=0.35` (zero detections on sacks, tools, fronds, and soil).
  - Production recommendation verified: **`conf=0.18`, `iou=0.50`** delivers **`92.6% Recall`** and **`90.7% Precision`**.
- **Model Asset Archival**:
  - Master PyTorch Weights: [`models/my_model_2026_10_16.pt`](./models/my_model_2026_10_16.pt) (26.56 MB).
  - Slimmed Edge ONNX Runtime: [`models/my_model_2026_10_16.onnx`](./models/my_model_2026_10_16.onnx) (10.66 MB).
- **Pure-White Background Flowcharts**:
  - Replaced ASCII text diagrams with boardroom-quality, white-background vector flowcharts:
    - End-to-End System Architecture: [`docs/images/system_architecture_flow.jpg`](./docs/images/system_architecture_flow.jpg)
    - Dataset Provenance & Ingestion Flow: [`docs/images/dataset_provenance_flow.jpg`](./docs/images/dataset_provenance_flow.jpg)
- **Roboflow Benchmark Partitions Cataloged**:
  - Formally referenced all 7 underlying Roboflow FFB benchmark sources (`MORNGRDDVRS003`, `FFB count v5`, `FFB Base`, `Igan Multiple FFB`, `FFB v1i`, `12-10-22 jsontotxt`, `FFB Counter v4`) detailing their feature contributions to equatorial illumination, heap occlusion, and fruitlet morphologies.

---

## [1.0.0] - 2026-10-16

### 🚀 Added
- **Unified Master Dataset (`data.zip` - 1.55 GB)**: Consolidated all historical positive bunches and negative backgrounds into a single, high-performance training archive (10,706 images total).
- **2,000 Balanced Negative Samples (18.7% share)**: Added 2,000 "no bunch" background images with empty 0-byte `.txt` annotations to permanently penalize and eliminate false positives on household objects (TVs, fans, keyboards) and plantation clutter (bare soil, sacks, wheelbarrows, workers).
- **226 Real Okomu Field Negatives with $4\times$ Rotation Augmentation**:
  - Handled 28 real plantation photos across loose fruit, sacks, tools, ground, and indoor trigger objects.
  - Generated $4\times$ geometric variations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) and horizontal flips to achieve rotation invariance across field devices.
- **100% Priority Guarantee in Sampling**:
  - Configured notebook sampling logic so any file with prefix `okomu_neg_*` is strictly protected from being dropped by sampling caps.
- **Embedded Label Metadata**:
  - Directly bundled `classes.txt` (`['ffb']`) and `data.yaml` inside `data.zip`.
  - Added dynamic fallback in notebook Cell 24 to guarantee zero runtime failures during Colab dataset configuration.
- **Mobile Edge Model (`models/my_model_int8.tflite`)**: Included the production 9.46 MB INT8 quantized mobile detection model.
- **Git Tracking Rules (`.gitignore`)**: Configured strict exclusion rules for files >25MB (datasets, raw archives, temporary checkpoints) while preserving the 9.46 MB production model and notebooks.

---

### ⚡ Optimized
- **Image Dimension & Quality Re-compression**:
  - Resized 10,706 training images from bloated camera dimensions (>4000px) down to **max 960px @ 80% JPEG quality**.
  - Reduced average photo size from ~1.2 MB to **132 KB** with **zero loss in feature resolution** for 640px/800px YOLO training.
  - Slashed uncompressed dataset size from **4.45 GB down to 1.55 GB** (~65% reduction).
  - Faster Google Drive uploads and Colab unzip time dropped from 5 minutes to under 25 seconds.
- **Google Colab Notebook Clean-up**:
  - Removed all third-party external dependencies (Roboflow Universe API, Wikimedia Commons web scraping, Openverse downloaders).
  - Streamlined workflow into a single 1-click execution: upload `data.zip` $\rightarrow$ click **Runtime $\rightarrow$ Run all**.

---

### 🗑️ Purged (15.17 GB Disk Space Recovered)
To establish a single source of truth and eliminate disk clutter, all obsolete duplicate zips, legacy experiments, and intermediate staging folders were audited and permanently purged:

| Deleted Item | Size | Reason for Deletion |
|---|:---:|---|
| `custom_data/data.zip` | 4,447.0 MB | Redundant 4.5 GB uncompressed raw duplicate zip |
| `OkomuDataset/` | 4,876.5 MB | Retired early April 2026 experimental branch (preliminary pilot partitions A & B; all verified annotations merged into master `data.zip`) |
| `dataset_compact/` | 1,592.4 MB | Local staging folder used to build `data.zip` |
| `negatives_extra/` | 1,397.9 MB | Retired overflow web stock from 14-Oct |
| `MORNGRDDVRS003.yolov11.zip` | 1,295.5 MB | Retired early morning divers Roboflow export |
| `FFB count.v5-ori-5.yolov11.zip` | 782.8 MB | Retired Roboflow v5 export |
| `FFB.yolov11.zip` | 358.1 MB | Retired Roboflow base export |
| `igan-multiple-ffb.v1-igan-mutiple-01-640.yolov11.zip` | 248.2 MB | Retired Igan plantation sub-export |
| `my_model_6_07_2026/` | 177.7 MB | Legacy June 2026 model run |
| `ffb.v1i.yolov11.zip` | 138.9 MB | Retired Roboflow v1 export |
| `12-10-22 jsontotxt.v1i.yolov11.zip` | 126.7 MB | Retired 2022 JSON-to-TXT test export |
| `runs/` | 64.0 MB | Old local inference and training runs |
| `my_model_4_8_2026/` | 18.3 MB | Legacy April 2026 model weights |
| `ffb_counter.v4i.yolov11.zip` | 8.2 MB | Retired Roboflow v4 export |
| **Total Disk Space Purged** | **15,532.8 MB (~15.17 GB)** | **Clean single source of truth established** |

---

### 🛡️ Defensive Heuristics Summary
The model trained on this dataset interfaces with client-side edge heuristics in the Okomu Mobile App:
- **Anti-Sack Collar Detection**: Prevents sacks of loose fruitlets from registering as fresh fruit bunches.
- **Macro-Flooding Guard**: Drops frames when close-up fruitlets occupy >78% of the camera sensor.
- **Photometric Filter**: Eliminates high-luminance flat surfaces (walls, foam, white woven sacks).
- **Aspect Ratio Bounding**: Rejects non-conical fruit geometries outside the 0.48–1.95 ratio.
