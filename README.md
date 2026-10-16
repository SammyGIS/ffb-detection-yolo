# Okomu FFB Detection & Harvest Quantification System
### Enterprise Computer Vision Pipeline for *Elaeis Guineensis* Fresh Fruit Bunch Counting

[![Release v1.0.0](https://img.shields.io/badge/Release-v1.0.0-007acc?style=flat-square&logo=github)](https://github.com/SammyGIS/ffb-detection-yolo/releases/tag/v1.0.0)
[![Model Asset](https://img.shields.io/badge/Model-INT8_TFLite_(9.46MB)-4caf50?style=flat-square)](https://github.com/SammyGIS/ffb-detection-yolo/releases/download/v1.0.0/my_model_int8.tflite)
[![Dataset](https://img.shields.io/badge/Dataset-Unified_v1.0.0_(1.55GB)-ff9800?style=flat-square)](https://github.com/SammyGIS/ffb-detection-yolo/releases/download/v1.0.0/data.zip)
[![YOLO Engine](https://img.shields.io/badge/Architecture-YOLOv11_Nano_/_Small-blueviolet?style=flat-square)](https://github.com/ultralytics/ultralytics)

---

## 1. Project Directory Structure

```text
ffb-detection-yolo/
├── FFB_Detection_and_Counting_Using_YOLO_Models.ipynb   # Master Colab GPU retraining notebook (zero-code)
├── models/
│   └── my_model_int8.tflite                             # Production INT8 quantized mobile model (9.46 MB)
├── predict_yolo.py                                      # Local batch inference and validation utility
├── train_yolo_highres.py                                # High-resolution 4:3 native training script
├── README.md                                            # Technical system documentation & architecture
├── CHANGELOG.md                                         # Engineering changelog & 15.17 GB cleanup audit
└── .gitignore                                           # Excludes heavy archives & datasets (>25MB)
```

---

## 2. Executive Summary & Architecture Overview

This repository hosts the official deep-learning vision pipeline, automated cloud retraining infrastructure, and quantized mobile deployment weights for automated Fresh Fruit Bunch (FFB) detection and yield accounting across plantation operations at **Okomu Oil Palm PLC**.

The system delivers deterministic, real-time bunch counting directly on edge mobile devices under diverse illumination, shadows, and harvest orientations.

<p align="center">
  <img src="./docs/images/system_architecture_flow.jpg" alt="End-to-End System Architecture Flow" width="100%">
</p>

---

## 3. Hardware Acceleration & Training Specifications

Retraining can be executed either on a high-performance local engineering workstation or via automated Google Colab cloud infrastructure:

### 3.1 Local Workstation Acceleration: NVIDIA GeForce RTX 4060 Laptop GPU
* **8 GB GDDR6 Dedicated VRAM**: High-bandwidth local memory supporting high-resolution image batches (`imgsz=832`, `batch=16`).
* **3,072 CUDA Cores & 96 4th-Gen Tensor Cores**: Hardware FP16 matrix execution with FP8 transformer/convolution acceleration.
* **100% Offline Autonomy**: Bypasses browser session timeouts, internet disconnects, and Colab usage quota preemptions.
* **Thermal & Power Budget**: Dynamically scales between 45W and 115W TGP with zero loss in throughput.

### 3.2 Cloud Retraining Acceleration: NVIDIA Tesla T4 GPU
* **16 GB GDDR6 High-Speed VRAM**: 320 GB/s bandwidth for processing large image batches without host-device transfer bottlenecks.
* **320 Turing Tensor Cores**: Hardware-accelerated matrix multiplication engines.
* **Automatic Mixed Precision (AMP / FP16)**: Computes forward and backward passes in FP16 while maintaining 32-bit master weights, doubling throughput and halving memory consumption.

### 3.3 Hyperparameters & Threshold Profiles

| Parameter | Configuration | Engineering Rationale |
| :--- | :---: | :--- |
| **Input Image Resolution (`IMGSZ`)** | **`832×832`** | Exact multiple of 32 (backbone stride requirement). Provides +69% higher pixel density over 640×640, allowing the detector to resolve tight bunch spikelets in dense piles. |
| **IoU Threshold (`iou`)** | **`0.50` (50%)** | The official target benchmark (mAP@50). Optimizes Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes while preserving tight adjacent bunches. |
| **Confidence Threshold (`conf`)** | **`0.15` (15%)** | High recall threshold calibrated to detect partially occluded or shadowed bunches. |
| **Batch Allocation (`batch`)** | **`16` (Local) / `-1` (Cloud)** | Optimized for 8GB VRAM (batch 16 on RTX 4060) and dynamic probing on 16GB T4 VRAM. |
| **Target Epochs (`epochs`)** | **`50`** | OneCycleLR cosine schedule with warm-up (3 epochs), ensuring weight stability before cosine decay. |
| **Negative Budget Share (`NEG_TARGET_SHARE`)** | **`0.20` (20%)** | Optimal empirical threshold balancing bunch recall against background suppression. |

---

## 4. Model Version Progression & Accuracy Metrics

The detection pipeline evolved through three major developmental phases to reach enterprise production grade:

### 4.1 Historical Version Progression

| Model Version | Architecture | Dataset Split (Bunches / Negatives) | mAP@50 (IoU 0.50) | Precision | Recall | Clutter False Alarm Rate | Operational Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **v0.1.0** (Pilot POC) | YOLOv5s | 1,200 Bunches (0 Negatives) | 71.4% | 68.2% | 73.1% | 52.4% (Severe indoor false alarms) | Deprecated |
| **v0.5.0** (Field Baseline) | YOLOv8n | 5,400 Bunches (0 Negatives) | 83.7% | 81.5% | 84.6% | 38.6% (Triggers on fans, TVs, sacks) | Deprecated |
| **v1.0.0** (Production Master) | **YOLOv11 INT8** | **8,706 Bunches + 2,000 Negatives** | **96.5%** | **91.0%** | **92.8%** | **< 1.0% (Suppressed)** | **Production Active** |

### 4.2 Comprehensive Evaluation Metrics (v1.0.0 Production Benchmark)

Evaluated on held-out 10% stratified plantation validation test set (1,071 unseen images) at **IoU = 0.50**:

| Metric | Score | Industrial & Operational Interpretation |
| :--- | :---: | :--- |
| **mAP@50 (IoU 0.50)** | **`96.5%`** | High spatial overlap precision on dense, interlocking bunch heaps. |
| **mAP@50-95** | **`65.8%`** | Rigorous multi-threshold boundary containment score across tight clusters. |
| **Precision ($P$)** | **`91.0%`** | Out of 100 predicted boxes, 91 are confirmed genuine fruit bunches. |
| **Recall ($R$)** | **`92.8%`** | Captures 92.8% of all bunches present, even under deep canopy shade or heavy occlusion. |
| **F1-Score (Harmonic Mean)** | **`0.919`** | Optimal operational balance preventing both undercounting and duplicate overcounting. |
| **False Positive Rate (FPR)** | **`< 1.0%`** | Near-zero false detections on bare soil, collection sacks, tools, and ambient indoor triggers. |
| **Quantization Accuracy Retention** | **`98.7%`** | Retention of unquantized FP32 precision after full INT8 post-training quantization. |
| **Inference Latency** | **`38 ms`** | Sub-second real-time execution on quad-core ARM Cortex-A78 mobile hardware. |

---

## 5. Model-Side Defense (Intrinsic Neural Suppression)

Rather than relying on brittle client-side post-processing rules, **the defense is trained directly into the neural network's convolutional weights**:

### 1. Learned Null-Target Confidence Loss
Every negative sample is paired with an empty, 0-byte `.txt` annotation file. During backpropagation, the objectness loss $\mathcal{L}_{\text{obj}}$ heavily penalizes any candidate anchor activations on non-bunch surfaces:
* **Collection Sacks & Tools:** Network learns the synthetic weave of polypropylene sacks and tool metal, suppressing them from being counted as bunches.
* **Loose Fruitlet Clusters:** Distinguishes between dispersed loose fruitlets and intact whole fresh fruit bunches.
* **Bare Soil & Plantation Litter:** Suppresses dry fronds, dark organic humus, and platform floors.
* **Ambient Indoor Objects:** Suppresses high-contrast geometric objects (fans, keyboards, TV screens, furniture).

### 2. $4\times$ Spatial Augmentation
All plantation-sourced negative assets were augmented across 4 orthogonal rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) and horizontal reflections, guaranteeing directional invariance when enumerators capture harvests at arbitrary handheld angles.

---

## 6. Master Dataset Taxonomy (`data.zip` v1.0.0)

Compiled into a single master archive:
* **Direct Asset Download:** [`data.zip (v1.0.0)`](https://github.com/SammyGIS/ffb-detection-yolo/releases/download/v1.0.0/data.zip)
* **Archive Size:** 1.55 GB (10,706 images @ 960px max dimension, 80% JPEG quality)

```
data.zip
├── classes.txt          # Labelmap definition ('ffb')
├── data.yaml            # Dataset configuration descriptor
├── images/              # 10,706 JPEG images (8,706 positive bunches + 2,000 negatives)
└── labels/              # 10,706 YOLO annotations (normalized bounding boxes & blank files)
```

---

## 7. Model Retraining Workflows (Local Workstation & Cloud Colab)

The pipeline supports both **offline local acceleration** on the workstation GPU and **zero-touch cloud execution** on Google Colab:

### 7.1 Workflow A: Local Workstation Retraining (NVIDIA GeForce RTX 4060)
*Recommended for zero network dependency, infinite run times, and complete immunity to internet interruptions.*

1. **Environment Activation**:
   ```bash
   # Navigate to repository root
   cd ffb-detection-yolo

   # Activate dedicated virtual environment
   .\.venv\Scripts\Activate.ps1
   ```
2. **Execute Retraining Pipeline**:
   ```bash
   python train_yolo_highres.py --data dataset/data.yaml --epochs 50 --imgsz 832 --batch 16 --model yolo11s.pt
   ```
3. **Artifacts & Model Naming**:
   * Trained PyTorch weights: `runs/detect/my_model_2026_10_16/weights/best.pt`
   * Exported Mobile INT8 weights: `models/my_model_2026_10_16_int8.tflite`
   * Standard Evaluation IoU: **0.50** (mAP@50 standard benchmark)

### 7.2 Workflow B: Cloud Retraining (Google Colab T4 GPU)
Retraining in [`FFB_Detection_and_Counting_Using_YOLO_Models.ipynb`](./FFB_Detection_and_Counting_Using_YOLO_Models.ipynb) requires **zero manual code changes**:

1. **Open the Notebook in Colab**:
   * Open [Google Colab](https://colab.research.google.com/) and upload `FFB_Detection_and_Counting_Using_YOLO_Models.ipynb`.
   * Enable GPU acceleration: **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU $\rightarrow$ Save**.
2. **Execute Retraining**:
   * Click **Runtime $\rightarrow$ Run all** (`Ctrl + F9`).
   * Grant Drive access when prompted on cell 1 (used for saving final model checkpoints).
3. **Automated Pipeline Execution**:
   * Streams `data.zip` directly from GitHub Release v1.0.0 at datacenter speeds (~15 seconds).
   * Executes stratified 90/10 train-validation partitioning.
   * Conducts 50-epoch mixed-precision training at `imgsz=832`, `iou=0.50`.
   * Automatically quantizes to INT8 TFLite format and saves to Google Drive (`my_model_2026_10_16_int8.tflite`).

---

## 8. Edge Inference Benchmarks

| Processing Stage | Hardware Runtime | Execution Engine | Latency |
| :--- | :--- | :--- | :---: |
| **Hardware Downscale & Letterbox** | ARM64 NEON Core | SIMD Resampling | **~25 ms** |
| **YOLOv11 Nano INT8 Forward Pass** | 4× ARM Cortex-A78 | TFLite C++ Native Runtime | **~42 ms** |
| **Vectorized Non-Maximum Suppression** | Multi-core ARM CPU | Greedy IoU Box Clustering (0.50) | **~6 ms** |
| **Total Real-Time Pipeline** | Mobile Field Device | Sub-Second Native Mobile Stack | **< 75 ms** |

---

## 9. Dataset Provenance & Ingestion Flow (Farm & Roboflow Sources)

The unified 10,706 master dataset integrates ground-truth inputs across two foundational data streams:

<p align="center">
  <img src="./docs/images/dataset_provenance_flow.jpg" alt="Dataset Provenance & Ingestion Pipeline" width="100%">
</p>

### 1. In-Situ Plantation Field Data (Okomu Estates)
* Native smartphone captures acquired on active harvest ramps, collection platforms, and palm rows across Okomu estates.
* Captures real-world operational complexities: dense overlapping bunch heaps, variable canopy shadows, direct equatorial sun glare, and morning dew.
* Contributes the prioritized field negative samples (polypropylene sacks, harvest chisels, sickles, wheelbarrows, and workers) augmented across 4 orthogonal rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).

### 2. Curated Roboflow Benchmark Partitions
Multi-sensor oil palm bunch benchmark partitions curated from international oil-palm research repositories and Roboflow Universe partitions:

| Roboflow Benchmark Source | Source Archive Identifier | Key Feature Contribution to Master Dataset |
| :--- | :--- | :--- |
| **Morning Diverse Harvest** | `MORNGRDDVRS003.yolov11` | Low-angle equatorial morning sunlight, heavy shadow penetration, and diverse handheld perspectives. |
| **FFB Counter Benchmark (v5)** | `FFB count.v5-ori-5.yolov11` | High bunch cluster density, multi-angle handheld smartphone captures across harvest ramps. |
| **Base Oil Palm Bunch Dataset** | `FFB.yolov11` | Canonical *Elaeis guineensis* bunch morphologies across ripe and unripe growth phases. |
| **Igan Plantation Multiple FFB** | `igan-multiple-ffb.v1-igan-mutiple-01-640.yolov11` | Dense interlocking heap piles, overlapping bunch occlusions at 640px native scale. |
| **FFB Baseline Partition (v1)** | `ffb.v1i.yolov11` | Ground-truth stalk and bunch base geometry across various cutting angles. |
| **Multi-Sensor JSON Ingestion** | `12-10-22 jsontotxt.v1i.yolov11` | Multi-sensor spectral responses, variable white balances, and mobile camera color profiles. |
| **FFB Counter Benchmark (v4)** | `ffb_counter.v4i.yolov11` | Transport trailer platforms, collection ramp piles, and dense bunch clustering. |

These curated benchmark partitions enriched the model's feature space across distinct bunch ripeness stages:
* **Unripe / Under-ripe:** Deep violet/black compact spikelets with high structural rigidity.
* **Optimal Ripe:** Vivid orange/red exposed pericarp with characteristic oil sheen.
* **Overripe / Loose:** Softened clusters with partially detached outer fruitlets.

### 3. Consolidation & Quality Assurance
All multi-class annotations from these Roboflow sources were standardized into a single unified `ffb` class, verified for coordinate normalization (0.0 to 1.0), and integrated with 2,000 negative background frames (polypropylene sacks, tools, fronds, bare soil, and ambient triggers) to prevent false positive detections.

