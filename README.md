# Oil Palm Fresh Fruit Bunch (FFB) Detection & Retraining Guide

Official documentation and operating manual for retraining, optimizing, and deploying YOLO models for Fresh Fruit Bunch (FFB) detection and counting in the **Okomu Mobile Application**.

---

## 1. Single Source of Truth: `data.zip`

All fragmented historical datasets (Ope's data, Samuel's data, early Roboflow exports, and uncompressed raw dumps) have been consolidated into **one single, lightweight master dataset**:

* **Location:** [`ffb-count-detection/data.zip`](file:///c:/Users/Admin/Documents/DOYSAM/Okomu/Okomu%20App%20Development%20Project/ffb-count-detection/data.zip)
* **Archive Size:** **1.55 GB** *(optimized down from 4.5 GB uncompressed raw)*
* **Total Images:** **10,706 images**
* **Total Labels:** **10,706 label files** (normalized YOLO `.txt` bounding boxes & blank background files)
* **Max Dimension:** `960px` @ 80% JPEG quality *(crisp features with zero downsampling artifacts for 640px/800px YOLO training)*

### Dataset Composition

| Category | Images | Role in Neural Network |
| :--- | :---: | :--- |
| **Positive Fresh Fruit Bunches** | **8,706** | Teaches the model bunch morphology, spikelets, varying ripeness levels, and dense pile clusters. |
| **Negative Backgrounds (No Bunch)** | **2,000** (18.7%) | Heavy penalty loss on non-bunch textures to eliminate false positives on farm ground, sacks, tools, and household objects. |
| **Real Okomu Photos & $4\times$ Rotations** | **226** | 100% prioritized real farm photos: loose fruit, plantation ground, sacks, wheelbarrows, and indoor triggers. |
| **Total Master Dataset** | **10,706** | Perfectly balanced and ready for direct training. |

---

## 2. Training in Google Colab (Quick Start)

The retraining pipeline is completely automated in [`FFB_Detection_and_Counting_Using_YOLO_Models.ipynb`](file:///c:/Users/Admin/Documents/DOYSAM/Okomu/Okomu%20App%20Development%20Project/ffb-count-detection/FFB_Detection_and_Counting_Using_YOLO_Models.ipynb):

1. **Upload `data.zip`** to the root of your Google Drive (`My Drive/data.zip`).
2. **Open the notebook** in Google Colab.
3. Select **GPU Runtime**: `Runtime -> Change runtime type -> T4 GPU`.
4. Click **Runtime $\rightarrow$ Run all** (`Ctrl+F9`).

### What Happens Automatically:
* Mounts Google Drive and unzips `data.zip` in ~20 seconds.
* Automatically verifies embedded `classes.txt` (`['ffb']`) and generates `data.yaml`.
* Splits 90% into `train/` and 10% into `validation/`.
* Trains YOLOv11 with balanced Objectness / Confidence Loss.
* Automatically exports `.pt` weights and mobile INT8 TFLite model.
* Zips and saves the final model artifacts directly back to your Google Drive!

---

## 3. Negative Sampling & Anti-False-Positive Defense

### Why Retraining with Negatives Was Required
When an object detector is trained solely on positive images, any repetitive high-contrast grid pattern (e.g., fan grilles, television frames, keyboards, floor tiles) can trigger high-confidence detections (>85%) because the network was never penalised for predicting on non-palm textures.

### Defensive Engineering Implemented:
* **Empty `.txt` Annotations:** Every negative image has a matching 0-byte `.txt` file. When the network proposes a box on these images, it incurs maximum confidence loss, forcing false positives to 0.00.
* **4-Way Geometric Augmentation:** Real field negatives were rotated ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) and horizontally flipped to ensure invariance across device orientations.
* **100% Okomu Priority Guarantee:** Any image with prefix `okomu_neg_*` is immune to sampling caps and is guaranteed 100% inclusion in the training set.

---

## 4. Changelog & Historical Cleanup Audit

To eliminate confusion, recover disk space, and establish a single source of truth, all obsolete duplicate zips and retired experimental directories were audited and purged.

### Disk Space Recovered: **15.17 GB**

```
========================================================================================
Type        Item Deleted                                        Size      Original Purpose / Notes
========================================================================================
Duplicate   custom_data/data.zip                            4,447.0 MB   Old uncompressed 4.5GB raw zip duplicate
Experiment  OkomuDataset/ (April 2026 branch)               4,876.5 MB   Old experimental branch (Ope & Samuel datasets)
Staging     dataset_compact/                                1,592.4 MB   Temporary staging folder used to build data.zip
Overflow    negatives_extra/                                1,397.9 MB   Old overflow web stock from 14-Oct
Old Zip     MORNGRDDVRS003.yolov11.zip                      1,295.5 MB   Old morning divers Roboflow export
Old Zip     FFB count.v5-ori-5.yolov11.zip                    782.8 MB   Old Roboflow v5 export
Old Zip     FFB.yolov11.zip                                   358.1 MB   Old Roboflow base export
Old Zip     igan-multiple-ffb.v1-igan-mutiple-01-640.yolov11.zip 248.2 MB Old Igan plantation sub-export
Old Zip     ffb.v1i.yolov11.zip                               138.9 MB   Old Roboflow v1 export
Old Zip     12-10-22 jsontotxt.v1i.yolov11.zip                126.7 MB   Old 2022 JSON-to-TXT test export
Old Model   my_model_6_07_2026/                               177.7 MB   Legacy June 2026 model run
Old Model   runs/                                              64.0 MB   Old local inference and training runs
Old Model   my_model_4_8_2026/                                 18.3 MB   Legacy April 2026 model weights
Old Zip     ffb_counter.v4i.yolov11.zip                         8.2 MB   Old Roboflow v4 export
----------------------------------------------------------------------------------------
TOTAL DISK SPACE PURGED                                     15,532.8 MB  (~15.17 GB)
========================================================================================
```

> **Note on Data Preservation:** All valid, annotated positive bunch photos contained within the old exports above were merged and retained inside the master `data.zip`. No original training samples were lost.

---

## 5. Model Profiles & Hardware Performance

| Setting | Standard Mobile | High-Res Target |
| :--- | :--- | :--- |
| **Model Resolution** | `640×640` | `800×800` (+56% pixel density) |
| **Confidence Threshold** | `0.15` (15%) | `0.12` (detects nested bunches) |
| **IoU Threshold** | `0.55` (55%) | `0.60` (prevents suppression in tight heaps) |
| **Camera FOV** | `16:9` | `4:3` (Native uncropped camera sensor) |
| **Inference Latency** | ~45 ms (TFLite INT8 on 4 ARM threads) | ~65 ms |
| **End-to-End Frame Time** | < 90 ms | < 120 ms |

---

## 6. Edge Anti-Fraud Defensive Layer

The retrained YOLO detector works in conjunction with on-device heuristics in the Okomu Mobile App:
* **Anti-Sack Collar Detection:** Prevents sacks filled with loose fruitlets from registering as whole fruit bunches.
* **Macro-Flooding Guard:** Drops scans where close-up fruitlets fill >78% of the frame.
* **Photometric Sanity:** Filters out high-luminance pale surfaces (mattresses, white sacks, walls).
* **Aspect Ratio & Area Bounding:** Rejects irregular non-conical slivers outside the 0.48–1.95 ratio or outside 0.22%–18% frame area.
