# Terrain Detection and Segmentation for Autonomous Vehicle Navigation

A lightweight multi-modal terrain segmentation framework (Trifecta Fusion Model) that fuses RGB appearance, depth geometry, and surface normals for robust pixel-level terrain understanding — evaluated across structured urban, semi-urban, and unstructured off-road environments.

> Key result: Under zero-shot cross-dataset transfer from Cityscapes to RELLIS-3D, the Trifecta model achieves 0.600 mIoU versus 0.395 for RGB-only — a 52% relative improvement.

Published research conducted at BITS Pilani Dubai Campus in collaboration with faculty supervisors.

---

## The Problem

RGB-only segmentation models perform well in structured urban environments (Cityscapes, KITTI) where lighting is consistent and road geometry is well-defined. However, their performance degrades substantially in off-road settings — mud, vegetation, uneven soil — where visually similar surfaces differ in traversability and appearance cues alone are unreliable.

This work addresses that gap through principled multi-modal fusion.

---

## The Trifecta Fusion Model (TFM)

TFM is a unified encoder-decoder architecture built on DeepLabV3+ with a MobileNetV3-Large backbone, modified to accept 7 or 8 channel inputs (RGB + depth + surface normals + optional IMU) and augmented with a dedicated modality-aware fusion module.

The key architectural contribution is **dynamic adaptive gating**: rather than naive concatenation or fixed attention, TFM computes per-modality importance weights at each forward pass using global average pooling and a lightweight MLP with softmax output. This allows the model to autonomously downweight degraded sensor streams at inference time — a capability that prior multi-stream architectures lack.

### Trifecta Fusion Module — Four Stages

| Stage | Operation |
|-------|-----------|
| Early channel fusion | All modalities concatenated and projected via 1x1 Conv + BN + ReLU |
| Geometry-guided spatial fusion | Depth features gate RGB via element-wise sigmoid weighting |
| Surface normal refinement | Normal features injected via learnable conv to refine terrain boundaries |
| Kinematic fusion (IMU) | IMU motion features stabilise predictions under pitch, roll, blur (KITTI only) |

### Input Configuration per Dataset

| Dataset | Channels | Modalities |
|---------|----------|-----------|
| KITTI Road | 8 | RGB + Depth + Normals + IMU |
| Cityscapes | 7 | RGB + Depth + Normals |
| RELLIS-3D | 7 | RGB + Depth + Normals |

---

## Results

### In-Dataset Evaluation

| Variant | Modalities | Cityscapes mIoU | KITTI mIoU | RELLIS-3D mIoU |
|---------|-----------|----------------|------------|----------------|
| A | RGB | 0.8702 | 0.934 | 0.7947 |
| B | RGB + Depth | 0.8704 | 0.922 | 0.752 |
| C | RGB + Depth + Normals (Trifecta) | 0.8634 | 0.930 | 0.759 |

In structured environments (Cityscapes, KITTI), all variants achieve comparable performance — confirming TFM introduces no performance penalty when appearance cues are reliable. On unstructured RELLIS-3D, Trifecta outperforms both baselines.

### Cross-Dataset Generalisation (Zero-Shot Transfer)

| Train → Test | RGB | RGB-D | Trifecta |
|-------------|-----|-------|----------|
| Cityscapes → KITTI Road | 0.5522 | 0.5300 | **0.6034** |
| Cityscapes → RELLIS-3D | 0.3947 | 0.2882 | **0.5998** |
| KITTI Road → RELLIS-3D | 0.5120 | 0.3710 | **0.5480** |
| RELLIS-3D → Cityscapes | 0.4010 | 0.3360 | **0.4890** |

The most significant result: Cityscapes → RELLIS-3D transfer, where RGB-only collapses to 0.395 mIoU while TFM achieves 0.600 — a **52% relative improvement**. Notably, RGB+Depth (0.288) underperforms RGB-only on this transfer, confirming that raw depth encodes sensor-specific noise that compounds under domain shift. TFM's surface normals and adaptive gating compensate by providing higher-level geometric abstraction.

---

## Architecture

```
Input: RGB (3ch) + Depth (1ch) + Surface Normals (3ch) + IMU (1ch, KITTI only)
     |
Modified DeepLabV3+ Encoder (MobileNetV3-Large)
First conv layer widened to accept 7/8 channels
     |
Trifecta Fusion Module
  - Early channel fusion (1x1 Conv)
  - Geometry-guided spatial fusion (depth gates RGB)
  - Surface normal refinement
  - Kinematic fusion / IMU injection (KITTI only)
  - Adaptive gating (GAP + MLP + Softmax per modality)
     |
ASPP (Atrous Spatial Pyramid Pooling)
     |
DeepLabV3+ Decoder + Skip Connections
     |
Pixel-level Terrain Segmentation Mask
```

---

## Datasets

| Dataset | Environment | Classes | Modalities Available |
|---------|-------------|---------|---------------------|
| Cityscapes | Structured urban (50 European cities) | 30 (mapped to 7) | RGB, stereo disparity → depth + normals |
| KITTI Road | Semi-urban highway | 2 (road/non-road) | RGB, LiDAR depth, IMU/OXTS |
| RELLIS-3D | Unstructured off-road | 19 (mapped to 6) | RGB, LiDAR depth, surface normals |

Labels are harmonised across datasets into a 5-class terrain-centric taxonomy: road/flat, sidewalk, vegetation, rough/obstacle, sky/background.

---

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Optimiser | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| LR schedule | Cosine annealing with linear warm-up |
| Epochs | 40 |
| Batch size | 4 |
| Early stopping | Validation mIoU |
| Loss | Pixel-wise cross-entropy with class-frequency weighting |

Data augmentation applied synchronously across all modalities: random horizontal flip, brightness/contrast adjustment, random scaling and cropping, Gaussian depth noise (RELLIS-3D).

---

## Pretrained Models

| Model | Dataset | Input | File |
|-------|---------|-------|------|
| RGB baseline | Cityscapes | RGB | `city_rgb_best.pt` |
| RGB+Depth | Cityscapes | RGB + Depth | `city_rgbd_best.pt` |
| Trifecta | Cityscapes | RGB + Depth + Normals | `city_rgbd_normals_best.pt` |
| RGB baseline | KITTI | RGB | `kitti_rgb_mobilenet_2class_best.pt` |
| RGB+Depth | KITTI | RGB + Depth | `kitti_rgbd_mobilenet_2class_best.pt` |
| Trifecta | KITTI | RGB + Depth + Normals + IMU | `kitti_trifecta_mobilenet_2class_best.pt` |
| RGB baseline | RELLIS-3D | RGB | `rellis_rgb_mobilenet_6class_best.pt` |
| RGB+Depth | RELLIS-3D | RGB + Depth | `rellis_rgbd_mobilenet_6class_best.pt` |
| Trifecta | RELLIS-3D | RGB + Depth + Normals | `rellis_trifecta_mobilenet_6class_best.pt` |

---

## Project Structure

```
Terrain-Detection-Segmentation/
|
|-- models/                          # Pretrained model weights (.pt)
|-- results/
|   |-- graphs/                      # Training loss and mIoU curves
|   |-- confusion_matrices/          # Per-dataset confusion matrices
|   |-- per_class_iou/               # Per-class IoU bar charts
|   |-- qualitative/                 # Segmentation output visualisations
|
|-- preprocessing/                   # Data preprocessing pipeline scripts
|-- training/                        # Model training scripts
|-- evaluation/                      # Evaluation and cross-dataset transfer scripts
|
|-- report/
|   |-- FinalReport_TerrainDetection.pdf
|   |-- poster.pdf
|
|-- README.md
|-- requirements.txt
```

---

## Repo Name

```
Terrain-Detection-Segmentation-Autonomous-Navigation
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch, DeepLabV3+ |
| Backbone | MobileNetV3-Large |
| Fusion | Custom Trifecta Fusion Module (SE-attention-style adaptive gating) |
| Datasets | Cityscapes, KITTI Road, RELLIS-3D |
| Depth Estimation | MiDaS (for sensor-free inference) |
| Evaluation | mIoU, pixel accuracy, per-class IoU, confusion matrix |

---

## Key Findings

1. RGB appearance cues are sufficient in structured domains but collapse under domain shift — RGB-only degrades from 0.934 in-dataset to 0.395 on Cityscapes → RELLIS-3D transfer.
2. Geometric cues provide domain-invariant representations only when fused coherently — raw depth alone (RGB+D = 0.288) underperforms RGB-only under the same transfer, due to sensor-specific noise.
3. Dynamic adaptive gating is architecturally necessary — without it, multi-modal fusion can degrade performance relative to the RGB-only baseline.

---

## Future Work

- Formal FPS and GFLOPs benchmarking on embedded hardware
- Explicit uncertainty estimation for per-modality predictions
- Grad-CAM analysis to quantify per-modality spatial contribution
- Temporal sequence fusion over multi-frame video for improved stability
- Validation of MiDaS-based sensor-agnostic inference in depth-sensor-free deployment

---

## Team

| Name | Role | Contact |
|------|------|---------|
| Sana Firdous | Lead researcher | f20220193@dubai.bits-pilani.ac.in |
| Dr. Raja Muthalagu | Supervisor | raja.m@dubai.bits-pilani.ac.in |
| Prof. Pranav M. Pawar | Co-supervisor | pranav@dubai.bits-pilani.ac.in |
| Dr. Tojo Mathew | Co-supervisor | tojomathew@dubai.bits-pilani.ac.in |

BITS Pilani Dubai Campus, Department of Computer Science and Engineering

---

## Tech Used
`Python` · `PyTorch` · `DeepLabV3+` · `MobileNetV3` · `MiDaS` · `KITTI` · `Cityscapes` · `RELLIS-3D` · `LiDAR` · `IMU` · `Semantic Segmentation` · `Multi-Modal Fusion`
