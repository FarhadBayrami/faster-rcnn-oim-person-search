<div align="center">

# 🔍 Person Search with Faster R-CNN + OIM Loss
### End-to-End Pedestrian Detection and Re-Identification on the PRW Dataset

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://img.shields.io/badge/Dataset-PRW-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Backbone-ResNet--50%20FPN-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Loss-OIM-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Task-Person%20Search-red?style=flat-square"/>
</p>

*A two-stage person search pipeline combining Faster R-CNN pedestrian detection with OIM-based re-identification, evaluated on the PRW benchmark.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Method](#-method)
- [Dataset](#-dataset)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [References](#-references)
- [Author](#-author)

---

## 🔬 Overview

**Person Search** is the task of simultaneously detecting and re-identifying a query person across a gallery of raw, uncropped scene images — a more realistic and challenging setting than standard Re-ID where bounding boxes are pre-given.

This project implements a full person search pipeline on the **PRW (Person Re-identification in the Wild)** dataset using:
- **Faster R-CNN** (ResNet-50 FPN) for pedestrian detection
- **OIM (Online Instance Matching) loss** for discriminative Re-ID embedding learning
- **Cosine similarity search** to match query against gallery detections

---

## ⚙️ Method

| Stage | Component | Description |
|-------|-----------|-------------|
| **1. Detection** | Faster R-CNN (ResNet-50 FPN) | Fine-tuned on PRW to produce pedestrian bounding boxes |
| **2. Re-ID** | ResNet-50 + OIM Loss | Maps each detected crop to a discriminative feature vector |
| **3. Search** | Cosine Similarity | Query embedding compared against all gallery detections; top-k matches returned |

The architecture is inspired by the OIM loss formulation from Xiao et al. (CVPR 2017), adapted for the PRW benchmark.

---

## 📦 Dataset

**PRW (Person Re-identification in the Wild)** — Zheng et al., CVPR 2017

🔗 [github.com/liangzheng06/PRW-baseline](https://github.com/liangzheng06/PRW-baseline)

| Split | Frames | IDs | Pedestrians | Pedestrians w/ ID |
|-------|--------|-----|-------------|-------------------|
| Train | 5,134  | 482 | 16,243      | 13,416            |
| Val   | 570    | 482 | 1,805       | 1,491             |
| Test  | 6,112  | 450 | 25,062      | 19,127            |

After downloading, place the dataset as:
data/
└── PRW/
├── frames/
├── annotations/
├── query_info.txt
└── frame_test.mat / frame_train.mat / ...
---

## 📊 Results

Evaluated on PRW test set:

| Model | mAP | Top-1 | Top-5 |
|-------|-----|-------|-------|
| Detector only (baseline) | — | — | — |
| + OIM Re-ID head | — | — | — |
| + Joint fine-tuning | — | — | — |

*(Results to be filled in after running experiments)*

---

## 🚀 Getting Started

### Prerequisites

```bash
conda create -n personsearch python=3.10 -y
conda activate personsearch
pip install -r requirements.txt
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/FarhadBayrami/faster-rcnn-oim-person-search.git
cd faster-rcnn-oim-person-search

# 2. Create environment and install dependencies
conda create -n personsearch python=3.10 -y
conda activate personsearch
pip install -r requirements.txt

# 3. Download PRW dataset
#    → https://github.com/liangzheng06/PRW-baseline
#    → Place in data/PRW/ as described above
```

---

## 🏋️ Training

```bash
# 1. Train the detector
python scripts/train_detector.py --config configs/detector.yaml

# 2. Train the Re-ID head with OIM loss
python scripts/train_reid.py --config configs/reid.yaml

# 3. Optional: joint end-to-end fine-tuning
python scripts/train_joint.py --config configs/joint.yaml
```

---

## 📐 Evaluation

```bash
python scripts/evaluate.py --config configs/joint.yaml --checkpoint results/best_model.pth
```

Reports **mAP** and **top-k recall** on the PRW test set.

---

## 📁 Project Structure

| Path | Description |
|------|-------------|
| `configs/` | YAML configuration files for detector, Re-ID, and joint training |
| `models/` | Detector, Re-ID head, and full pipeline model definitions |
| `scripts/` | Training and evaluation entry points |
| `utils/` | Metrics, visualisation, and helper functions |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT License |
| `CITATION.cff` | How to cite this work |
| `README.md` | Project documentation |

---

## 🔮 Future Work

- [ ] Report full mAP and top-k results after training
- [ ] Experiment with SeqNet or COAT for end-to-end joint training
- [ ] Add query visualisation (show top-5 matches in gallery)
- [ ] Extend to CUHK-SYSU dataset
- [ ] Integrate transformer-based Re-ID backbone (e.g. TransReID)

---

## 📚 References

1. Zheng, L. et al. — *Person Re-identification in the Wild (PRW)*, CVPR 2017.
2. Xiao, T. et al. — *Joint Detection and Identification Feature Learning for Person Search (OIM)*, CVPR 2017.
3. [torchvision Faster R-CNN](https://pytorch.org/vision/stable/models/faster_rcnn.html)
4. [torchreid — Deep Person Re-ID](https://github.com/KaiyangZhou/deep-person-reid)

---

## 👤 Author

**Farhad Bayrami**
MSc Student — University of Bologna
📧 [farhad.bayrami@studio.unibo.it](mailto:farhad.bayrami@studio.unibo.it)
🔗 [GitHub](https://github.com/FarhadBayrami)

---

<div align="center">
  <sub>Built with ❤️ as part of a Machine Learning for Computer Vision course project at the University of Bologna · 2025–2026</sub>
</div>
