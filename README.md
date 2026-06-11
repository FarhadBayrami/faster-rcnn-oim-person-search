# Person Search on PRW Dataset
**Machine Learning for Computer Vision — University of Bologna, A.Y. 2025–2026**

Person Search combines **pedestrian detection** and **person re-identification** into a single end-to-end task. Given a query image, the model must locate and match the same person across a gallery of raw, uncropped scene images.

---

## Method Overview

This repo implements a two-stage Person Search pipeline:

1. **Detection** — Faster R-CNN (ResNet-50 FPN backbone) fine-tuned on PRW to produce pedestrian bounding boxes.
2. **Re-ID** — An embedding network (ResNet-50 + OIM loss) that maps each detected crop to a discriminative feature vector.
3. **Search** — The query embedding is compared against all gallery detections via cosine similarity; the top-k matches are returned.

The architecture is inspired by the OIM (Online Instance Matching) loss formulation from [2], adapted for the PRW benchmark.

---

## Dataset: PRW

| Split | #Frames | #IDs | #Peds | #Peds w/ ID |
|-------|---------|------|-------|-------------|
| Train | 5134 | 482 | 16243 | 13416 |
| Val   | 570  | 482 | 1805  | 1491  |
| Test  | 6112 | 450 | 25062 | 19127 |

Download from the [official PRW page](https://github.com/liangzheng06/PRW-baseline) and place it as:

```
data/
└── PRW/
    ├── frames/
    ├── annotations/
    ├── query_info.txt
    └── frame_test.mat / frame_train.mat / ...
```

---

## Setup

```bash
git clone https://github.com/<your-username>/person-search-prw.git
cd person-search-prw
conda create -n personsearch python=3.10 -y
conda activate personsearch
pip install -r requirements.txt
```

---

## Training

```bash
# 1. Train the detector
python scripts/train_detector.py --config configs/detector.yaml

# 2. Train the Re-ID head (with OIM loss)
python scripts/train_reid.py --config configs/reid.yaml

# 3. (Optional) Joint end-to-end fine-tuning
python scripts/train_joint.py --config configs/joint.yaml
```

---

## Evaluation

```bash
python scripts/evaluate.py --config configs/joint.yaml --checkpoint results/best_model.pth
```

Reports **mAP** and **top-k recall** on the PRW test set.

---

## Results

| Model | mAP | Top-1 | Top-5 |
|-------|-----|-------|-------|
| Detector only (baseline) | — | — | — |
| + OIM Re-ID head | — | — | — |
| + Joint fine-tuning | — | — | — |

*(Fill in after running experiments)*

---

## Repository Structure

```
person-search-prw/
├── configs/            # YAML config files
├── data/               # Dataset loaders and transforms
├── models/             # Detector, Re-ID head, full pipeline
├── utils/              # Metrics, visualisation, helpers
├── scripts/            # train / eval entry-points
├── notebooks/          # Exploratory analysis
├── results/            # Checkpoints and logs (gitignored)
└── requirements.txt
```

---

## References

1. Zheng, L. et al. *Person Re-identification in the Wild.* CVPR 2017.
2. Xiao, T. et al. *Joint Detection and Identification Feature Learning for Person Search.* CVPR 2017.
3. [torchvision Faster R-CNN](https://pytorch.org/vision/stable/models/faster_rcnn.html)
4. [torchreid](https://github.com/KaiyangZhou/deep-person-reid)
