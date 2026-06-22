# FPN-Mamba: Multi-Scale Medulloblastoma Classification in Brain MRI

This repository contains the implementation of **FPN-Mamba**, an architecture that integrates a **Feature Pyramid Network (FPN)** with **Bidirectional Mamba State Space Models** for binary classification of medulloblastoma (MB) in brain MRI. The work is motivated by and compared against InceptentionNet:

> Fang C, Li C, Liu H, et al. Precise identification of medulloblastoma in MRI images using a convolutional neural network integrated with a self-attention mechanism. *Digital Health*. 2025;11:20552076251351536.

---

## 1. Project Overview

Medulloblastoma is the most common malignant brain tumor in children, accounting for approximately 20% of all pediatric brain tumors. Early and accurate identification is critical because a missed diagnosis delays treatment during the period of greatest therapeutic opportunity.

InceptentionNet achieves strong binary classification performance but applies self-attention to a **single, downsampled feature map**. This means the model reasons at one coarse spatial scale, which limits its sensitivity to small lesions, weakly contrasted tumors, and atypically located masses — precisely the cases identified in Fang et al.'s own error analysis as the dominant source of false negatives.

FPN-Mamba addresses this through two simultaneous design changes:

1. **Multi-scale feature hierarchy via FPN** — every pyramid level from fine (56×56) to coarse (7×7) reaches the classifier, so no presentation is structurally excluded from detection.
2. **Linear-complexity global context via Mamba** — Bidirectional Mamba State Space Models replace self-attention at each pyramid level, operating in O(L) time rather than O(L²). This makes global context modeling feasible at fine spatial resolutions where self-attention would be computationally prohibitive.

The improved metrics are a downstream consequence of this architectural design. The primary objective is clinical: reducing the false negative rate for the hardest MB cases.

---

## 2. Architecture

FPN-Mamba consists of four functional stages:

### 2.1 EfficientNet-B2 Backbone
- ImageNet pretrained, accessed via `timm` in features-only mode
- Outputs four feature levels: C2 (56×56, 24ch), C3 (28×28, 48ch), C4 (14×14, 120ch), C5 (7×7, 352ch)
- First three child modules frozen; deeper modules fine-tuned
- Backbone channels probed dynamically at initialisation to avoid hardcoded dimension bugs

### 2.2 FPN Neck with LocalityMixing
- Lateral 1×1 convolutions project every level to 256 channels
- Top-down pathway adds upsampled deeper features to shallower lateral projections
- The standard 3×3 FPN smoothing convolution is **replaced at every level** by a **LocalityMixing block**:
  - Depthwise 3×3 conv for local neighbourhood refinement
  - Bidirectional Mamba SSM for full-image global context (O(L) complexity)
  - Learned sigmoid gate blends local and global per spatial position
  - Serialises BCHW → B(HW)C for Mamba, then deserialises back to BCHW

### 2.3 Cross-Scale Bidirectional Mamba
- All four pyramid levels (P2–P5) are flattened and concatenated into one sequence
- A single Bidirectional Mamba scan propagates context across levels simultaneously
- Output is split by known token counts and deserialized to per-level spatial maps
- Enables fine-scale evidence (P2) to directly inform coarse-scale semantics (P5) and vice versa

### 2.4 Classification Head
- **GeM Pooling** (learned p=3): pools each pyramid level independently, emphasising high-activation positions over background tissue
- **SE Channel Attention** (1024→64→1024): recalibrates which of the 1024 concatenated channels are most informative for MB classification
- **FC Head**: Linear(1024→512) BN GELU Dropout(0.3) → Linear(512→128) GELU Dropout(0.15) → Linear(128→1) → Sigmoid

---

## 3. Data

### 3.1 Training and Internal Validation Data

**Source:** Kaggle – *Brain Tumor for 14 classes*  
**Link:** https://www.kaggle.com/datasets/waseemnagahhenes/brain-tumor-for-14-classes

We follow the data selection described by Fang et al. (2025):

- 131 original medulloblastoma (MB) images in the Kaggle dataset.
- 25 duplicate MB images removed (pHash + manual review).
- Final MB images: **106**.
- 636 non-MB images randomly selected from the remaining 13 classes.
- 6 duplicate non-MB images removed.
- Final non-MB images: **630**.
- Total: **736** images.
- After augmentation (rotation, flipping, zooming): **2944** images.

**Important:** We only use the **Kaggle dataset**, which is public.  
The **clinical external validation dataset** from Shanghai Children’s Medical Center used in the original paper is **not publicly available**, so it is **not included** in this repository.

### 3.2 Labels

- Positive class: **Medulloblastoma (MB)** images.
- Negative class: **Non-MB** images (other tumor types and normal).

Labels are derived from the directory structure / class names provided in the Kaggle dataset and then consolidated into a binary label.

---

## 4. Repository Structure (Planned)

> Note: This is an initial layout and may evolve as code is added.

```text
.
├── data/
│   └── README.md           # Instructions on downloading and organizing Kaggle data
├── notebooks/
│   └── exploration.ipynb   # Data exploration, sanity checks, sample visualizations
├── src/
│   ├── datasets.py         # Dataset and dataloader utilities
│   ├── transforms.py       # Preprocessing and augmentation pipelines
│   ├── models/
│   │   ├── inceptentionnet.py      # Baseline InceptentionNet implementation
│   │   ├── fpn_inceptentionnet.py  # Proposed FPN-InceptentionNet
│   │   └── attention.py            # Self-attention modules
│   ├── train.py            # Training loop and cross-validation
│   ├── eval.py             # Evaluation and metrics
│   └── visualization.py    # Grad-CAM and attention heatmaps
├── experiments/
│   └── configs/            # YAML/JSON configs for experiments
├── Literatures/            # Papers and references
├── Writeups/               # Project writeups / notes
├── README.md               # Project overview and instructions
└── requirements.txt        # Python dependencies
```

---

## 5. Getting Started

### 5.1 Environment

- Python ≥ 3.9
- PyTorch (or preferred deep learning framework)
- CUDA-enabled GPU is strongly recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

### 5.2 Data Setup

1. Download the Kaggle dataset:

   - Go to:  
     https://www.kaggle.com/datasets/waseemnagahhenes/brain-tumor-for-14-classes
   - Download and extract it under `data/brain_tumor_14_classes/` (or follow the path specified in `data/README.md` once added).

2. Run a preparation script (to be added) to:
   - Filter MB vs. non-MB classes.
   - Remove duplicates (optional replication of original procedure).
   - Create train/validation splits for cross-validation.

### 5.3 Running Experiments

> Detailed commands will be added once the training scripts are in place.

Planned workflow:

```bash
# Train baseline InceptentionNet with 5-fold cross-validation
python src/train.py --config experiments/configs/inceptentionnet_baseline.yaml

# Train FPN-InceptentionNet with 5-fold cross-validation
python src/train.py --config experiments/configs/fpn_inceptentionnet.yaml

# Evaluate and generate metrics & ROC curves
python src/eval.py --run_id <run_id>
```

---

## 6. Status

- [x] Baseline InceptentionNet implementation
- [x] Training and evaluation scripts
- [x] Initial 5-fold cross-validation run on Kaggle data (see Section 6.1 for results)
- [ ] FPN-InceptentionNet implementation
- [ ] Grad-CAM / attention visualization tools
- [ ] Reproduction of baseline metrics on Kaggle data (see known issues in Section 6.1)
- [ ] Comparative experiments (baseline vs. FPN)

Updates will be pushed as the implementation and experiments progress.

---

## 6.1 Baseline Training Results

Results are from a 5-fold stratified cross-validation run logged in
`experiments/runs/inceptentionnet_baseline_notebook/cv_results.json`.

### Dataset

| Split | Count |
|-------|-------|
| Medulloblastoma (MB) | 106 |
| Non-MB (13 other classes) | 630 |
| **Total** | **736** |

After 4× augmentation the training set contains approximately **2 944** samples.
Training was performed on a CUDA GPU with `batch_size=2`, `image_size=128`,
`learning_rate=0.005`, and `early_stopping_patience=10`.

### 5-Fold Cross-Validation Summary

| Metric | Mean | Std | Paper Target (Fang et al. 2025) | Gap |
|--------|------|-----|--------------------------------|-----|
| Accuracy | 0.856 | ±0.003 | 0.981 | −0.125 |
| Precision | 0.000 | ±0.000 | 0.914 | −0.914 |
| Recall | 0.000 | ±0.000 | 0.960 | −0.960 |
| F1 Score | 0.000 | ±0.000 | 0.935 | −0.935 |
| AUC | 0.500 | ±0.000 | 0.994 | −0.494 |

### Per-Fold Results

| Fold | Best Epoch | Best Val Loss | Accuracy | Precision | Recall | F1 | AUC |
|------|-----------|--------------|----------|-----------|--------|----|-----|
| 1 | 7 | 0.4204 | 0.851 | 0.000 | 0.000 | 0.000 | 0.500 |
| 2 | 5 | 0.4204 | 0.857 | 0.000 | 0.000 | 0.000 | 0.500 |
| 3 | 15 | 0.4084 | 0.857 | 0.000 | 0.000 | 0.000 | 0.500 |
| 4 | 13 | 0.4084 | 0.857 | 0.000 | 0.000 | 0.000 | 0.500 |
| 5 | 11 | 0.4084 | 0.857 | 0.000 | 0.000 | 0.000 | 0.500 |

### Analysis

The model converges quickly (best epoch between 5 and 15) but **collapses to
predicting every sample as non-MB** across all five folds.  The ~85.6% accuracy
is simply the majority-class rate; precision, recall, F1, and AUC all indicate
no ability to distinguish MB from non-MB.

**Root cause: unweighted binary cross-entropy on a severely imbalanced dataset.**
The class ratio is roughly 1 : 5.9 (MB : non-MB).  Without a positive-class
weight in `BCEWithLogitsLoss`, the loss is minimised by always predicting 0,
which suffices to drive the loss below 0.41 while leaving recall at zero.

**Fix applied in `src/train.py`:** `pos_weight` is now computed from the
training fold class distribution and passed to `BCEWithLogitsLoss`.  This
rescales the gradient contribution of the minority class and is the standard
PyTorch mechanism for handling imbalanced binary classification.

---

## 7. Citation

If you use this repository or build upon its ideas, please cite the original baseline paper and the FPN paper:

```text
Fang C, Li C, Liu H, et al. Precise identification of medulloblastoma in MRI images using a convolutional neural network integrated with a self-attention mechanism. Digital Health. 2025;11:20552076251351536.

Lin T-Y, Dollár P, Girshick R, He K, Hariharan B, Belongie S. Feature Pyramid Networks for Object Detection. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017:2117–2125.
```

You may also cite this repository once you decide on its final name and URL.

---

## 8. License

> To be decided.  
> A permissive license such as **MIT** or **Apache-2.0** is recommended if you intend others to reuse and extend this work.

---

## 9. Acknowledgements

- Fang et al. (2025) for the original InceptentionNet architecture and experimental setup.
- The creators of the Kaggle “Brain Tumor for 14 classes” dataset.
- Lin et al. (2017) for the Feature Pyramid Network design that motivates the proposed extension.

## 10. Authors

- Simbiat Adetoro - sadetoro@andrew.cmu.edu
- Samuel Adeniji - ifeoluwasamuel40@gmail.com
