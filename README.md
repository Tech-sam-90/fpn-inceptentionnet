# FPN-InceptentionNet: Multi-Scale Medulloblastoma Classification in Brain MRI

This repository contains an extended implementation of **InceptentionNet**, a hybrid CNN + self-attention model for medulloblastoma (MB) classification in MRI, with an additional **Feature Pyramid Network (FPN)** module to better handle lesion size and appearance variability.

The work is inspired by and uses the architecture in:

> Fang C, Li C, Liu H, et al. Precise identification of medulloblastoma in MRI images using a convolutional neural network integrated with a self-attention mechanism. *Digital Health*. 2025;11:20552076251351536.

---

## 1. Project Overview

Medulloblastoma is the most common malignant brain tumor in children and often arises in the posterior fossa. In Fang et al.’s InceptentionNet, an Inception-based backbone is combined with a multihead self-attention module to perform **binary image classification** (MB vs. non-MB) using MRI.

While InceptentionNet achieves strong performance, its self-attention operates on a **single, downsampled feature map**, inheriting the usual CNN limitation that deep features are semantically strong but spatially coarse. This can reduce sensitivity to:

- Very small medulloblastoma lesions,
- Weakly contrasted or atypical tumors,
- Tumors that overlap anatomically and visually with other posterior fossa entities (e.g., cystic ependymoma, midline glioma).

Feature Pyramid Networks (FPN) were proposed to address exactly this kind of multi-scale issue by constructing a **top–down feature hierarchy with lateral connections**, yielding feature maps that are both **semantically rich and spatially detailed** at multiple scales (Lin et al., 2017).

### Goal of this Repository

This repository:

- Reproduces the **baseline InceptentionNet** architecture as described by Fang et al. (2025).
- Introduces an **FPN module** on top of the Inception backbone to build a **multi-scale feature pyramid** before self-attention.
- Compares **baseline vs. FPN-InceptentionNet** on the same publicly available dataset used for training in the original paper (Kaggle “Brain Tumor for 14 classes”).
- Evaluates whether FPN improves robustness to lesion size and appearance variation in MB classification.

---

## 2. Methods (High-Level)

### 2.1 Baseline: InceptentionNet

The baseline follows Fang et al. (2025):

1. **Preprocessing**
   - Gaussian filtering (σ = 2) for noise reduction.
   - Histogram equalization for contrast enhancement.
   - Data augmentation: random rotation (0°–45°), horizontal flipping, zooming (0.8–1.2).

2. **Architecture**
   - **Stem**: 3×3 convolution (stride 1, 64 filters).
   - **Modified Inception block**: parallel 1×1, 3×3, and 5×5 convolutions with batch normalization and ReLU, stride-2 convolution instead of max-pooling for downsampling.
   - **Self-attention module**: multihead self-attention (e.g., 4 heads, 64-dim queries/keys) applied on the extracted feature map.
   - **Classification head**: dense layers with dropout, sigmoid output for MB vs. non-MB.

3. **Training**
   - Optimizer: Adam (initial learning rate 0.005, reduce-on-plateau).
   - Batch size: 8.
   - Early stopping: patience 10.
   - 5-fold cross-validation.

### 2.2 Proposed: FPN-InceptentionNet

The proposed model inserts an **FPN** between the Inception backbone and the self-attention:

1. **Bottom-up pathway**
   - Inception blocks produce feature maps at multiple depths (e.g., C3, C4, C5).

2. **Top–down FPN with lateral connections**
   - C5 is upsampled and merged with C4 via a 1×1 lateral convolution → P4.
   - P4 is upsampled and merged with C3 → P3.
   - Each merge is followed by a 3×3 convolution to reduce aliasing.

3. **Scale-aware self-attention**
   - Self-attention is applied at each pyramid level (P3, P4, P5) separately.
   - Attention outputs are pooled and concatenated before classification.

4. **Classification**
   - Global average pooling per scale.
   - Concatenation → dense layer + batch norm + ReLU → dropout → sigmoid.

The task remains **binary classification**, so only **image-level labels** are required.

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

This repository is in an **early stage**:

- [ ] Baseline InceptentionNet implementation
- [ ] FPN-InceptentionNet implementation
- [ ] Training and evaluation scripts
- [ ] Grad-CAM / attention visualization tools
- [ ] Reproduction of baseline metrics on Kaggle data
- [ ] Comparative experiments (baseline vs. FPN)

Updates will be pushed as the implementation and experiments progress.

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
