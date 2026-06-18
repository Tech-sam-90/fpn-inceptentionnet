# FPN-Mamba: Feature Pyramid Network with Bidirectional Mamba for Medical MRI Classification

A PyTorch implementation of FPN-Mamba applied to binary medulloblastoma (MB) screening on brain MRI, with a full 5-fold ablation study and statistical comparison against a re-implemented InceptentionNet baseline.

**Key results (5-fold CV, 761 images, A100 80GB):**

| Model | AUC | F1 | Sensitivity | Specificity | Params | GFLOPs | FPS |
|---|---|---|---|---|---|---|---|
| InceptentionNet (re-impl.) | 82.97 ± 9.46% | 61.46 ± 2.96% | 76.58 ± 12.34% | 87.62 ± 5.98% | 1.1M | 19.3 | 224 |
| **FPN-Mamba (ours)** | **99.35 ± 0.68%** | **94.54 ± 5.49%** | **96.95 ± 3.21%** | **98.25 ± 1.81%** | 17.8M | 33.1 | 112 |

Bootstrap 95% CIs are non-overlapping (AUC: \[75.1, 89.2\]% vs \[98.8, 99.8\]%). Wilcoxon p = 0.0625 — the minimum achievable for n = 5 paired folds with all differences in one direction.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Dataset](#dataset)
3. [Folder Structure](#folder-structure)
4. [Setup](#setup)
5. [Running on Colab (recommended)](#running-on-colab-recommended)
6. [Running Locally](#running-locally)
7. [Configuration](#configuration)
8. [Ablation Study](#ablation-study)
9. [Resume After Disconnect](#resume-after-disconnect)
10. [Results Files](#results-files)

---

## Architecture

FPN-Mamba has four stages:

```
Input (3×224×224)
     │
     ▼
EfficientNet-B2 backbone (pretrained, first 3 stages frozen during training)
     │  C2: 24ch @ 56×56
     │  C3: 48ch @ 28×28
     │  C4: 120ch @ 14×14
     │  C5: 352ch @ 7×7
     ▼
FPN neck (top-down pathway, lateral 1×1 projections → 256ch at each level)
  Each level: LocalityMixing block
              DWConv3×3 ──► BidirMamba ──► learned gate ──► fused features
     │  P2 @ 56×56
     │  P3 @ 28×28
     │  P4 @ 14×14
     │  P5 @ 7×7
     ▼
Cross-scale Mamba (all 4 pyramid levels concatenated and scanned jointly)
     ▼
GeM Pooling (learnable p, init=3) per level → concat → 1024-dim vector
     ▼
SE channel attention (reduction=16)
     ▼
FC head: Linear(1024→512)–BN–GELU–Dropout(0.3)–Linear(512→128)–GELU–Dropout(0.15)–Linear(128→1)–Sigmoid
     ▼
  P(MB) ∈ (0,1)
```

All Mamba blocks are pure-PyTorch SelectiveSSMs — no proprietary CUDA extensions required.

---

## Dataset

**Binary MB task:** 761 images (131 MB + 630 non-MB) from the public Brain Tumor MRI dataset ([Kaggle: waseemnagahhenes](https://www.kaggle.com/datasets/waseemnagahhenes/brain-tumor-mri-dataset-14-classes)). All 14 classes are in `data/`; the training pipeline samples MB and non-MB counts via config.

**3-class benchmark:** Meningioma, glioma, pituitary from the Cheng et al. dataset (Abiwinanda et al. split), not included in this repo.

The dataset is committed directly in `data/` — no separate download is needed when cloning.

---

## Folder Structure

```
fpn-inceptentionnet/
│
├── data/                          # Raw MRI images — 14 class subdirectories
│   ├── Meduloblastoma/            # 131 MB images (positive class)
│   ├── Astrocitoma/
│   ├── Carcinoma/
│   └── ...                        # 12 other tumor classes (non-MB pool)
│
├── configs/
│   ├── fpn_mamba.yaml             # FPN-Mamba training config
│   ├── inceptentionnet.yaml       # InceptentionNet baseline config (paper-exact)
│   └── ablation.yaml              # Shared config for all 5 ablation variants
│
├── src/
│   ├── models/
│   │   ├── fpn_mamba.py           # FPNMambaClassifier — full model + ablation flags
│   │   ├── inceptentionnet.py     # InceptentionNet — CNN stem + Inception + self-attention
│   │   └── ablation.py            # ABLATION_VARIANTS dict + build_ablation_model()
│   │
│   ├── data/
│   │   ├── dataset.py             # BrainTumorDataset, build_binary_samples(), WeightedSampler
│   │   └── transforms.py          # Train / eval / MB-specific augmentation pipelines
│   │
│   ├── training/
│   │   ├── trainer.py             # run_crossval() — 5-fold CV with fold-level resume
│   │   └── losses.py              # Weighted BCE, label smoothing, pos_weight
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # compute_metrics(), summarize_folds(), Youden threshold
│   │   ├── stats.py               # compare_models() — bootstrap CIs + Wilcoxon
│   │   └── gradcam.py             # Grad-CAM utility (for future visualisation work)
│   │
│   └── utils/
│       ├── flops.py               # count_parameters(), compute_flops() via thop
│       └── seed.py                # seed_everything() for reproducibility
│
├── scripts/
│   ├── train.py                   # Entry point: trains one model (FPN-Mamba or InceptentionNet)
│   └── run_ablation.py            # Trains all 5 ablation variants sequentially
│
├── notebooks/
│   └── colab_runner.ipynb         # End-to-end Colab notebook (A100 recommended)
│
├── experiments/
│   └── runs/                      # Default local output directory (gitignored)
│       ├── fpn_mamba/
│       │   ├── fold_1.pt          # Best checkpoint per fold (by validation F1)
│       │   ├── fold_1_result.json # Per-fold metrics (enables fold-level resume)
│       │   ├── ...
│       │   └── cv_results.json    # Full 5-fold summary
│       ├── inceptentionnet/
│       └── ablation/
│           ├── efficientnet_only/
│           ├── fpn_standard/
│           ├── fpn_locality/
│           ├── fpn_cross_mamba/
│           ├── fpn_mamba_full/
│           └── ablation_summary.json
│
├── overleaf_paper/
│   └── summarised_emerge.tex      # Conference paper (LLNCS format, 10 pages)
│
├── Literatures/                   # Reference PDFs
├── Writeups/                      # Project proposals and reports
└── README.md
```

---

## Setup

```bash
pip install torch torchvision timm einops scikit-learn scipy thop pyyaml tqdm matplotlib seaborn pandas wandb
```

Python 3.9+ required. No CUDA extensions — the Mamba blocks are pure PyTorch and run on any GPU.

---

## Running on Colab (recommended)

Open [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) in Google Colab. Switch the runtime to **A100 GPU** before running.

Run the cells in order:

| Cell | What it does | Time |
|---|---|---|
| 1 | Verify A100 GPU and bfloat16 support | instant |
| 2 | Mount Google Drive (checkpoints save here) | instant |
| 3 | Clone this repo and verify the dataset | ~1 min |
| 4 | Install dependencies | ~2 min |
| 5 | Log in to Weights & Biases | instant |
| 6 | Train InceptentionNet baseline | ~45 min |
| 7 | Train FPN-Mamba full model | ~90 min |
| 8 | Run 5-variant ablation study | ~2 hrs |
| 9 | Statistical comparison (bootstrap CI + Wilcoxon) | instant |
| 10 | Print ablation table | instant |
| 11 | Verify all checkpoints are saved to Drive | instant |
| 12 | Profile FLOPs and parameter counts | ~30 sec |

Cells 6–8 **skip automatically** if results already exist on Drive — safe to re-run after any disconnect.

### Weights & Biases
Replace `paste_your_wandb_key_here` in Cell 5 with your key from [wandb.ai](https://wandb.ai). To skip wandb entirely, set `wandb.enabled: false` in the YAML configs.

---

## Running Locally

### Train FPN-Mamba
```bash
python scripts/train.py --config configs/fpn_mamba.yaml
```

### Train InceptentionNet baseline
```bash
python scripts/train.py --config configs/inceptentionnet.yaml
```

### Run the full ablation
```bash
python scripts/run_ablation.py --config configs/ablation.yaml

# Run a subset of variants:
python scripts/run_ablation.py --config configs/ablation.yaml \
    --variants fpn_standard fpn_locality
```

### Statistical comparison (after both models are trained)
```python
import json
from src.evaluation.stats import compare_models, print_comparison_table

with open('experiments/runs/inceptentionnet/cv_results.json') as f:
    baseline = json.load(f)
with open('experiments/runs/fpn_mamba/cv_results.json') as f:
    fpn = json.load(f)

table = compare_models(baseline['fold_results'], fpn['fold_results'],
                       name_a='InceptentionNet', name_b='FPN-Mamba')
print_comparison_table(table, name_a='InceptentionNet', name_b='FPN-Mamba')
```

### FLOPs and parameter counts
```python
from src.models.fpn_mamba import FPNMambaClassifier
from src.models.inceptentionnet import InceptentionNet
from src.utils.flops import model_summary

print("=== InceptentionNet ===")
model_summary(InceptentionNet())

print("\n=== FPN-Mamba ===")
model_summary(FPNMambaClassifier(freeze_layers=0))  # freeze_layers=0 to count all params
```

---

## Configuration

All experiments are driven by YAML configs in `configs/`. Any key can be overridden at runtime using the `make_config()` helper in the notebook (deep-merges a patch dict into the base YAML).

### Key FPN-Mamba settings (`configs/fpn_mamba.yaml`)

| Key | Default | Notes |
|---|---|---|
| `model.fpn_channels` | 256 | Feature channels at each FPN level |
| `model.d_state` | 16 | Mamba SSM state dimension |
| `model.freeze_layers` | 3 | EfficientNet-B2 stages to freeze |
| `training.batch_size` | 16 | Use 64 on A100 80GB |
| `training.learning_rate` | 1e-4 | 10-epoch linear warmup then cosine annealing |
| `training.max_epochs` | 60 | Early stopping: patience=15 on val AUC |
| `training.grad_clip` | 0.3 | Gradient norm clipping (stabilises SSM layers) |
| `training.num_workers` | 0 | Use 4 on A100 (12+ CPU cores available) |

### Key InceptentionNet settings (`configs/inceptentionnet.yaml`)

| Key | Default | Notes |
|---|---|---|
| `model.stem_channels` | 64 | Paper-exact |
| `model.branch_channels` | 64 | Paper-exact (4 branches → 256-dim embed) |
| `training.batch_size` | 8 | Paper-exact — do not change |
| `training.learning_rate` | 1e-3 | Reduced from paper's 5e-3 for stability on this dataset |
| `training.max_epochs` | 40 | Paper-exact |
| `data.gaussian_sigma` | 2.0 | Paper-exact preprocessing |
| `data.mb_target_count` | 106 | Paper: 106 after pHash + manual deduplication |
| `training.num_workers` | 0 | Must stay 0 — 736 samples + batch=8 deadlocks with workers |

---

## Ablation Study

Five variants are trained sequentially, each adding one component:

| Variant | Description | AUC (%) |
|---|---|---|
| `efficientnet_only` | EfficientNet-B2 backbone + GAP + FC head | 94.81 ± 2.66 |
| `fpn_standard` | + FPN with standard 3×3 convolution at each level | 99.12 ± 0.58 |
| `fpn_locality` | + LocalityMixing (DWConv + BidirMamba + learned gate) | 99.44 ± 0.41 |
| `fpn_cross_mamba` | + Cross-scale Mamba (joint scan across all pyramid levels) | 99.13 ± 0.56 |
| `fpn_mamba_full` | + GeM pooling + SE channel attention | 99.35 ± 0.68 |

FPN is the dominant single contributor (+4.31 pp AUC over backbone alone). LocalityMixing maximises sensitivity (99.23%) at the cost of some specificity; GeM + SE restores the balance.

Each variant is controlled by flags directly in `FPNMambaClassifier`: `use_fpn`, `use_locality_mixing`, `use_cross_mamba`, `use_gem`, `use_se`. The `ABLATION_VARIANTS` dict in [`src/models/ablation.py`](src/models/ablation.py) maps variant names to flag combinations.

---

## Resume After Disconnect

Two-level resume is built in — a Colab disconnect loses at most the currently-running fold:

**Variant level** (`run_ablation.py`): if `<run_dir>/<variant>/cv_results.json` exists, that variant is skipped entirely and its fold results are replayed to the console.

**Fold level** (`trainer.py`): after each fold completes, `fold_N_result.json` is written to the run directory. On reconnect, completed folds are loaded from disk and skipped — training resumes from the next fold.

Just re-run the cell — it picks up exactly where it left off.

---

## Results Files

After training, each run directory contains:

```
<run_dir>/
├── fold_1.pt              # Best model checkpoint for fold 1 (by val F1)
├── fold_1_result.json     # Full metrics for fold 1 (used for resume)
├── fold_2.pt
├── fold_2_result.json
├── fold_3.pt  /  fold_3_result.json
├── fold_4.pt  /  fold_4_result.json
├── fold_5.pt  /  fold_5_result.json
└── cv_results.json        # 5-fold summary: all fold results + dataset stats
```

The `ablation_summary.json` in the ablation run directory aggregates mean ± std for all metrics across all five variants.

---

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{adeniji2025fpnmamba,
  title   = {FPN-Mamba: Feature Pyramid Network with Bidirectional Mamba
             for Medical MRI Classification},
  author  = {Adeniji, Samuel Akinwumi and others},
  year    = {2025}
}
```
