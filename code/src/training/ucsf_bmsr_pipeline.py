"""
Shared training logic for the UCSF-BMSR full-slice pipeline -- extracted from
notebooks/ucsf_bmsr_fullslice_pipeline.ipynb so the multi-GPU per-fold script
(scripts/run_fpn_mamba_full_fold.py) and the notebook run identical code, not
a duplicated copy that can drift out of sync.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.models.ablation import build_ablation_model, ABLATION_VARIANTS
from code.src.models.inceptentionnet import InceptentionNet
from src.models.timm_classifier import TimmClassifier
from src.evaluation.metrics import compute_metrics, find_best_threshold, recall_by_group

IMG_SIZE = 256
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
THRESHOLD_MIN, THRESHOLD_MAX = 0.05, 0.95


class UCSFBMSRDataset(Dataset):
    """Per-slice min-max to [0,1], then ImageNet norm. modality_mode toggle:
      "adjacent_t1c" (default) -- 3 adjacent T1c slices as the 3 channels, full axial slice.
      "per_modality" -- 1 center slice each from T1c/subtraction/FLAIR (the second option
      considered for the 2.5D input design, deferred as a later ablation).
    Works for both full-slice and crop-based data -- just point crop_dir/manifest/img_size at
    whichever pipeline_output* directory you want (see scripts/run_crop_fold.py for crop-based)."""

    def __init__(self, df: pd.DataFrame, crop_dir: Path, img_size: int = IMG_SIZE, augment: bool = False,
                 modality_mode: str = "adjacent_t1c"):
        self.df = df.reset_index(drop=True)
        self.crop_dir = crop_dir
        self.img_size = img_size
        self.augment = augment
        self.modality_mode = modality_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with np.load(self.crop_dir / row["crop_file"]) as z:
            if self.modality_mode == "adjacent_t1c":
                arr = z["t1post"].astype(np.float32)  # (3, H, W): 3 adjacent T1c slices
            elif self.modality_mode == "per_modality":
                t1 = z["t1post"].astype(np.float32)
                center = t1.shape[0] // 2
                sub = z["subtraction"][center].astype(np.float32)
                flair = z["flair"][center].astype(np.float32)
                arr = np.stack([t1[center], sub, flair], axis=0)  # (3, H, W): T1c/sub/FLAIR, center slice each
            else:
                raise ValueError(f"Unknown modality_mode: {self.modality_mode}")
        t = torch.from_numpy(arr).unsqueeze(0)
        lo, hi = t.amin(dim=(2, 3), keepdim=True), t.amax(dim=(2, 3), keepdim=True)
        t = (t - lo) / (hi - lo + 1e-6)
        t = F.interpolate(t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                t = torch.flip(t, dims=[2])
            if torch.rand(1).item() < 0.5:
                t = torch.flip(t, dims=[1])
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return t, torch.tensor(float(row["label"]))


def build_model(variant: str):
    if variant in ABLATION_VARIANTS:
        return build_ablation_model(variant, dropout=0.3, fpn_channels=256, d_state=16, freeze_stages=2)
    if variant == "inceptentionnet":
        return InceptentionNet(stem_channels=64, branch_channels=64, num_heads=4, dropout=0.3)
    if variant == "resnet50":
        return TimmClassifier(model_name="resnet50", pretrained=True, dropout=0.3, freeze_layers=2, hidden_dim=256)
    if variant == "resnet18":
        # ~11.7M params -- size-matched to the FPN-Mamba variants (12-14M), unlike resnet50 (~24M).
        return TimmClassifier(model_name="resnet18", pretrained=True, dropout=0.3, freeze_layers=2, hidden_dim=256)
    raise ValueError(f"Unknown variant: {variant}")


@torch.no_grad()
def evaluate(model, loader, device, threshold=None):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).reshape(-1)  # (batch,1) -> (batch,) -- matches src/training/trainer.py's convention;
        probs.append(torch.sigmoid(logits).cpu().numpy())  # without this, BCEWithLogitsLoss silently broadcasts
        labels.append(yb.numpy())                          # (batch,1) vs (batch,) into a wrong (batch,batch) loss
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    thr = threshold if threshold is not None else find_best_threshold(labels, probs, THRESHOLD_MIN, THRESHOLD_MAX)
    metrics = compute_metrics(labels, probs, thr, THRESHOLD_MIN, THRESHOLD_MAX)
    return metrics, labels, probs


def train_one_fold(variant, manifest_folded, fold, crop_dir, device, checkpoint_dir,
                    max_epochs=30, batch_size=4, lr=1e-4, patience=8, calib_fraction=0.15, seed=42,
                    img_size=IMG_SIZE, modality_mode="adjacent_t1c"):
    train_pool = manifest_folded[manifest_folded.fold != fold]
    held_out_df = manifest_folded[manifest_folded.fold == fold]

    # inner calibration split, grouped by patient (never leak a patient across inner-train/calib)
    gss = GroupShuffleSplit(n_splits=1, test_size=calib_fraction, random_state=seed)
    inner_idx, calib_idx = next(gss.split(train_pool, groups=train_pool["patient_id"]))
    inner_train_df = train_pool.iloc[inner_idx]
    calib_df = train_pool.iloc[calib_idx]

    train_ds = UCSFBMSRDataset(inner_train_df, crop_dir, img_size=img_size, augment=True, modality_mode=modality_mode)
    calib_ds = UCSFBMSRDataset(calib_df, crop_dir, img_size=img_size, augment=False, modality_mode=modality_mode)
    held_out_ds = UCSFBMSRDataset(held_out_df, crop_dir, img_size=img_size, augment=False, modality_mode=modality_mode)

    labels_arr = inner_train_df["label"].values.astype(int)
    counts = np.bincount(labels_arr)
    weights = 1.0 / counts[labels_arr]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    calib_loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    held_out_loader = DataLoader(held_out_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Warm restarts every 5 epochs: fpn_mamba_full's much larger trainable parameter count
    # (FPN + LocalityMixing/SelectiveSSM + cross-Mamba + head, all unfrozen) was overshooting
    # past its calib-AUC peak under a fixed lr (avg 0.043 AUC decay after peak vs 0.005-0.016
    # for the other variants -- see METRICS_LOG.md). Decaying lr within each 5-epoch window lets
    # it settle before the next restart kicks it back out.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_calib_auc, best_state, epochs_no_improve = 0.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).reshape(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        calib_metrics, _, _ = evaluate(model, calib_loader, device)
        if calib_metrics["auc"] > best_calib_auc:
            best_calib_auc = calib_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(f"  [{variant}] fold {fold} epoch {epoch+1}/{max_epochs}  calib_auc={calib_metrics['auc']:.4f}  best={best_calib_auc:.4f}", flush=True)
        if epochs_no_improve >= patience:
            print(f"  [{variant}] fold {fold} early stop at epoch {epoch+1}", flush=True)
            break

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save(last_state, checkpoint_dir / f"{variant}_fold{fold}_last.pt")
    torch.save(best_state, checkpoint_dir / f"{variant}_fold{fold}_best.pt")

    model.load_state_dict(best_state)
    calib_final, calib_labels, calib_probs = evaluate(model, calib_loader, device)
    threshold = find_best_threshold(calib_labels, calib_probs, THRESHOLD_MIN, THRESHOLD_MAX)
    held_out_metrics, held_out_labels, held_out_probs = evaluate(model, held_out_loader, device, threshold=threshold)
    held_out_metrics["best_calib_auc"] = best_calib_auc
    held_out_metrics["recall_by_size"] = recall_by_group(
        held_out_labels, held_out_probs, threshold, held_out_df["size_bin"].values
    )

    # Free the optimizer's per-parameter state and loaders before returning.
    del optimizer, train_loader, calib_loader, held_out_loader, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return held_out_metrics, model
