"""Training pipeline for the kernel-fused ablation variants (fpn_locality, fpn_cross_mamba,
fpn_mamba_full) -- identical protocol to src/training/ucsf_bmsr_pipeline.py (same dataset class,
same calib-AUC checkpoint selection, same threshold/recall-by-size evaluation), but the model
comes from scripts/kernel_fpn_mamba.py (real mamba-ssm kernel) instead of src/models/fpn_mamba.py
(naive chunked scan). Runs only under the isolated venv (~/venv_mamba_kernel)."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

REPO_ROOT = Path("/lustre07/scratch/joyinola/fpn_mamba/repo")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from code.scripts.kernel_fpn_mamba import build_kernel_model
from src.evaluation.metrics import compute_metrics, find_best_threshold, recall_by_group

IMG_SIZE = 256
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
THRESHOLD_MIN, THRESHOLD_MAX = 0.05, 0.95


def dice_loss(logits, targets, eps=1e-6):
    """Soft Dice loss -- scale-invariant to the foreground/background position count, unlike plain
    BCE, which gets swamped by the overwhelming majority of background positions in a sparse
    per-position lesion-map target and just learns to suppress scores everywhere (confirmed: this
    is what happened when the auxiliary loss used plain BCE -- specificity rose, recall fell)."""
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, logits.dim()))
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class UCSFBMSRDataset(Dataset):
    """modality_mode toggle:
      "adjacent_t1c" (default, current pipeline) -- 3 adjacent T1c slices as the 3 channels.
      "per_modality" -- 1 center slice each from T1c/subtraction/FLAIR as the 3 channels
      (the second option considered for the 2.5D input design, deferred as a later ablation).
    Works unchanged for both full-slice and crop-based data -- which one you get is just a
    matter of which crop_dir/manifest/img_size the caller points at (see run_crop_*.py for the
    crop-based toggle vs run_512_*.py / run_kernel_ablation_fold.py for full-slice)."""

    def __init__(self, df, crop_dir, img_size=IMG_SIZE, augment=False, load_mask=False,
                 modality_mode="adjacent_t1c"):
        self.df = df.reset_index(drop=True)
        self.crop_dir = crop_dir
        self.img_size = img_size
        self.augment = augment
        self.load_mask = load_mask
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
            mask = z["mask"].astype(np.float32) if self.load_mask else None  # (H,W), 0/1
        t = torch.from_numpy(arr).unsqueeze(0)
        lo, hi = t.amin(dim=(2, 3), keepdim=True), t.amax(dim=(2, 3), keepdim=True)
        t = (t - lo) / (hi - lo + 1e-6)
        t = F.interpolate(t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)
        if mask is not None:
            m = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            m = F.interpolate(m, size=(self.img_size, self.img_size), mode="nearest").squeeze(0)  # (1,img,img)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                t = torch.flip(t, dims=[2])
                if mask is not None:
                    m = torch.flip(m, dims=[2])
            if torch.rand(1).item() < 0.5:
                t = torch.flip(t, dims=[1])
                if mask is not None:
                    m = torch.flip(m, dims=[1])
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        if self.load_mask:
            return t, torch.tensor(float(row["label"])), m
        return t, torch.tensor(float(row["label"]))


@torch.no_grad()
def evaluate(model, loader, device, threshold=None):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).reshape(-1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(yb.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    thr = threshold if threshold is not None else find_best_threshold(labels, probs, THRESHOLD_MIN, THRESHOLD_MAX)
    metrics = compute_metrics(labels, probs, thr, THRESHOLD_MIN, THRESHOLD_MAX)
    return metrics, labels, probs


def train_one_fold(variant, manifest_folded, fold, crop_dir, device, checkpoint_dir,
                    max_epochs=15, batch_size=4, lr=1e-4, patience=8, calib_fraction=0.15, seed=42,
                    img_size=IMG_SIZE, use_aux_loss=False, aux_weight=0.5, modality_mode="adjacent_t1c"):
    """use_aux_loss requires the model to be built with use_topk_head=True (its per-position
    score maps are what get supervised) and the crops to have a 'mask' key (see
    data/ucsf-bmsr/add_mask_to_crops.py). Auxiliary target is the true lesion footprint
    downsampled (area/soft-average, not nearest) to each pyramid level's resolution -- gives the
    per-position classifier a direct "look here" signal instead of learning location only
    indirectly through the pooled whole-slice label."""
    train_pool = manifest_folded[manifest_folded.fold != fold]
    held_out_df = manifest_folded[manifest_folded.fold == fold]

    gss = GroupShuffleSplit(n_splits=1, test_size=calib_fraction, random_state=seed)
    inner_idx, calib_idx = next(gss.split(train_pool, groups=train_pool["patient_id"]))
    inner_train_df = train_pool.iloc[inner_idx]
    calib_df = train_pool.iloc[calib_idx]

    train_ds = UCSFBMSRDataset(inner_train_df, crop_dir, img_size=img_size, augment=True,
                                load_mask=use_aux_loss, modality_mode=modality_mode)
    calib_ds = UCSFBMSRDataset(calib_df, crop_dir, img_size=img_size, augment=False, modality_mode=modality_mode)
    held_out_ds = UCSFBMSRDataset(held_out_df, crop_dir, img_size=img_size, augment=False, modality_mode=modality_mode)

    labels_arr = inner_train_df["label"].values.astype(int)
    counts = np.bincount(labels_arr)
    weights = 1.0 / counts[labels_arr]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    calib_loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    held_out_loader = DataLoader(held_out_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_kernel_model(variant, dropout=0.3, fpn_channels=256, d_state=16, freeze_stages=2,
                                img_size=img_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_calib_auc, best_state, epochs_no_improve = 0.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            if use_aux_loss:
                xb, yb, mb = batch
                mb = mb.to(device)
            else:
                xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if use_aux_loss:
                logits, score_maps = model(xb, return_aux=True)
                logits = logits.reshape(-1)
                cls_loss = loss_fn(logits, yb)
                aux_loss = 0.0
                for sm in score_maps:
                    target = F.interpolate(mb, size=sm.shape[-2:], mode="area")
                    aux_loss = aux_loss + dice_loss(sm, target)
                aux_loss = aux_loss / len(score_maps)
                loss = cls_loss + aux_weight * aux_loss
            else:
                logits = model(xb).reshape(-1)
                loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

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
    torch.save(last_state, checkpoint_dir / f"kernel_{variant}_fold{fold}_last.pt")
    torch.save(best_state, checkpoint_dir / f"kernel_{variant}_fold{fold}_best.pt")

    model.load_state_dict(best_state)
    calib_final, calib_labels, calib_probs = evaluate(model, calib_loader, device)
    threshold = find_best_threshold(calib_labels, calib_probs, THRESHOLD_MIN, THRESHOLD_MAX)
    held_out_metrics, held_out_labels, held_out_probs = evaluate(model, held_out_loader, device, threshold=threshold)
    held_out_metrics["best_calib_auc"] = best_calib_auc
    held_out_metrics["recall_by_size"] = recall_by_group(
        held_out_labels, held_out_probs, threshold, held_out_df["size_bin"].values
    )

    del optimizer, train_loader, calib_loader, held_out_loader, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return held_out_metrics, model
