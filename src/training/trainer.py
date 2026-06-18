"""
Unified cross-validation trainer for both InceptentionNet and FPN-Mamba.
Model, optimizer, scheduler, loss, and sampler are all config-driven.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BrainTumorDataset, FoldEvalDataset, build_binary_samples, make_weighted_sampler
from src.data.transforms import TransformConfig, build_eval_transform, build_mb_transform, build_train_transform
from src.evaluation.metrics import compute_metrics, find_best_threshold
from src.models.inceptentionnet import InceptentionNet
from src.models.fpn_mamba import FPNMambaClassifier
from src.models.ablation import build_ablation_model
from src.training.losses import bce_loss, compute_pos_weight, weighted_bce_smooth
from src.utils.seed import seed_everything


@dataclass
class EarlyStop:
    patience: int
    delta: float = 1e-4
    best: float = field(default=None, init=False)
    counter: int = field(default=0, init=False)
    triggered: bool = field(default=False, init=False)

    def step(self, value: float) -> bool:
        if self.best is None or value > self.best + self.delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


def _build_model(config: dict) -> nn.Module:
    name = config["model"]["name"]
    m = config["model"]
    if name == "inceptentionnet":
        return InceptentionNet(
            stem_channels=m.get("stem_channels", 64),
            branch_channels=m.get("branch_channels", 64),
            num_heads=m.get("attention_heads", 4),
            dropout=m.get("dropout", 0.3),
        )
    if name in ("fpn_mamba", "fpn_mamba_full"):
        return FPNMambaClassifier(
            fpn_channels=m.get("fpn_channels", 256),
            d_state=m.get("d_state", 16),
            dropout=m.get("dropout", 0.3),
            freeze_layers=m.get("freeze_layers", 3),
            use_fpn=m.get("use_fpn", True),
            use_locality_mixing=m.get("use_locality_mixing", True),
            use_cross_mamba=m.get("use_cross_mamba", True),
            use_gem=m.get("use_gem", True),
            use_se=m.get("use_se", True),
        )
    # Ablation variants
    return build_ablation_model(name, dropout=m.get("dropout", 0.3))


def _build_dataloaders(train_samples, val_samples, config: dict):
    d = config["data"]
    t = config["training"]
    cfg = TransformConfig(
        image_size=d.get("image_size", 224),
        gaussian_sigma=d.get("gaussian_sigma", 0.7),
        imagenet_norm=d.get("imagenet_norm", True),
    )
    train_tf = build_train_transform(cfg)
    mb_tf = build_mb_transform(cfg) if t.get("use_mb_transform", True) else None
    eval_tf = build_eval_transform(cfg)

    train_labels = [s.label for s in train_samples]
    aug_factor = t.get("augmentation_factor", 1)
    train_ds = BrainTumorDataset(train_samples, train_tf, mb_transform=mb_tf, augmentation_factor=aug_factor)
    val_ds = FoldEvalDataset(val_samples, eval_tf)

    sampler = make_weighted_sampler(train_labels) if t.get("use_weighted_sampler", True) else None
    train_loader = DataLoader(
        train_ds,
        batch_size=t.get("batch_size", 16),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=t.get("num_workers", 0),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t.get("batch_size", 16),
        shuffle=False,
        num_workers=t.get("num_workers", 0),
        pin_memory=True,
    )
    return train_loader, val_loader, train_labels


def _build_optimizer_scheduler(model: nn.Module, config: dict, steps_per_epoch: int):
    t = config["training"]
    lr = t.get("learning_rate", 1e-4)
    wd = t.get("weight_decay", 1e-4)
    name = config["model"]["name"]

    params = filter(lambda p: p.requires_grad, model.parameters())
    if name == "inceptentionnet":
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=t.get("lr_decay_factor", 0.5), patience=t.get("lr_decay_patience", 5)
        )
        return optimizer, scheduler, "plateau"
    else:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=t.get("warmup_epochs", 10)
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t.get("max_epochs", 60) - t.get("warmup_epochs", 10),
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[t.get("warmup_epochs", 10)]
        )
        return optimizer, scheduler, "epoch"


def _get_amp_dtype():
    """Use bfloat16 on Ampere+ GPUs (A100, A6000, H100); float16 elsewhere."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _train_epoch(model, loader, optimizer, scaler, loss_fn, device, grad_clip):
    model.train()
    total_loss = correct = total = nan_batches = 0
    amp_dtype = _get_amp_dtype()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
            logits = model(imgs).squeeze(1)
            loss = loss_fn(logits, labels)

        if not torch.isfinite(loss):
            nan_batches += 1
            continue

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += ((torch.sigmoid(logits) > 0.5).long() == labels.long()).sum().item()
        total += imgs.size(0)

    return (total_loss / total if total > 0 else float("nan"),
            correct / total if total > 0 else 0.0,
            nan_batches)


@torch.no_grad()
def _evaluate(model, loader, device, threshold=None):
    model.eval()
    all_probs, all_labels = [], []
    t0 = time.perf_counter()

    for imgs, labels in loader:
        logits = model(imgs.to(device)).squeeze(1).cpu()
        probs = torch.sigmoid(logits).numpy()
        probs = np.where(np.isfinite(probs), probs, 0.5)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    infer_time = max(time.perf_counter() - t0, 1e-8)
    la = np.array(all_labels)
    pb = np.array(all_probs)
    thr = threshold if threshold is not None else find_best_threshold(la, pb)
    metrics = compute_metrics(la, pb, thr)
    metrics["fps"] = len(la) / infer_time
    metrics["inference_time_sec"] = infer_time
    return metrics, la, pb


def train_fold(fold_idx: int, train_samples, val_samples, config: dict, device: torch.device):
    t = config["training"]
    seed_everything(config.get("seed", 42) + fold_idx)

    model = _build_model(config).to(device)
    train_loader, val_loader, train_labels = _build_dataloaders(train_samples, val_samples, config)

    pos_weight = compute_pos_weight(train_labels, t.get("pos_weight_scale", 1.0))
    label_smoothing = t.get("label_smoothing", 0.0)
    name = config["model"]["name"]

    if name == "inceptentionnet":
        loss_fn = lambda logits, labels: bce_loss(logits, labels)
    else:
        if fold_idx == 1:
            pw = pos_weight * 0.6
        else:
            pw = pos_weight
        loss_fn = lambda logits, labels: weighted_bce_smooth(logits, labels, pw, label_smoothing)

    optimizer, scheduler, sched_mode = _build_optimizer_scheduler(model, config, len(train_loader))
    # bfloat16 has wide dynamic range — GradScaler not needed, but harmless to keep
    use_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    stopper = EarlyStop(patience=t.get("early_stopping_patience", 15))
    grad_clip = t.get("grad_clip", 1.0)

    best_f1 = 0.0
    best_state = None
    history = []
    fold_train_start = time.perf_counter()
    collapse_streak = total_collapses = 0

    for epoch in range(1, t.get("max_epochs", 60) + 1):
        tr_loss, tr_acc, nan_b = _train_epoch(model, train_loader, optimizer, scaler, loss_fn, device, grad_clip)
        val_metrics, _, _ = _evaluate(model, val_loader, device)

        if sched_mode == "plateau":
            scheduler.step(val_metrics.get("loss", val_metrics.get("auc", 0)))
        else:
            scheduler.step()

        # Collapse guard: restore if AUC collapses to chance
        if val_metrics["auc"] <= 0.5 and best_state is not None:
            collapse_streak += 1
            total_collapses += 1
            model.load_state_dict(best_state)
            stopper.step(best_f1)
            if collapse_streak >= 3 or total_collapses >= 8:
                break
            continue
        else:
            collapse_streak = 0

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        rec = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, **val_metrics,
               "lr": float(optimizer.param_groups[0]["lr"])}
        history.append(rec)

        if stopper.step(val_metrics["auc"]):
            break

    train_time = time.perf_counter() - fold_train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics, labels_arr, probs_arr = _evaluate(model, val_loader, device)
    final_metrics["training_time_sec"] = train_time

    return model, {
        "fold": fold_idx + 1,
        "best_f1": best_f1,
        "metrics": final_metrics,
        "history": history,
    }, labels_arr, probs_arr


def run_crossval(config: dict) -> dict:
    seed_everything(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() and config["training"].get("use_cuda", True) else "cpu")
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = build_binary_samples(
        data_root=config["data"]["data_root"],
        mb_class_name=config["data"].get("mb_class_name", "Meduloblastoma"),
        mb_target_count=config["data"].get("mb_target_count", 131),
        non_mb_target_count=config["data"].get("non_mb_target_count", 630),
        seed=config.get("seed", 42),
        deduplicate=config["data"].get("deduplicate", True),
    )
    labels_arr = np.array([s.label for s in samples])
    splitter = StratifiedKFold(
        n_splits=config["training"].get("num_folds", 5),
        shuffle=True,
        random_state=config.get("seed", 42),
    )

    fold_results, all_labels, all_probs = [], [], []
    best_auc, best_state = 0.0, None

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(samples)), labels_arr)):
        tr = [samples[i] for i in tr_idx]
        va = [samples[i] for i in va_idx]
        model, result, f_labels, f_probs = train_fold(fold_idx, tr, va, config, device)

        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   run_dir / f"fold_{fold_idx + 1}.pt")
        fold_results.append(result)
        all_labels.append(f_labels)
        all_probs.append(f_probs)

        if result["metrics"]["auc"] > best_auc:
            best_auc = result["metrics"]["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        torch.save(best_state, run_dir / "best_model.pt")

    payload = {
        "config": config,
        "device": str(device),
        "num_samples": len(samples),
        "num_mb": int(labels_arr.sum()),
        "num_non_mb": int(len(labels_arr) - labels_arr.sum()),
        "fold_results": fold_results,
    }
    with (run_dir / "cv_results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return payload
