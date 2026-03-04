from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import FoldEvalDataset, FoldTrainDataset, build_binary_samples
from models.inceptentionnet import InceptentionNet
from transforms import BaselineTransformConfig, QuadAugmentDatasetTransform, build_eval_transform


@dataclass
class EarlyStopState:
    best_loss: float
    best_epoch: int
    wait: int
    best_state_dict: dict | None


def load_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(train_samples, val_samples, config: dict):
    transform_cfg = BaselineTransformConfig(
        image_size=config["data"]["image_size"],
        gaussian_sigma=config["data"]["gaussian_sigma"],
    )
    train_transform = QuadAugmentDatasetTransform(
        image_size=transform_cfg.image_size,
        mean=transform_cfg.mean,
        std=transform_cfg.std,
    )
    eval_transform = build_eval_transform(transform_cfg)

    train_dataset = FoldTrainDataset(
        train_samples,
        transform=train_transform,
        augmentation_factor=config["data"]["augmentation_factor"],
    )
    val_dataset = FoldEvalDataset(val_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    losses = []
    targets: list[float] = []
    probabilities: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            losses.append(loss.item())
            targets.extend(labels.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())

    predicted = [1 if p >= 0.5 else 0 for p in probabilities]
    targets_np = np.array(targets)
    probs_np = np.array(probabilities)
    metrics = {
        "loss": float(np.mean(losses) if losses else 0.0),
        "accuracy": float(accuracy_score(targets_np, predicted)),
        "precision": float(precision_score(targets_np, predicted, zero_division=0)),
        "recall": float(recall_score(targets_np, predicted, zero_division=0)),
        "f1": float(f1_score(targets_np, predicted, zero_division=0)),
        "auc": float(roc_auc_score(targets_np, probs_np)) if len(np.unique(targets_np)) == 2 else 0.0,
    }
    return metrics


def train_one_fold(fold_index: int, train_samples, val_samples, config: dict, device: torch.device):
    model = InceptentionNet(
        stem_channels=config["model"]["stem_channels"],
        branch_channels=config["model"]["branch_channels"],
        num_heads=config["model"]["attention_heads"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["lr_decay_factor"],
        patience=config["training"]["lr_decay_patience"],
    )
    criterion = nn.BCEWithLogitsLoss()

    train_loader, val_loader = build_dataloaders(train_samples, val_samples, config)

    early_stop = EarlyStopState(best_loss=float("inf"), best_epoch=-1, wait=0, best_state_dict=None)
    history = []

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        epoch_losses = []
        progress = tqdm(train_loader, desc=f"Fold {fold_index + 1} | Epoch {epoch + 1}", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device, criterion)
        train_loss = float(np.mean(epoch_losses) if epoch_losses else 0.0)
        scheduler.step(val_metrics["loss"])

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        if val_metrics["loss"] < early_stop.best_loss:
            early_stop.best_loss = val_metrics["loss"]
            early_stop.best_epoch = epoch + 1
            early_stop.wait = 0
            early_stop.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            early_stop.wait += 1

        if early_stop.wait >= config["training"]["early_stopping_patience"]:
            break

    if early_stop.best_state_dict is not None:
        model.load_state_dict(early_stop.best_state_dict)

    final_metrics = evaluate(model, val_loader, device, criterion)
    fold_result = {
        "fold": fold_index + 1,
        "best_epoch": early_stop.best_epoch,
        "best_val_loss": early_stop.best_loss,
        "metrics": final_metrics,
        "history": history,
    }
    return model, fold_result


def summarize_results(fold_results: list[dict]) -> dict:
    keys = ["accuracy", "precision", "recall", "f1", "auc"]
    summary = {}
    for key in keys:
        values = np.array([fold["metrics"][key] for fold in fold_results], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InceptentionNet baseline with 5-fold cross-validation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and config["training"]["use_cuda"] else "cpu")
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = build_binary_samples(
        data_root=config["data"]["data_root"],
        mb_class_name=config["data"]["mb_class_name"],
        mb_target_count=config["data"]["mb_target_count"],
        non_mb_target_count=config["data"]["non_mb_target_count"],
        seed=config["seed"],
        deduplicate=config["data"]["deduplicate"],
    )
    labels = np.array([sample.label for sample in samples])

    splitter = StratifiedKFold(n_splits=config["training"]["num_folds"], shuffle=True, random_state=config["seed"])

    fold_results: list[dict] = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(samples)), labels)):
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        model, fold_result = train_one_fold(fold_index, train_samples, val_samples, config, device)

        model_path = run_dir / f"fold_{fold_index + 1}.pt"
        torch.save(model.state_dict(), model_path)
        fold_results.append(fold_result)

    summary = summarize_results(fold_results)
    payload = {
        "config": config,
        "device": str(device),
        "num_samples": len(samples),
        "num_mb": int(labels.sum()),
        "num_non_mb": int(len(labels) - labels.sum()),
        "fold_results": fold_results,
        "summary": summary,
    }

    result_path = run_dir / "cv_results.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
