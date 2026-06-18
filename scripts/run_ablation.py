"""
Ablation runner: trains all five variants in sequence and writes a comparison table.

Usage:
    python scripts/run_ablation.py --config configs/ablation.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.ablation import ABLATION_VARIANTS
from src.training.trainer import run_crossval
from src.evaluation.metrics import summarize_folds


ORDERED_VARIANTS = [
    "efficientnet_only",
    "fpn_standard",
    "fpn_locality",
    "fpn_cross_mamba",
    "fpn_mamba_full",
]

DISPLAY_NAMES = {
    "efficientnet_only": "EfficientNet-B2 only",
    "fpn_standard":      "+ FPN (standard 3×3)",
    "fpn_locality":      "+ LocalityMixing (Mamba neck)",
    "fpn_cross_mamba":   "+ Cross-scale Mamba",
    "fpn_mamba_full":    "+ GeM + SE (full model)",
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to ablation YAML config")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=ORDERED_VARIANTS,
        help="Subset of variants to run (default: all five)",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    ablation_results: dict[str, dict] = {}

    for variant in args.variants:
        if variant not in ABLATION_VARIANTS:
            print(f"[SKIP] Unknown variant '{variant}'")
            continue

        print(f"\n{'='*60}")
        print(f"  Variant: {DISPLAY_NAMES.get(variant, variant)}")
        print(f"{'='*60}")

        variant_run_dir = Path(base_config["output"]["run_dir"]) / variant
        cv_results_path = variant_run_dir / "cv_results.json"

        if cv_results_path.exists():
            print(f"[RESUME] Found existing results at {cv_results_path} — skipping training.")
            with cv_results_path.open(encoding="utf-8") as f:
                payload = json.load(f)
            summary = summarize_folds(payload["fold_results"])
            ablation_results[variant] = summary
            continue

        config = {**base_config}
        config["model"] = {**base_config.get("model", {}), "name": variant, **ABLATION_VARIANTS[variant]}
        config["output"] = {"run_dir": str(variant_run_dir)}

        payload = run_crossval(config)
        summary = summarize_folds(payload["fold_results"])
        ablation_results[variant] = summary

    # Print ablation table
    metrics_to_show = ["accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "auc"]
    print(f"\n{'='*90}")
    print("  ABLATION TABLE")
    print(f"{'='*90}")
    header = f"{'Variant':<32}" + "".join(f"{m:>10}" for m in metrics_to_show)
    print(header)
    print("-" * len(header))
    for variant in args.variants:
        if variant not in ablation_results:
            continue
        row = f"{DISPLAY_NAMES.get(variant, variant):<32}"
        for m in metrics_to_show:
            val = ablation_results[variant].get(m, {}).get("mean", float("nan"))
            row += f"{val:>10.4f}"
        print(row)

    # Save full results
    out_path = Path(base_config["output"]["run_dir"]) / "ablation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, default=str)
    print(f"\nAblation summary saved to {out_path}")


if __name__ == "__main__":
    main()
