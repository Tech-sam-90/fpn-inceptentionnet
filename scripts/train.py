"""
Unified training entry point.

Usage:
    python scripts/train.py --config configs/inceptentionnet.yaml
    python scripts/train.py --config configs/fpn_mamba.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Ensure src/ is on the path when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import run_crossval
from src.evaluation.metrics import summarize_folds
from src.evaluation.stats import compare_models, print_comparison_table


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--baseline_results",
        default=None,
        help="Optional path to a baseline cv_results.json for comparison table",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Training: {config['model']['name']}")
    print(f"Run dir : {config['output']['run_dir']}")

    payload = run_crossval(config)
    summary = summarize_folds(payload["fold_results"])

    print("\n=== Cross-Validation Summary ===")
    for metric, stats in summary.items():
        print(f"  {metric:<18}: {stats['mean']:.4f} ± {stats['std']:.4f}  "
              f"[{stats['min']:.4f} – {stats['max']:.4f}]")

    if args.baseline_results:
        with open(args.baseline_results, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        table = compare_models(
            baseline["fold_results"],
            payload["fold_results"],
            name_a="Baseline",
            name_b=config["model"]["name"],
        )
        print("\n=== Statistical Comparison ===")
        print_comparison_table(table, name_a="Baseline", name_b=config["model"]["name"])


if __name__ == "__main__":
    main()
