from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPER_TARGETS = {
    "accuracy": 0.9807,
    "precision": 0.9143,
    "recall": 0.9603,
    "f1": 0.9354,
    "auc": 0.9941,
}


def load_results(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CV results and compare against baseline paper metrics")
    parser.add_argument("--results", required=True, help="Path to cv_results.json")
    args = parser.parse_args()

    payload = load_results(args.results)
    summary = payload["summary"]

    print("Cross-validation summary")
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        mean = summary[metric]["mean"]
        std = summary[metric]["std"]
        target = PAPER_TARGETS[metric]
        delta = mean - target
        print(f"{metric:10s} mean={mean:.4f} std={std:.4f} target={target:.4f} delta={delta:+.4f}")


if __name__ == "__main__":
    main()
