"""Paired t-tests (paired by fold, n=3) between the 3 crop-based per-modality seed-42 variants:
inceptentionnet, efficientnet_only, fpn_mamba_full. Same pattern as stage2_final_analysis.py but
paired by fold instead of seed, since this is a single-seed (42) run."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULT_DIR = Path("/lustre07/scratch/joyinola/fpn_mamba/data/ucsf-bmsr/pipeline_output/results_crop_permodality")

VARIANTS = ["inceptentionnet", "efficientnet_only", "fpn_locality", "fpn_cross_mamba",
            "fpn_mamba_full", "fpn_mamba_full_topk"]

def load_by_fold(variant):
    out = {}
    for f in sorted(RESULT_DIR.glob(f"{variant}_seed42_fold*_result.json")):
        with open(f) as fh:
            r = json.load(fh)
        out[r["fold"]] = {
            "auc": r["auc"],
            "recall": r["recall"],
            "specificity": r["specificity"],
            "tiny_recall": r["recall_by_size"]["tiny"]["recall"],
            "small_recall": r["recall_by_size"]["small"]["recall"],
        }
    return out

by_fold = {v: load_by_fold(v) for v in VARIANTS}

print("=== Crop-based per-modality, seed 42: mean +/- std across 3 folds ===")
summary_rows = []
for v, fm in by_fold.items():
    n = len(fm)
    row = {"variant": v, "n_folds": n}
    for metric in ["auc", "recall", "specificity", "tiny_recall", "small_recall"]:
        vals = [fm[f][metric] for f in fm]
        row[f"{metric}_mean"] = np.mean(vals)
        row[f"{metric}_std"] = np.std(vals)
    summary_rows.append(row)
    print(f"{v} (n={n}): AUC={row['auc_mean']:.4f}+/-{row['auc_std']:.4f}  "
          f"recall={row['recall_mean']:.4f}+/-{row['recall_std']:.4f}  "
          f"spec={row['specificity_mean']:.4f}+/-{row['specificity_std']:.4f}  "
          f"tiny_recall={row['tiny_recall_mean']:.4f}+/-{row['tiny_recall_std']:.4f}  "
          f"small_recall={row['small_recall_mean']:.4f}+/-{row['small_recall_std']:.4f}")

pd.DataFrame(summary_rows).to_csv(RESULT_DIR / "crop_permodality_summary.csv", index=False)

print("\n=== Pairwise paired t-tests (paired by fold, n=3) ===")
sig_rows = []
for v1, v2 in combinations(VARIANTS, 2):
    common_folds = sorted(set(by_fold[v1]) & set(by_fold[v2]))
    for metric in ["auc", "recall", "specificity", "tiny_recall", "small_recall"]:
        a = [by_fold[v1][f][metric] for f in common_folds]
        b = [by_fold[v2][f][metric] for f in common_folds]
        t, p = stats.ttest_rel(a, b)
        sig_rows.append({"v1": v1, "v2": v2, "metric": metric, "n": len(common_folds),
                          "v1_mean": np.mean(a), "v2_mean": np.mean(b), "t": t, "p": p})
        flag = "**" if p < 0.05 else ""
        print(f"{v1} vs {v2} [{metric}]: {np.mean(a):.4f} vs {np.mean(b):.4f}  t={t:.3f} p={p:.4f} {flag}")

pd.DataFrame(sig_rows).to_csv(RESULT_DIR / "crop_permodality_significance.csv", index=False)
print("\nWrote crop_permodality_summary.csv, crop_permodality_significance.csv")
