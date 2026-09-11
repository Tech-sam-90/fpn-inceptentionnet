"""Consolidates all UCSF-BMSR full-slice grid results (baselines + full 5-variant ablation ladder,
kernel-fused fpn_mamba_full as the primary reported result) into one clean set of tables:
pooled metrics, size-stratified recall, ablation deltas, statistical tests, and the
naive-vs-kernel wall-clock comparison. Writes paper_final_results.csv + prints everything."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path("/lustre07/scratch/joyinola/fpn_mamba/data/ucsf-bmsr/pipeline_output_fullslice_256")
ARCHIVE = OUT_DIR / "archive_no_scheduler"

# ---- gather all rows ----
grid = pd.read_csv(OUT_DIR / "grid_results.csv")
baseline_rows = grid[grid.variant.isin(["inceptentionnet", "resnet50", "efficientnet_only"])].to_dict("records")

rows = list(baseline_rows)

# fpn_standard (main venv, no mamba)
for fold in range(3):
    with open(OUT_DIR / f"fpn_standard_fold{fold}_result.json") as f:
        r = json.load(f)
    rows.append(r)

# kernel-fused: fpn_locality, fpn_cross_mamba, fpn_mamba_full (this is now THE reported fpn_mamba_full)
for variant in ["kernel_fpn_locality", "kernel_fpn_cross_mamba", "kernel_fpn_mamba_full"]:
    clean_name = variant.replace("kernel_", "")
    for fold in range(3):
        with open(OUT_DIR / f"{variant}_fold{fold}_result.json") as f:
            r = json.load(f)
        r["variant"] = clean_name  # normalize name (drop "kernel_" prefix in the final table)
        rows.append(r)

# naive fpn_mamba_full (fixed-LR, archived) -- kept only for the naive-vs-kernel wall-clock comparison
naive_fm_rows = []
for fold in range(3):
    with open(ARCHIVE / f"fpn_mamba_full_fold{fold}_result.json") as f:
        r = json.load(f)
    naive_fm_rows.append(r)

df = pd.DataFrame(rows)
df["recall_by_size"] = df["recall_by_size"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

VARIANT_ORDER = ["inceptentionnet", "resnet50", "efficientnet_only", "fpn_standard",
                  "fpn_locality", "fpn_cross_mamba", "fpn_mamba_full"]

print("=== Pooled metrics, mean +/- std across 3 folds ===")
pooled = df.groupby("variant")[["auc", "recall", "specificity"]].agg(["mean", "std"]).reindex(VARIANT_ORDER)
print(pooled.round(4))

print("\n=== Size-stratified recall (n-weighted across folds) ===")
long_rows = []
for _, r in df.iterrows():
    for size, d in r["recall_by_size"].items():
        long_rows.append({"variant": r["variant"], "size_bin": size, "n": d["n"], "recall": d["recall"]})
size_df = pd.DataFrame(long_rows)
def wavg(g):
    return (g["recall"] * g["n"]).sum() / g["n"].sum()
size_summary = size_df.groupby(["variant", "size_bin"]).apply(wavg).unstack()[["tiny", "small", "medium", "large"]]
print(size_summary.reindex(VARIANT_ORDER).round(4))

print("\n=== Ablation ladder deltas (pooled AUC) ===")
ladder = ["efficientnet_only", "fpn_standard", "fpn_locality", "fpn_cross_mamba", "fpn_mamba_full"]
prev_auc = None
for v in ladder:
    auc = df[df.variant == v]["auc"].mean()
    delta = "" if prev_auc is None else f"  (delta {auc - prev_auc:+.4f})"
    print(f"{v}: AUC={auc:.4f}{delta}")
    prev_auc = auc

print("\n=== Paired t-tests across 3 folds: fpn_mamba_full (kernel) vs. each other variant ===")
fm = df[df.variant == "fpn_mamba_full"].sort_values("fold")
sig_rows = []
for metric in ["auc", "recall", "specificity"]:
    for v in [x for x in VARIANT_ORDER if x != "fpn_mamba_full"]:
        other = df[df.variant == v].sort_values("fold")
        if len(other) != 3:
            continue
        t, p = stats.ttest_rel(fm[metric].values, other[metric].values)
        sig_rows.append({"metric": metric, "vs": v, "fpn_mamba_full_mean": fm[metric].mean(),
                          "other_mean": other[metric].mean(), "t": t, "p": p})
sig_df = pd.DataFrame(sig_rows)
print(sig_df.round(4).to_string(index=False))

# tiny-recall paired tests
print()
tiny_recalls = {}
for v in VARIANT_ORDER:
    sub = df[df.variant == v].sort_values("fold")
    tiny_recalls[v] = [r["tiny"]["recall"] for r in sub["recall_by_size"]]
for v in [x for x in VARIANT_ORDER if x != "fpn_mamba_full"]:
    t, p = stats.ttest_rel(tiny_recalls["fpn_mamba_full"], tiny_recalls[v])
    print(f"tiny_recall fpn_mamba_full vs {v}: t={t:.3f} p={p:.4f} "
          f"(means {np.mean(tiny_recalls['fpn_mamba_full']):.4f} vs {np.mean(tiny_recalls[v]):.4f})")
    sig_rows.append({"metric": "tiny_recall", "vs": v, "fpn_mamba_full_mean": np.mean(tiny_recalls["fpn_mamba_full"]),
                      "other_mean": np.mean(tiny_recalls[v]), "t": t, "p": p})

print("\n=== Naive vs. kernel-fused fpn_mamba_full: wall-clock + accuracy ===")
naive_wall = []
kernel_wall = df[df.variant == "fpn_mamba_full"]["wall_clock_sec"].tolist()
for fold in range(3):
    p = ARCHIVE / f"fpn_mamba_full_fold{fold}_last.pt"
naive_fold_times_sec = [17411, 17054, 14815]  # derived earlier from checkpoint mtimes (job 2664874)
for fold in range(3):
    naive_auc = naive_fm_rows[fold]["auc"]
    kernel_row = df[(df.variant == "fpn_mamba_full") & (df.fold == fold)].iloc[0]
    print(f"fold{fold}: naive={naive_fold_times_sec[fold]}s auc={naive_auc:.4f}  |  "
          f"kernel={kernel_row['wall_clock_sec']:.0f}s auc={kernel_row['auc']:.4f}  "
          f"speedup={naive_fold_times_sec[fold]/kernel_row['wall_clock_sec']:.1f}x")
avg_naive = np.mean(naive_fold_times_sec)
avg_kernel = np.mean(kernel_wall)
print(f"\nAvg fold wall-clock: naive={avg_naive:.0f}s ({avg_naive/60:.1f}min)  "
      f"kernel={avg_kernel:.0f}s ({avg_kernel/60:.1f}min)  speedup={avg_naive/avg_kernel:.1f}x")

df.drop(columns=["recall_by_size"]).to_csv(OUT_DIR / "paper_final_results_flat.csv", index=False)
pd.DataFrame(sig_rows).to_csv(OUT_DIR / "paper_final_significance.csv", index=False)
size_summary.to_csv(OUT_DIR / "paper_final_size_stratified.csv")
print("\nWrote paper_final_results_flat.csv, paper_final_significance.csv, paper_final_size_stratified.csv")
