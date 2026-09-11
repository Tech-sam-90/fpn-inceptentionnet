"""Trains one (variant, fold) pair on the ORIGINAL crop-based pipeline (100mm lesion-centered
crops, native pixel size resized to 128px) -- for variants with no Mamba component (resnet50,
resnet18, efficientnet_only, inceptentionnet). Main venv (torch 2.12.0).

TOGGLE: this is the crop-based counterpart to run_512_fold.py (full-slice). Same UCSFBMSRDataset
class handles both -- swap OUTPUT_DIR/IMG_SIZE below (or use run_512_fold.py) to switch back to
full-slice; nothing else needs to change. modality_mode="per_modality" here uses 1 center slice
each from T1c/subtraction/FLAIR instead of 3 adjacent T1c slices -- the second 2.5D-construction
option considered earlier, tested here since crops are already tight enough that losing the
2-slice depth context matters less than it would on a full slice.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path("/lustre07/scratch/joyinola/fpn_mamba/repo")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)

from code.src.training.ucsf_bmsr_pipeline import train_one_fold

# --- TOGGLE: crop-based (this file) vs full-slice (run_512_fold.py / run_ablation_fold.py) ---
OUTPUT_DIR = Path("/lustre07/scratch/joyinola/fpn_mamba/data/ucsf-bmsr/pipeline_output")  # crop-based
CROP_DIR = OUTPUT_DIR / "crops"
IMG_SIZE = 128
MODALITY_MODE = "per_modality"  # 1 center slice each: T1c, subtraction, FLAIR
# ------------------------------------------------------------------------------------------------

RESULT_DIR = OUTPUT_DIR / "results_crop_permodality"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints_crop_permodality"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, required=True)
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{args.variant} seed {args.seed} fold {args.fold}] crop-based, {MODALITY_MODE}, device: {device}", flush=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

manifest_folded = pd.read_csv(OUTPUT_DIR / "manifest_with_folds.csv")

result_path = RESULT_DIR / f"{args.variant}_seed{args.seed}_fold{args.fold}_result.json"
t0 = time.time()
try:
    metrics, _ = train_one_fold(
        args.variant, manifest_folded, args.fold, CROP_DIR, device, CHECKPOINT_DIR,
        max_epochs=15, batch_size=4, patience=8, img_size=IMG_SIZE, modality_mode=MODALITY_MODE,
    )
    row = {"variant": args.variant, "seed": args.seed, "fold": args.fold,
           "wall_clock_sec": time.time() - t0, **metrics}
except Exception as e:
    print(f"ERROR [{args.variant} fold {args.fold}]: {e}", flush=True)
    traceback.print_exc()
    row = {"variant": args.variant, "fold": args.fold, "error": str(e)}

with open(result_path, "w") as f:
    json.dump(row, f)
print(f"[{args.variant} fold {args.fold}] DONE, wrote {result_path}  ({time.time()-t0:.0f}s)", flush=True)
