"""Trains one (variant, fold) pair on the main venv (torch 2.12.0) -- for ablation variants with
no Mamba component (fpn_standard), where the naive-vs-kernel distinction doesn't apply."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import json
import sys
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

OUTPUT_DIR = Path("/lustre07/scratch/joyinola/fpn_mamba/data/ucsf-bmsr/pipeline_output_fullslice_256")
CROP_DIR = OUTPUT_DIR / "crops"

parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, required=True)
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--seed", type=int, default=None)  # training-run seed (init/dropout/sampler);
# omit for the original single-seed screening runs; when set, namespaces checkpoints/results and
# keeps train_one_fold's internal calib-split seed fixed so the sweep isolates training variance.
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{args.variant} fold {args.fold}]" + (f" seed {args.seed}" if args.seed is not None else "") + f" device: {device}", flush=True)

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

manifest_folded = pd.read_csv(OUTPUT_DIR / "manifest_with_folds.csv")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / (f"seed{args.seed}" if args.seed is not None else "")

suffix = f"_seed{args.seed}" if args.seed is not None else ""
result_path = OUTPUT_DIR / f"{args.variant}{suffix}_fold{args.fold}_result.json"
try:
    metrics, _ = train_one_fold(
        args.variant, manifest_folded, args.fold, CROP_DIR, device, CHECKPOINT_DIR,
        max_epochs=15, batch_size=4, patience=8,
    )
    row = {"variant": args.variant, "seed": args.seed, "fold": args.fold, **metrics}
except Exception as e:
    print(f"ERROR [{args.variant} fold {args.fold}]: {e}", flush=True)
    traceback.print_exc()
    row = {"variant": args.variant, "fold": args.fold, "error": str(e)}

with open(result_path, "w") as f:
    json.dump(row, f)
print(f"[{args.variant} fold {args.fold}] DONE, wrote {result_path}", flush=True)
