"""Trains one (variant, fold) pair using the kernel-fused model. Run as one of several parallel
processes (one per GPU) via scripts/run_kernel_ablation_parallel.sbatch."""
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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)

from code.scripts.kernel_ucsf_pipeline import train_one_fold

OUTPUT_DIR = Path("/lustre07/scratch/joyinola/fpn_mamba/data/ucsf-bmsr/pipeline_output_fullslice_256")
CROP_DIR = OUTPUT_DIR / "crops"

parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, required=True)
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--seed", type=int, default=42)  # training-run seed (init/dropout/sampler);
# distinct from train_one_fold's internal calib-split seed, which stays fixed at 42 so the
# multi-seed sweep isolates training variance from data-split variance.
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{args.variant} seed {args.seed} fold {args.fold}] device: {device}", flush=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

manifest_folded = pd.read_csv(OUTPUT_DIR / "manifest_with_folds.csv")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / f"seed{args.seed}"

result_path = OUTPUT_DIR / f"kernel_{args.variant}_seed{args.seed}_fold{args.fold}_result.json"
t0 = time.time()
try:
    metrics, _ = train_one_fold(
        args.variant, manifest_folded, args.fold, CROP_DIR, device, CHECKPOINT_DIR,
        max_epochs=15, batch_size=4, patience=8,
    )
    row = {"variant": args.variant, "seed": args.seed, "fold": args.fold, "wall_clock_sec": time.time() - t0, **metrics}
except Exception as e:
    print(f"ERROR [{args.variant} fold {args.fold}]: {e}", flush=True)
    traceback.print_exc()
    row = {"variant": args.variant, "fold": args.fold, "error": str(e)}

with open(result_path, "w") as f:
    json.dump(row, f)
print(f"[{args.variant} fold {args.fold}] DONE, wrote {result_path}  ({time.time()-t0:.0f}s)", flush=True)
