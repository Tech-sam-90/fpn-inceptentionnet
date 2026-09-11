# FPN-Mamba vs. InceptentionNet -- comparison bundle

Self-contained snapshot of the code and results behind `REPORT.md` (metrics + efficiency
comparison between `fpn_mamba_full` and `inceptentionnet` on the UCSF-BMSR brain-metastasis
screening task). Read `REPORT.md` first -- this README only covers how to reproduce it.

**Base commit**: `4a6ac28afd336a8fc51021d53ee9b3dfae92f88d` (branch `version_2` of the main repo)
plus uncommitted working-tree files at the time this bundle was made -- every file under `code/`
here is a snapshot copy, not a live symlink, since several of them were not yet committed.

## Layout

```
REPORT.md                    -- the write-up: metrics + efficiency, with p-values
code/
  src/models/inceptentionnet.py       -- InceptentionNet architecture
  src/training/ucsf_bmsr_pipeline.py  -- main-venv dataset + train_one_fold (inceptentionnet, efficientnet_only, resnet50/18)
  scripts/kernel_fpn_mamba.py         -- FPNMambaClassifier + ABLATION_VARIANTS (kernel-fused Mamba)
  scripts/kernel_ucsf_pipeline.py     -- kernel-venv dataset + train_one_fold (fpn_mamba_full and ablation ladder)
  scripts/run_ablation_fold.py        -- full-slice 256px screening runner (main venv)
  scripts/run_kernel_ablation_fold.py -- full-slice 256px screening runner (kernel venv)
  scripts/run_crop_fold.py            -- crop-based per-modality runner (main venv)
  scripts/run_crop_kernel_fold.py     -- crop-based per-modality runner (kernel venv)
  scripts/run_crop_main_parallel.sbatch, run_crop_kernel_parallel.sbatch  -- 3-GPU sbatch wrappers
  scripts/consolidate_final_results.py     -- builds the full-slice significance/summary CSVs
  scripts/crop_permodality_significance.py -- builds the crop-based significance/summary CSVs
  scripts/benchmark_inceptentionnet_full.py, benchmark_kernel_full_model_512.py -- efficiency benchmarks
results/
  full_slice_screening/    -- 256px, 3 folds, seed 42 (REPORT.md Section 1a)
  crop_permodality/        -- 128px lesion-centered crops, per-modality input, 3 folds, seed 42 (REPORT.md Section 1b)
  efficiency/              -- raw benchmark logs, 256px + 512px, batch=4, single A100 (REPORT.md Section 2)
```

## Environment

Two separate virtualenvs, kept apart because `mamba-ssm`'s CUDA kernel pins an older torch:

- **Main venv** (torch 2.12.0): runs everything with no Mamba component --
  `inceptentionnet`, `efficientnet_only`, `resnet50`/`resnet18`.
- **Kernel venv** (`~/venv_mamba_kernel`: torch 2.5.1, `mamba-ssm==2.2.4`, `causal-conv1d`, `timm`,
  scikit-learn/scipy/pandas, all installed `--no-index` from the Compute Canada wheelhouse): runs
  every `fpn_*` variant (`fpn_locality`, `fpn_cross_mamba`, `fpn_mamba_full`, `fpn_mamba_full_topk`).

`code/` mirrors the original repo's package layout (`src/models/...`, `src/training/...`,
`scripts/...`) so the runner scripts' imports resolve unmodified if this folder is dropped in
place of (or merged into) a checkout of the main repo at the base commit above.

## Data

Requires the UCSF-BMSR extraction pipeline's outputs (not included here -- regenerate via the
main repo's `data/ucsf-bmsr/run_extraction_fullslice.py` and the crop-based equivalent, or point
at an existing `pipeline_output_fullslice_256/` and `pipeline_output/` directory):

- **Full-slice screening**: `pipeline_output_fullslice_256/manifest_with_folds.csv` +
  `pipeline_output_fullslice_256/crops/` (256px canonical-FOV slices).
- **Crop-based per-modality**: `pipeline_output/manifest_with_folds.csv` +
  `pipeline_output/crops/` (100mm lesion-centered crops, `t1post`/`subtraction`/`flair` all saved
  per record).

Edit `OUTPUT_DIR` near the top of each `run_*.py` if your paths differ from the originals'
(`/lustre07/scratch/joyinola/fpn_mamba/data/ucsf-bmsr/...`).

## Reproducing REPORT.md Section 1a (full-slice screening, 3 folds, seed 42)

```bash
# main venv
source ~/venv/bin/activate
for variant in inceptentionnet efficientnet_only; do
  for fold in 0 1 2; do
    python3 code/scripts/run_ablation_fold.py --variant $variant --fold $fold
  done
done

# kernel venv
source ~/venv_mamba_kernel/bin/activate
for fold in 0 1 2; do
  python3 code/scripts/run_kernel_ablation_fold.py --variant fpn_mamba_full --fold $fold
done

# back in main venv: builds paper_final_results_flat.csv, paper_final_significance.csv,
# paper_final_size_stratified.csv from the per-fold result JSONs
source ~/venv/bin/activate
python3 code/scripts/consolidate_final_results.py
```

## Reproducing REPORT.md Section 1b (crop-based, per-modality, 3 folds, seed 42)

```bash
# main venv -- inceptentionnet, efficientnet_only
source ~/venv/bin/activate
for fold in 0 1 2; do
  python3 code/scripts/run_crop_fold.py --variant inceptentionnet --seed 42 --fold $fold
  python3 code/scripts/run_crop_fold.py --variant efficientnet_only --seed 42 --fold $fold
done

# kernel venv -- fpn_mamba_full and the ablation ladder
source ~/venv_mamba_kernel/bin/activate
for variant in fpn_locality fpn_cross_mamba fpn_mamba_full fpn_mamba_full_topk; do
  for fold in 0 1 2; do
    python3 code/scripts/run_crop_kernel_fold.py --variant $variant --seed 42 --fold $fold
  done
done

# significance + summary CSVs
python3 code/scripts/crop_permodality_significance.py
```

On a Slurm cluster, `run_crop_main_parallel.sbatch` / `run_crop_kernel_parallel.sbatch` run all 3
folds of one variant in parallel across 3 GPUs (`sbatch --export=ALL,VARIANT=<name> ...`).

## Reproducing REPORT.md Section 2 (efficiency)

Single training-step timing (batch=4, forward+backward, single A100), no dataset needed --
synthetic input tensors:

```bash
source ~/venv/bin/activate
python3 code/scripts/benchmark_inceptentionnet_full.py     # -> inceptentionnet, 256px + 512px

source ~/venv_mamba_kernel/bin/activate
python3 code/scripts/benchmark_kernel_full_model_512.py     # -> fpn_mamba_full, 256px + 512px
```

## Caveats carried over from REPORT.md

- Full-slice screening is 3 folds / 1 seed -- exploratory, not confirmatory power. Only the AUC
  comparison (p=0.035) clears significance there.
- Crop-based results are also 3 folds / 1 seed (seed 42) -- every metric favors `fpn_mamba_full`
  directionally but none reaches p<0.05 yet; would need a multi-seed run to confirm.
- The ablation ladder on the crop-based pipeline (`fpn_locality`, `fpn_cross_mamba`,
  `fpn_mamba_full`, `fpn_mamba_full_topk`) clusters tightly at this sample size -- no ablation rung
  is confirmed better than another; see `results/crop_permodality/crop_permodality_significance.csv`.
