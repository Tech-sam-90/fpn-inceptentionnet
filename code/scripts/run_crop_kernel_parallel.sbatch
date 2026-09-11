#!/bin/bash
#SBATCH --account=def-uanazodo-ab_gpu
#SBATCH --job-name=crop_permod_kernel
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=/scratch/joyinola/fpn_mamba/logs/%x_%j.log
set -euo pipefail
module load python/3.11.5
source /home/joyinola/venv_mamba_kernel/bin/activate
cd "$SCRATCH/fpn_mamba/repo"

echo "[$(date -Is)] crop-based per-modality seed=42 run for variant=${VARIANT} on $(hostname)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_crop_kernel_fold.py --variant "${VARIANT}" --seed 42 --fold 0 &
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_crop_kernel_fold.py --variant "${VARIANT}" --seed 42 --fold 1 &
CUDA_VISIBLE_DEVICES=2 python3 scripts/run_crop_kernel_fold.py --variant "${VARIANT}" --seed 42 --fold 2 &
wait
echo "[$(date -Is)] finished for variant=${VARIANT}"
