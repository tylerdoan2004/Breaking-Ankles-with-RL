#!/bin/bash
#SBATCH --job-name=reactive_avoidance
#SBATCH --account=cs175_class_gpu
#SBATCH --partition=gpu           # Change if needed (check: sinfo -s)
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=8         # 8 CPUs for parallel envs
#SBATCH --mem=16G                 # 16GB RAM
#SBATCH --time=08:00:00           # 8 hour time limit
#SBATCH --output=slurm_%j.out     # Output log (%j = job ID)
#SBATCH --error=slurm_%j.err      # Error log

# ── Load modules ──
module load python/3.10.2

# ── Activate your virtual environment ──
# Option A: conda
# conda activate venv

# Option B: local venv
# source .venv/bin/activate

# ── Install dependencies (first run only) ──
# pip install -r requirements.txt

# ── Run single-agent randomized training ──
echo "Starting single-agent training at $(date)"
python3 -m src.train_ppo
echo "Training completed at $(date)"

# ── Uncomment below for multi-agent training instead ──
# echo "Starting multi-agent training at $(date)"
# python3 -m src.train_multi_agent_ppo
# echo "Training completed at $(date)"
