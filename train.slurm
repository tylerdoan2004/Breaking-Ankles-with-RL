#!/bin/bash
#SBATCH --job-name=reactive_avoidance
#SBATCH --account=cs175_class
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# ============================================
# Reactive Avoidance RL Training - SLURM Script
# ============================================
# Usage:
#   1. Transfer project to the cluster
#   2. Set up a virtual environment (see below)
#   3. Submit: sbatch train.slurm
#   4. Monitor: squeue -u $USER
#   5. View logs: tail -f slurm_logs/train_<JOB_ID>.out
# ============================================

# --- Configuration ---
# Adjust these to match your cluster setup
PROJECT_DIR="$HOME/ReactiveAvoidanceRL"
CONDA_ENV="reactive_avoidance"

# --- Setup ---
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Create log directory
mkdir -p "$PROJECT_DIR/slurm_logs"

# Navigate to project directory
cd "$PROJECT_DIR" || exit 1

# Load any required modules (uncomment/modify as needed for your cluster)
# module load python/3.11
# module load cuda/12.0

# Activate conda environment
# If this is the first run, create the env first:
#   conda create -n reactive_avoidance python=3.11 -y
#   conda activate reactive_avoidance
#   pip install -r requirements.txt
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# --- Training ---
echo "Starting training..."
python -m src.train

echo "Training completed at: $(date)"
