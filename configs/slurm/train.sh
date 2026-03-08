#!/bin/bash
#SBATCH --account=cs175_class
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=16

srun python3 -m src.train
