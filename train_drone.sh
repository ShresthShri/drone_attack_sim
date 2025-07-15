#!/bin/bash
#SBATCH --job-name=drone_progressive
#SBATCH --partition=gpgpuB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/training_%j.log
#SBATCH --error=logs/training_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ss5921@ic.ac.uk

echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

# Navigate to project directory
cd ~/Dissertation/drone_attack_sim

# Activate virtual environment
source venv/bin/activate

# Show GPU info
nvidia-smi

# Set environment variables
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

# Run training
echo "Starting drone training..."
python agents/baseline_RL.py

echo "Job finished at: $(date)"
