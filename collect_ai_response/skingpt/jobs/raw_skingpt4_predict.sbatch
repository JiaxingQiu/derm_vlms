#!/bin/bash
#SBATCH --job-name=raw_skingpt4_predict
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH -t 4:00:00
#SBATCH -A cama
#SBATCH --export=NONE
#SBATCH --output=/scratch/jq2uw/derm_vlms/skingpt/jobs/log/slurm-%x.out
#SBATCH --error=/scratch/jq2uw/derm_vlms/skingpt/jobs/log/slurm-%x.err

set -euo pipefail

module purge
module load cuda/11.8.0
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skingpt4

echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd /scratch/jq2uw/derm_vlms/skingpt
python3 jobs/raw_skingpt4_predict.py
