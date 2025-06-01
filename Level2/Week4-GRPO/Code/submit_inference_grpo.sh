#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 4 -c 64                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=4		        # Specify total number of GPUs
#SBATCH -t 10:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A ai901504               # Specify project name
#SBATCH -J inference               # Specify job name


module restore
module load Mamba
module load cudatoolkit/22.7_11.7
module load gcc/10.3.0

conda deactivate
conda activate ./env

# Define model and validation data paths
MODEL_DIR="/project/ai901504-ai0004/501641_Big/output/20250514_154102/checkpoint-50"
VAL_PATH="/project/ai901504-ai0004/501641_Big/data/processed/val"

# Run inference script with command line arguments
python scripts/inference_grpo.py \
  --model_dir ${MODEL_DIR} \
  --val_path ${VAL_PATH}