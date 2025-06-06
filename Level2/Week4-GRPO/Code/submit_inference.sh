#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 64                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=4		        # Specify total number of GPUs
#SBATCH -t 10:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A <project_name>               # Specify project name
#SBATCH -J inference               # Specify job name


module restore
module load Mamba
module load cudatoolkit/22.7_11.7
module load gcc/10.3.0

conda deactivate
conda activate ./venv

# Define model and validation data paths
MODEL_DIR="/path/to/your/model/checkpoint"
VAL_PATH="/project/ai901504-ai0004/500101-Boss/superai-llm-reasoning/SuperAI_LLM_FineTune_2025/data/processed/val"


python scripts/inference.py \
  --model_dir ${MODEL_DIR} \
  --val_path ${VAL_PATH}