#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -t 10:00:00
#SBATCH -A lt_specific_user_name_here                    # Specify project name
#SBATCH -J compile_model_llm

module restore
module load Mamba
module load cuda/12.6
module load gcc/10.3.0

conda deactivate
conda activate ./env

# Define model and validation data paths
MODEL_DIR="/project/lt-user/uncompile_model/qwen14B"
SAVE_PATH="/project/lt-user/llm/qwen3-14B"


python script/compile_model.py \
  --model_dir ${MODEL_DIR} \
  --save_path ${SAVE_PATH}