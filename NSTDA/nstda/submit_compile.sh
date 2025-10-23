#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -t 10:00:00
#SBATCH -A lt200246                    # Specify project name
#SBATCH -J compile_model_llm

module restore
module load Mamba
module load cuda/12.6
module load gcc/10.3.0

conda deactivate
conda activate ./env

# Define model and validation data paths
MODEL_DIR="/project/lt200246-mmacma/Big_seq2seq/sft_llm/trained_model/uncompile_model/qwen14B_largeparam"
SAVE_PATH="/project/lt200246-mmacma/Big_seq2seq/trained_model/model_use_data5/llm/qwen3-14B-largeparam"


python script/compile_model.py \
  --model_dir ${MODEL_DIR} \
  --save_path ${SAVE_PATH}