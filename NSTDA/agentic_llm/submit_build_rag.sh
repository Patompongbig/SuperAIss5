#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A lt_specific_user_name_here
#SBATCH -J build_rag_index

module restore
module load Mamba
module load cuda/12.6

conda deactivate
conda activate ./env

python tools/build_rag.py