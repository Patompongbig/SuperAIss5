#!/bin/bash
#SBATCH -p compute                          # Specify partition
#SBATCH -N 1 -c 16                          # Adjust CPU cores as needed
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00                          # Adjust time as needed
#SBATCH -A ai901504                         # Your account
#SBATCH -J hf_download                     # Job name
#SBATCH -o logs/hf_download_%j.out         # Output log
#SBATCH -e logs/hf_download_%j.err         # Error log

# Load modules and activate environment
ml reset
ml Mamba
conda deactivate
conda activate ./env

# Run the model download script
python scripts/hf_download.py
