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

# Set the input file (test.csv)
INPUT_FILE="/project/ai901504-ai0004/501641_Big/example/submission.csv"

# Set the output directory for the datasets
OUTPUT_DIR="data/processed/submission"
MODEL_DIR="/project/ai901504-ai0004/501641_Big/output/2025-05-15_18-59-55/checkpoint-60

# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

# Run the conversion script for all modes (sft, sft_lora, grpo)
echo "Converting CSV data to TRL datasets..."
python /project/ai901504-ai0004/501641_Big/scripts/predict_dataset.py \
    --input_file $INPUT_FILE \
    --output_dir $OUTPUT_DIR \
    --model_dir ${MODEL_DIR}

echo "Conversion complete. Datasets saved to $OUTPUT_DIR"