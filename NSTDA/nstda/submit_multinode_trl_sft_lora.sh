#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 5 -c 64                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=4		        # Specify total number of GPUs
#SBATCH -t 24:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200246                    # Specify project name
#SBATCH -J llm_finetuning               # Specify job name

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn

START=`date`
starttime=$(date +%s)

export WANDB_PROJECT="seq2seq-training-dataver5"
# export WANDB_NAME="qwen3_run_$(date +%F_%H-%M)"
export WANDB_NAME="qwen14B_largeparam"
export WANDB_MODE="offline"  
export WANDB_DIR=/project/lt200246-mmacma/Big_seq2seq/sft_llm/wandb_report

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

srun sh smultinode_trl_sft_lora.sh