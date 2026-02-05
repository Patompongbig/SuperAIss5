#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 5 -c 64                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=4		        # Specify total number of GPUs
#SBATCH -t 24:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt_specific_user_name_here                    # Specify project name
#SBATCH -J llm_finetuning               # Specify job name

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn

START=`date`
starttime=$(date +%s)

export WANDB_PROJECT="llm-training"
export WANDB_NAME="Qwen3_14B_grpo"
export WANDB_MODE="offline"
export WANDB_DIR="/project/lt-user/grpo_llm"

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
export VLLM_NODE="${NODELIST[4]}"
export TRAIN_NODELIST=$(IFS=,; echo "${NODELIST[*]:0:4}")

echo vllm node: $VLLM_NODE
echo train nodes: $TRAIN_NODELIST

srun --nodes=1 --gpus-per-node=4 --ntasks-per-node=1 --nodelist=$VLLM_NODE \
    sh smultinode_vllm.sh & VLLM_STEP_PID=$!

srun --nodes=4 --gpus-per-node=4 --ntasks-per-node=1 --nodelist=$TRAIN_NODELIST \
    sh smultinode_trl_grpo.sh
TRAIN_RC=$?

exit $TRAIN_RC