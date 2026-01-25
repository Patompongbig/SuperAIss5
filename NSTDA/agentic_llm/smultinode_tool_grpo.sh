#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE from main script

module restore
module load Mamba
module load cuda/12.6
module load gcc/10.3.0

conda deactivate
conda activate ./env

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES=$HOSTNAMES
echo hostname=`hostname`
echo MASTER_ADDR=$MASTER_ADDR
echo MASTER_PORT=$MASTER_PORT
echo TRAIN_NODELIST=$TRAIN_NODELIST

H=`hostname`
TIME_TAG=$(date +%F_%H-%M)

THEID=`echo -e $HOSTNAMES | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID
echo SLURM_PROCID=$SLURM_PROCID

export NCCL_TIMEOUT=3600000
export NCCL_BLOCKING_WAIT=0

accelerate launch \
    --num_processes $((4 * $COUNT_NODE)) \
    --num_machines $(($COUNT_NODE)) \
    --multi_gpu \
    --mixed_precision bf16 \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    script/train_tool_grpo.py \
        --output_dir /project/lt-user/agent_grpo/qwen3_14B_tool \
        --save_strategy "no" \
        --max_completion_length 2048 \
        --num_generations 4 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --gradient_checkpointing True \
        --learning_rate 3e-6 \
        --warmup_steps 5 \
        --max_grad_norm 0.2 \
        --lr_scheduler_type "cosine" \
        --model_name_or_path /project/lt-user/llm/qwen3-14B \
        --train_data /project/lt-user/data/text-to-gloss \
        --bf16 True \
        --deepspeed /project/lt-user/agentic_llm/deepspeed_config/deepspeed_2.json \
        --report_to "wandb" \
        --use_vllm False \
