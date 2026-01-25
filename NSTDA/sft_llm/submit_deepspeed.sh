#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -t 10:00:00
#SBATCH -A lt_specific_user_name_here                    # Specify project name
#SBATCH -J inference

module restore
module load Mamba
module load cuda/12.6
module load gcc/10.3.0

conda deactivate
conda activate ./env

HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=12802
COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES=$HOSTNAMES
echo hostname=`hostname`
echo MASTER_ADDR=$MASTER_ADDR
echo MASTER_PORT=$MASTER_PORT

HOSTFILE="$PWD/hostfile.$SLURM_JOB_ID"
> $HOSTFILE
for n in $HOSTNAMES; do
  echo "$n slots=$SLURM_GPUS_ON_NODE" >> $HOSTFILE
done
echo "Hostfile:"
cat $HOSTFILE

H=`hostname`
TIME_TAG=$(date +%F_%H-%M)

THEID=`echo -e $HOSTNAMES | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID
echo SLURM_PROCID=$SLURM_PROCID

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_TIMEOUT=3600000
export NCCL_BLOCKING_WAIT=0

deepspeed \
    --hostfile $HOSTFILE \
    --num_nodes $COUNT_NODE \
    --num_gpus 4 \
    script/deepspeed_infer.py \
        --model_name_or_path /project/lt-user/llm/qwen3-14B \
        --test_data /project/lt-user/data/text-to-gloss \
        --save_dir /project/lt-user/transcript/llm/qwen3-14B/qwen3-14B_output \