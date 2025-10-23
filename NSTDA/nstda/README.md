# MBART Text-to-Gloss Training (LANTA HPC)

This repository contains scripts for training, compiling, and running inference with the MBART model using TRL and LoRA on the LANTA supercomputer.

---

## ⚙️ Environment Setup

Before running any scripts, set up your environment as follows:

```bash
ml reset
ml Mamba
conda deactivate
conda create -p ./env python=3.10.0 -y
conda activate ./env
conda install -c conda-forge mysqlclient pkg-config -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

## 🚀 Usage

This repository includes three main SLURM submit scripts for model training, compilation, and inference.

| File | Description |
| :--- | :--- |
| `submit_multinode_trl_sft_lora.sh` | Script for training the model with **TRL** and **LoRA** on multiple nodes. |
| `submit_compile.sh` | Script for compiling the **LoRA adapter** into the base model. |
| `submit_deepspeed.sh` | Script for performing distributed inference with **DeepSpeed**. |

### 1. Train Model

Submit the training job:

```bash
sbatch submit_multinode_trl_sft_lora.sh
```

The setup and parameters for training can be configured inside:

```bash
smultinode_trl_sft_lora.sh
```

This file defines all model paths, data locations, and training hyperparameters.

### 2. Compile LoRA Adapter

After training is complete, compile or merge the LoRA adapter into the base model:

```bash
sbatch submit_compile.sh
```

### 3. Run Multi-Node Inference

To run inference using DeepSpeed across multiple nodes:

```bash
sbatch submit_deepspeed.sh
```