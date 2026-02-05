# Text-to-Gloss Training (GRPO)

This repository contains scripts for training and inference with the LLM model using **GRPO** on the LANTA supercomputer.

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

This repository includes SLURM submit scripts for model training and inference.

| File | Description |
| :--- | :--- |
| `submit_multinode_grpo.sh` | Script for training the model with **GRPO** on multiple nodes. |
| `submit_inference_grpo.sh` | Script for performing inference. |

### 1. Train Model

Submit the training job:

```bash
sbatch submit_multinode_grpo.sh
```

The setup and parameters for training can be configured inside:

```bash
smultinode_trl_grpo.sh
```
The setup of vllm for training can be configured inside (setup model to be the same as the model used in training):

```bash
smultinode_vllm.sh
```

This file defines all model paths, data locations, and training hyperparameters.
It uses **DeepSpeed 3** and **vllm** for efficient training.

### 2. Run Inference

To run inference:

```bash
sbatch submit_inference_grpo.sh
```
