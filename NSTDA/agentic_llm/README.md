# Text-to-Gloss Training (Agentic Tool)

> [!WARNING]
> This module is currently **UNFINISHED** due to a mismatch in `vllm` versions, making training difficult.

This repository contains scripts for agentic tool training using **DeepSpeed 2**.

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
pip install --pre transformers
pip install -e .
```

## 🚀 Usage

This repository includes SLURM submit scripts for model training and tool testing.
**Note**: This setup uses **DeepSpeed 2** and **No vllm**.

| File | Description |
| :--- | :--- |
| `submit_build_rag.sh` | Script for building the RAG index. |
| `submit_tool_train.sh` | Script for training the tool model. |
| `submit_test_tool.sh` | Script for testing the tool. |

### 1. Build RAG

To build the RAG index (First Step):

```bash
sbatch submit_build_rag.sh
```

### 2. Train Tool

Submit the training job:

```bash
sbatch submit_tool_train.sh
```

The setup and parameters for training can be configured inside:

```bash
smultinode_tool_grpo.sh
```

This file defines all model paths, data locations, and training hyperparameters.

### 3. Test Tool

To test the tool:

```bash
sbatch submit_test_tool.sh
```
