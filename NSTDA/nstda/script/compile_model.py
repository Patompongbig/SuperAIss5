import transformers

from dataset_sft import get_agnews_questions_for_sft
from arguments import (
    ModelArguments,
    DataArguments,
    TRL_SFTTrainingArguments
)

from datasets import load_dataset
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, AutoPeftModelForCausalLM
import argparse

import torch
import datetime
import os
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--save_path', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_dir,
            torch_dtype=torch.float16,
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model_dir,
    )

    model = model.merge_and_unload()

    model.save_pretrained(args.save_path, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(args.save_path)