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
import torch
import datetime
import os
import time
import evaluate

rouge_score = evaluate.load("/project/lt200246-mmacma/Big_seq2seq/sft_llm/sft_lora/script/rouge")

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TRL_SFTTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args, 'training_args')

    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path
    )

    lora_config = LoraConfig(
        r=2048,
        lora_alpha=2048,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(data_args.train_data, cache_dir=None)
    data_module = get_agnews_questions_for_sft(dataset, tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=data_module["train"],
        eval_dataset=data_module["eval"],
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model()
    time.sleep(300)

if __name__ == "__main__":
    train()