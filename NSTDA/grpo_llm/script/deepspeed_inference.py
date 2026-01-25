from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from grpo_setting import test_dataset_grpo, extract_xml_answer, extract_xml_think
from datasets import load_dataset
import torch
import torch.distributed as dist
import deepspeed
import argparse
import os
import pandas as pd
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=-1)

    return parser.parse_args()

def setup_distributed(local_rank):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ngpus = torch.cuda.device_count()

    if local_rank != -1:
        torch.cuda.set_device(local_rank % ngpus)
    else:
        torch.cuda.set_device(rank % ngpus)

    print("rank: ", rank)
    print("world_size: ", world_size)
    return rank, world_size

if __name__ == "__main__":
    args = parse_args()
    rank, world_size = setup_distributed(args.local_rank)
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    if rank == 0:
        print(f"World Size = {world_size}, Rank = {rank}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=None,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )

    ds_engine = deepspeed.init_inference(
        model,
        mp_size=world_size,
        dtype=torch.bfloat16,
        replace_with_kernel_inject=False,
    )

    model = ds_engine.module
    print(model.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.test_data, cache_dir=None)
    print(dataset)
    data_module = test_dataset_grpo(dataset["test"], tokenizer)

    text_input, model_thinking, model_answer, gloss_translate = [], [], [], []

    model.eval()
    for i, item in enumerate(data_module):
        if rank == 0:
            print(f"Processing item {i+1}/{len(data_module)}")

        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                length_penalty=0.6,
                early_stopping=True,
                num_beams=4,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.5,
            )

            if rank == 0:
                generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                think_part = extract_xml_think(generated_text)
                answer_part = extract_xml_answer(generated_text)

                print(f"Text: {item['text_raw']}")
                print(f"Generated answer: {generated_text}")

                text_input.append(item['text_raw'])
                gloss_translate.append(item["text_sign"])
                model_thinking.append(think_part)
                model_answer.append(answer_part)
    
    if rank == 0:
        result = pd.DataFrame({
            "text": text_input,
            "true_gloss": gloss_translate,
            "predicted_answer_gloss": model_answer,
            "predicted_thinking": model_thinking
        })
        result.to_csv(f"{args.save_dir}.csv", index=False)
        print(f"✅ Results saved to {args.save_dir}.csv")