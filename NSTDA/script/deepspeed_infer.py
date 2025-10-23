from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from dataset_sft import get_agnews_questions_for_sft
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


# def setup_distributed(local_rank):
#     dist.init_process_group(backend="nccl")
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()

#     if local_rank != -1:
#         torch.cuda.set_device(local_rank)
#     else:
#         torch.cuda.set_device(rank % torch.cuda.device_count())

#     return rank, world_size

def setup_distributed(local_rank):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ngpus = torch.cuda.device_count()

    if local_rank != -1:
        # Some launchers (like accelerate) set local_rank correctly
        torch.cuda.set_device(local_rank % ngpus)
    else:
        # Fall back to mapping global rank -> GPU index
        torch.cuda.set_device(rank % ngpus)

    print("rank: ", rank)
    print("world_size: ", world_size)
    return rank, world_size




if __name__ == "__main__":
    print("🚀 Starting inference with DeepSpeed model sharding...")
    args = parse_args()
    rank, world_size = setup_distributed(args.local_rank)
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    if rank == 0:
        print(f"World size = {world_size}, Rank = {rank}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=None
    )

    # config = AutoConfig.from_pretrained(args.model_name_or_path)

    # # 2. Create an empty 'meta' model without allocating memory for weights
    # with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
    #     model = AutoModelForCausalLM.from_config(config)
    
    # Shard across all GPUs
    ds_engine = deepspeed.init_inference(
        model,
        mp_size=world_size,
        dtype=torch.bfloat16,
        # replace_with_kernel_inject=False
        replace_with_kernel_inject=True
    )

    # ds_engine = deepspeed.init_inference(
    #     model=AutoModelForCausalLM.from_pretrained(
    #         args.model_name_or_path,
    #         torch_dtype=torch.bfloat16,
    #         low_cpu_mem_usage=True,
    #     ),
    #     mp_size=world_size,
    #     dtype=torch.bfloat16,
    #     replace_with_kernel_inject=True,  # if kernels compiled
    #     use_safetensors=True,
    # )

    model = ds_engine.module

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(args.test_data, cache_dir=None)
    data_module = get_agnews_questions_for_sft(dataset, tokenizer)

    text_input, gloss_translate, answer = [], [], []

    model.eval()
    for i, item in enumerate(data_module["test"]):
        if rank == 0:
            print(f"Processing item {i+1}/{len(data_module['test'])}")

        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                length_penalty=0.6,
                early_stopping=True,
                num_beams=4,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.5,
            )
            # outputs = model.generate(
            #     **inputs,
            #     max_new_tokens=300,
            #     do_sample=True,
            #     temperature=0.6,
            #     top_p=0.95,
            #     use_cache=False, 
            # )

        # Decode only on rank 0
        if rank == 0:
            generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Extract <answer>...</answer>
            match = re.search(r"<answer>(.*?)<answer>", generated_text, re.DOTALL)
            clean_answer = match.group(1).strip() if match else generated_text.split("\n")[-1].strip()

            print(f"Text: {item['text_raw']}")
            print(f"Generated answer: {generated_text}\nClean answer: {clean_answer}\n")

            text_input.append(item["text_raw"])
            gloss_translate.append(item["text_sign"])
            answer.append(clean_answer)

    # Save only on rank 0
    if rank == 0:
        result = pd.DataFrame({
            "text": text_input,
            "true_gloss": gloss_translate,
            "predicted_gloss": answer
        })
        result.to_csv(f"{args.save_dir}.csv", index=False)
        print(f"✅ Results saved to {args.save_dir}.csv")
