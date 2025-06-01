#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
from datasets import Dataset
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from llm_finetune.grpo import SYSTEM_PROMPT_REASONING

BATCH_SZ  = 32
MAX_GEN   = 1024

# Logging time stamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Recieve Argument for setting up the dataset
def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV data to TRL dataset format and predict")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed dataset")
    parser.add_argument('--model_dir', type=str, help='Path to the model directory')
    return parser.parse_args()

# Read the CSV file using Pandas
def read_csv(file_path):
    """Read the CSV file and return a pandas DataFrame."""
    logger.info(f"Reading CSV file from {file_path}")
    df = pd.read_csv(file_path)
    return df

# Loop and create dictionary fo reach row
def create_example(row):
    input_train = f"Title: {row['id']}\nDescription: {row['story']}\n"
    answer_str = f"<answer>{row['answers']}</answer>"
    return {
        "id": row["id"],                # <-- Add this line
        "input_train": input_train,
        "answer": answer_str
    }



# Loop and create dataset format
def prepare_dataset(df):
    
    # Handle missing values
    df = df.fillna("")
    
    # Loop and create dictionary for each row
    examples = [create_example(row) for _, row in df.iterrows()]
    
    # Create a Dataset object
    dataset = Dataset.from_dict({
        key: [example[key] for example in examples]
        for key in examples[0].keys()
    })
    
    return dataset


# Save to target directory
def save_dataset_to_csv(df, dataset, predicted_labels, output_dir):
    logger.info(f"Saving dataset and predictions to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    data_dict = {
        "id": df["id"],                            # from original df
        "story": dataset["input_train"],           # from processed dataset
        "answers": predicted_labels                # from predictions
    }
    df_out = pd.DataFrame(data_dict)

    csv_path = os.path.join(output_dir, "predictions.csv")
    df_out.to_csv(csv_path, index=False)
    logger.info(f"Predictions saved to {csv_path}")


def extract_xml_answer(text: str) -> str:
        # exactly your earlier logic
        return text.split("<answer>")[-1].split("</answer>")[0].strip()


def batched_generation(dataset, tokenizer, llm, sampling, build_prompt_fn, batch_size=BATCH_SZ):
    all_preds = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        prompts = [build_prompt_fn(input_train) for input_train in batch['input_train']]
        outputs = llm.generate(prompts, sampling)
        all_preds.extend([extract_xml_answer(out.outputs[0].text) for out in outputs])
    return all_preds


if __name__ == "__main__":
    args = parse_args()

    df = read_csv(args.input_file)
    dataset = prepare_dataset(df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,
                                            padding_side="right",
                                            use_fast=False)

    def build_prompt(input_train):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT_REASONING},
            {"role": "user",   "content": input_train},
        ]
        return tokenizer.apply_chat_template(prompt,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=True)

    llm = LLM(model             = args.model_dir,
            dtype             = "bfloat16",
            gpu_memory_utilization = 0.90,     # optional
            trust_remote_code = True)          # allow custom code if needed

    sampling = SamplingParams(max_tokens=MAX_GEN,
                            temperature=0.5,   # greedy → deterministic answers
                            top_p=1.0)

    predicted_labels = batched_generation(dataset, tokenizer, llm, sampling, build_prompt)

    print("\n--- First 5 Predictions ---")
    for i in range(min(5, len(dataset))):
        print(f"[Example {i+1}]")
        print(f"Story        : {dataset[i]['input_train']}")
        print(f"True Answer  : {dataset[i]['answer']}")
        print(f"Prediction   : {predicted_labels[i]}")
        print("-" * 60)

    save_dataset_to_csv(df, dataset, predicted_labels, args.output_dir)

    print("Successfully Saved!")