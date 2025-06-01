#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSV to TRL Dataset Converter

This script reads a CSV file and converts it to a format compatible with TRL training
for supervised fine-tuning (SFT), SFT with LoRA, and General Reward-Based Policy Optimization (GRPO).

Usage:
    python csv_to_trl_dataset.py --input_file path/to/test.csv --output_dir path/to/output --mode [sft|sft_lora|grpo]
"""

import os
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
import random
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# Logging time stamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Recieve Argument for setting up the dataset
def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV data to TRL dataset format")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed dataset")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

# Read the CSV file using Pandas
def read_csv(file_path):
    """Read the CSV file and return a pandas DataFrame."""
    logger.info(f"Reading CSV file from {file_path}")
    df = pd.read_csv(file_path)
    return df


# Loop and create dictionary fo reach row
def create_example(row):
    input_train = f"Description: {row['story_masked']}\n"
    answer = str(row['answers'])
    return {
        "input_train": input_train,
        "answer": answer,
        "ner_map": row.get('story_ner_map', {})  # Optional: keep mapping if needed later
    }

# Save to target directory
def save_dataset(dataset, output_dir):
    """Save the dataset to disk."""
    logger.info(f"Saving dataset to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    dataset['train'].save_to_disk(os.path.join(output_dir, 'train'))
    
    return output_dir

def apply_ner_tagging_to_text(text_to_tag, model, tokenizer):

    prompt = f"""
        กรุณาวิเคราะห์ข้อความภาษาไทยต่อไปนี้อย่างละเอียด และระบุชื่อบุคคลทั้งหมด รวมทั้งชื่อสั้นๆ หรือชื่อที่ไม่มีคำนำหน้าด้วย เช่น 'แดง', 'ดำ', 'จรา', 'บัว'

        แทนที่ทุกชื่อบุคคลด้วยรูปแบบ {{p1}}, {{p2}}, ... ตามลำดับการปรากฏในข้อความ

        **แนวทางในการพิจารณาชื่อบุคคล:**

        1. **ตำแหน่งในประโยค**:
        - ชื่อบุคคลมักอยู่ในตำแหน่งประธานหรือกรรม
        - เช่น: "สมชายไปตลาด" → "สมชาย" เป็นประธาน
                  "พบกับจราในห้องเรียน" → "จรา" เป็นกรรม

        2. **บริบทของคำกริยา**:
        - พิจารณาคำกริยา เช่น “แพ้”, “ยิง”, “เดินทาง”, “ชนะ”, “กล่าว”, “พูด” ฯลฯ ซึ่งมักตามด้วยหรือมาก่อนชื่อบุคคล
        - เช่น: "เคี้ยนแพ้หมากรุกเลยยิงจรา" → ทั้ง "เคี้ยน" และ "จรา" เป็นชื่อบุคคล

        3. **ชื่อที่สั้นมากหรือไม่มีคำนำหน้า**:
        - ให้ระบุชื่อที่สั้น เช่น “แดง”, “จรา”, “ต๋อย”, “บัว” ว่าเป็นชื่อบุคคลถ้าอยู่ในตำแหน่งที่ตรงกับกฎข้างต้น

        4. หากมีชื่อจริงตามด้วยนามสกุล ให้นับเป็นบุคคลเดียว เช่น "นายแดง ใจดี" → {{p1}}

        5.  ระบุเฉพาะชื่อบุคคลที่มีอยู่จริงเท่านั้น ไม่รวมถึงชื่อตัวละครในนิยาย, นิทาน, หรือเรื่องแต่งอื่นๆ และไม่รวมถึงคำที่เป็นสำนวนหรือความหมายเชิงเปรียบเทียบ
        **อย่าข้ามชื่อบุคคลเพียงเพราะสั้นหรือไม่มีคำนำหน้า**

        6. **ชื่อบุคคลมักอยู่หน้าคำกริยา** เช่น "ยิง", "ลอบ", "ใช้", "วางแผน", "บุกรุก", "ดัดแปลง", "ขโมย", "ทำร้าย", "ฆ่า", "ขับรถ", "เดินทาง", "พูดคุย", "พบปะ", "ติดต่อ", "บอกให้" และอื่น ๆ

        ส่งคืนเฉพาะข้อความที่แทนชื่อบุคคลแล้วเท่านั้น โดยไม่มีคำอธิบายหรือเนื้อหาอื่น
    """
    try:
        full_prompt = prompt + "\n\nข้อความ: " + text_to_tag.strip()
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the part *after* the prompt
        response = decoded[len(full_prompt):].strip()
        return response.replace(" ", "")
    except Exception as e:
        logger.error(f"Error during model inference for text '{text_to_tag[:50]}...': {e}")
    return text_to_tag  # Fallback
    
def extract_ner_mapping(original_text, ner_text):
    import re
    
    # Strip both input texts if provided
    original_text = original_text.strip() if original_text else ""
    ner_text = ner_text.strip() if ner_text else ""
    
    # Create a mapping dictionary
    mapping = {}
    
    # Find all people tokens in the NER text
    token_pat = re.compile(r"\{p(\d+)\}")
    tokens = list(token_pat.finditer(ner_text))
    
    # If no tokens found, return empty mapping
    if not tokens:
        return [ner_text, mapping]
    
    # Get all segments between tokens
    segments = []
    last_end = 0
    
    for t in tokens:
        # Get text before the token
        segments.append(ner_text[last_end:t.start()])
        # Store the token itself
        segments.append(t.group())
        last_end = t.end()
    
    # Add the final segment after the last token
    segments.append(ner_text[last_end:])
    
    # Reconstruct the original text by replacing tokens with actual names
    current_pos = 0
    for i in range(0, len(segments), 2):
        if i + 1 >= len(segments):
            break
            
        segment = segments[i].strip()  # Strip the segment
        token = segments[i+1]
        
        # Find where this segment exists in the original text
        if segment:
            # Try to find the stripped segment in the original text
            search_from = current_pos
            found = False
            
            # Look for the segment, trying both stripped and unstripped versions
            segment_pos = original_text.find(segment, search_from)
            if segment_pos == -1:
                # Try finding unstripped version
                segment_pos = original_text.find(segments[i], search_from)
                
            if segment_pos != -1:
                # If found either way, update position accordingly
                current_pos = segment_pos + len(segment)
                found = True
            
            if not found:
                segment_pos = current_pos  # Use current position if not found
        else:
            segment_pos = current_pos
            
        # Find where the next segment begins
        if i+2 < len(segments):
            next_segment = segments[i+2].strip()  # Strip the next segment
            
            # Try both stripped and unstripped versions
            next_pos = original_text.find(next_segment, current_pos)
            if next_pos == -1:
                next_pos = original_text.find(segments[i+2], current_pos)
                
            if next_pos == -1:
                next_pos = len(original_text)
        else:
            next_pos = len(original_text)
            
        # Extract the name from the original text
        name = original_text[current_pos:next_pos]
        
        # Store in mapping
        token_id = re.search(r"p(\d+)", token).group(1)
        mapping[f"p{token_id}"] = name.strip()
        
        current_pos = next_pos
    
    return [ner_text, mapping]
    
def nertag_preprocess_df(df, story_column_name, model, tokenizer):
    masked_stories = []
    ner_mappings = []

    logger.info(f"Starting NER tagging and mapping for column '{story_column_name}'...")
    for index, row in df.iterrows():
        original_text = str(row[story_column_name])
        logger.info(f"Tagging row {index + 1}/{len(df)}: '{original_text[:50]}...'")

        # Get masked text using your model
        ner_text = apply_ner_tagging_to_text(df, story_column_name, model, tokenizer)

        # Get mapping of placeholders to original names
        masked_text, name_mapping = extract_ner_mapping(original_text, ner_text)

        masked_stories.append(masked_text)
        ner_mappings.append(name_mapping)

    df[f'{story_column_name}_masked'] = masked_stories
    df[f'{story_column_name}_ner_map'] = ner_mappings

    logger.info("NER tagging and mapping complete.")
    return df


# Pipeline to run the script
def main():
    args = parse_args()
    model_path = '/project/ai901504-ai0004/500101-Boss/superai-llm-reasoning/SuperAI_LLM_FineTune_2025/interesting_model/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    # Read the CSV file
    df = read_csv(args.input_file)

    df = nertag_preprocess_df(df, args.story_column_name, model, tokenizer)
    
    # Save the dataset
    output_path = save_dataset(df, args.output_dir)
    logger.info(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main() 