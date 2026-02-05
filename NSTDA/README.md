# Thai Text-to-Gloss Research Project

This repository contains the research pipeline for Thai Text-to-Gloss translation, divided into four main modules.

## 📂 Project Structure

### 1. `transfomer_model_pipeline`
Contains scripts for creating and training encoder-decoder baselines such as **mBart**, **mt5**, and **nllb**.

### 2. `sft_llm`
The first step of the LLM pipeline. Focuses on **Supervised Fine-Tuning (SFT)** of Large Language Models for the text-to-gloss task.

### 3. `grpo_llm`
The second step. Focuses on further fine-tuning the SFT model using **Reinforcement Learning** with **GRPO (Group Relative Policy Optimization)** and various reward functions.

### 4. `agentic_llm`
**Status: Unfinished / Experimental**
A future-proof module designed for agentic workflows. (Note: Currently incomplete due to dependency versions).

---

## 📊 Dataset Preparation

Every pipeline expects the dataset to be prepared in a specific format (`DatasetDict`) before training.

### Datset Structure

```python
DatasetDict({
    sft_train: Dataset({
        features: ['index', 'messages', 'text_raw', 'text_sign'],
        num_rows: 
    })
    sft_eval: Dataset({
        features: ['index', 'messages', 'text_raw', 'text_sign'],
        num_rows: 
    })
    grpo_train: Dataset({
        features: ['index', 'messages', 'text_raw', 'text_sign'],
        num_rows: 
    })
    grpo_eval: Dataset({
        features: ['index', 'messages', 'text_raw', 'text_sign'],
        num_rows: 
    })
    test: Dataset({
        features: ['index', 'messages', 'text_raw', 'text_sign'],
        num_rows: 
    })
})
```

### Data Format Example

The `messages` field should follow this JSON structure, including the specific system prompt for Thai-to-Gloss rules:

```json
{
  "index": 880,
  "messages": [
    {
      "content": "You are an expert in translating from Thai to Thai-gloss following these rules:\n\nRules for Thai-to-Thai Gloss Translation:\n1.Base Word Identification: Identify the smallest meaningful units (root words or isolated signs/single handshapes) based primarily on handshape, without interpreting their meaning based on the Thai linguistic context yet.\n2.Uncertainty Marker: Use phrases within brackets `[]` when you are expressed about rare occasion.\n3.Numbers/Counting: Handle numbers and counting appropriately.\n4.Spelling (Finger-spelling): Indicate finger-spelled words with `#spelling` and individual letters with `#s` (e.g., from TNN to `#s T|#s N|#s N`, from โรคฮีตสโตรก to `ชื่อ|สะกดนิ้วมือ|ชื่อ|#s ฮ|#s -ี|#s ต|#s ส|#s โ-|#s ต|#s ร|#s ก|`).\n5.Directional Handshapes: For the same word with different directional handshapes, mark with `#direction` and `#d` (e.g., #d อ่าวไทย, #d อันดามัน).\n6.Compound/Continuous Handshapes: Use `#compound` and `#c` for continuous or unclear handshape sequences (e.g., `#c 4-17` or `#c 10+3`).\n7.Unclear Compound Spelling: If a handshape represents spelling but is unclear or a compound handshape without clear segmentation, use `#compound` and `#c` (e.g., `#c #s ก+รม(กรม)`).\n\nTranslate the following Thai input into Thai-gloss following the rules with thai word only.\nOutput **Thai only**, inside `<answer> … </answer>` tags.",
      "role": "system"
    },
    {
      "content": "กรุงเทพมหานครและปริมณฑล มีเมฆบางส่วนกับมีหมอกบางในตอนเช้า อุณหภูมิต่ำสุด 24-26 องศาเซลเซียส",
      "role": "user"
    },
    {
      "content": "<answer>กรุงเทพมหานครและปริมณฑล|กับ_ที่|มี|เมฆ|เช้า|มี|หมอก|มี|หนาว|อุณหภูมิต่ำ_อุณหภูมิลดลง|#c 24-26</answer>",
      "role": "assistant"
    }
  ],
  "text_raw": "กรุงเทพมหานครและปริมณฑล มีเมฆบางส่วนกับมีหมอกบางในตอนเช้า อุณหภูมิต่ำสุด 24-26 องศาเซลเซียส",
  "text_sign": "กรุงเทพมหานครและปริมณฑล|กับ_ที่|มี|เมฆ|เช้า|มี|หมอก|มี|หนาว|อุณหภูมิต่ำ_อุณหภูมิลดลง|#c 24-26",
}
```
