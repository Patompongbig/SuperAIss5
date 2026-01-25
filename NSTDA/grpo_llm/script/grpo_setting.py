import re
import pandas as pd
import unicodedata
from rouge_score import rouge_scorer
from jiwer import wer


system_prompt = """You are an expert in translating from Thai to Thai-gloss following these rules:

Rules for Thai-to-Thai Gloss Translation:
1.Base Word Identification: Identify the smallest meaningful units (root words or isolated signs/single handshapes) based primarily on handshape, without interpreting their meaning based on the Thai linguistic context yet.
2.Uncertainty Marker: Use phrases within brackets `[]` when you are expressed about rare occasion.
3.Numbers/Counting: Handle numbers and counting appropriately.
4.Spelling (Finger-spelling): Indicate finger-spelled words with `#spelling` and individual letters with `#s` (e.g., from TNN to `#s T|#s N|#s N`, from โรคฮีตสโตรก to `ชื่อ|สะกดนิ้วมือ|ชื่อ|#s ฮ|#s -ี|#s ต|#s ส|#s โ-|#s ต|#s ร|#s ก|`).
5.Directional Handshapes: For the same word with different directional handshapes, mark with `#direction` and `#d` (e.g., #d อ่าวไทย, #d อันดามัน).
6.Compound/Continuous Handshapes: Use `#compound` and `#c` for continuous or unclear handshape sequences (e.g., `#c 4-17` or `#c 10+3`).
7.Unclear Compound Spelling: If a handshape represents spelling but is unclear or a compound handshape without clear segmentation, use `#compound` and `#c` (e.g., `#c #s ก+รม(กรม)`).

Translate the following Thai input into Thai-gloss following the rules with thai word only.
Output format:
<think>
[Step-by-step reasoning about word segmentation and rule application]
</think>
<answer>
[Final Thai Gloss only]
</answer>"""

#Dataset Create
def get_agnews_questions_for_grpo(dataset, tokenizer):
    dataset = dataset.map(
        lambda x: preprocess_for_grpo(x, tokenizer),
        remove_columns=[c for c in dataset["test"].column_names if c != "text_sign"],
    )
    return dataset

def preprocess_for_grpo(example, tokenizer):
    prompt = tokenizer.apply_chat_template(
        example["messages"][:-1],
        tokenize=False,
        add_generation_prompt=True
    )

    completion = example["messages"][-1]["content"] + tokenizer.eos_token
    return {
        "prompt": prompt,
        "completion": completion,
        "text_sign": example["text_sign"]
    }

def test_dataset_grpo(dataset, tokenizer):
    dataset = dataset.map(
        lambda x: {
            "prompt": tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x["text_raw"]},
                ],
                tokenize=False,
                add_generation_prompt=True
            )
        }
    )
    return dataset



# Utility functions
def normalize_thai(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    # 1) Fix decomposed "◌ํ + า" to "ำ"
    text = re.sub("\u0E4D\u0E32", "\u0E33", text)
    # 2) Fix reversed order "า + ◌ํ" (common from OCR or LLM)
    text = re.sub("\u0E32\u0E4D", "\u0E33", text)
    # 3) Remove duplicated "◌ํ◌ํ"
    text = re.sub("\u0E4D{2,}", "\u0E4D", text)
    # 4) Normalize again
    return unicodedata.normalize("NFC", text)



#Text process functions
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_xml_think(text: str) -> str:
    answer = text.split("<think>")[-1]
    answer = answer.split("</think>")[0]
    return answer.strip()


I = 0
def has_valid_format(text: str) -> bool:
    global I
    if I < 2:
        print("Checking format:", text)
        I += 1

    return bool(re.search(r"<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>", text))


def compute_jaccard_score(answer: str, prediction: str) -> float:
    answer_set = set(answer.split("|"))
    prediction_set = set(prediction.split("|"))

    jaccard = len(answer_set & prediction_set) / len(answer_set | prediction_set)

    return round(jaccard, 2)

rouge_score = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
def compute_rouge_L_score(answer: str, prediction: str) -> float:
    answer = " ".join(answer.split("|"))
    prediction = " ".join(prediction.split("|"))

    scores = rouge_score.score(answer, prediction)['rougeL'].fmeasure

    return round(scores, 2)

def compute_wer_score(answer: str, prediction: str) -> float:
    answer = " ".join(answer.split("|"))
    prediction = " ".join(prediction.split("|"))

    wer_score = wer(answer, prediction)

    return round(1 - wer_score, 2)

    
def compute_oov_score(prediction: str, vocab: list) -> float:
    prediction_list = prediction.split("|")
    oov_count = sum(1 for token in prediction_list if token not in vocab)

    if oov_count == 0:
        return 1.0
    elif oov_count <= 2:
        return 0.3
    elif oov_count <= 5:
        return 0.1
    else:
        return 0.0





#Reward functions

#Completions structure
# [
#   [ {"role": "assistant", "content": "... model output ..."} ],
#   [ {"role": "assistant", "content": "... model output ..."} ],
#   ...
# ]

# Format think and answer reward
def think_quality_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        if not has_valid_format(c):
            rewards.append(0.0)
            continue

        think = extract_xml_think(c)
        if "|" in think:
            rewards.append(-0.5)
            continue
        if len(think) >= 50 and len(think) <= 2500:
            rewards.append(0.05)
        else:
            rewards.append(0.0)
    return rewards

def repetition_penalty_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        if not has_valid_format(c):
            rewards.append(0.0)
            continue

        answer = extract_xml_answer(c)
        tokens = answer.split("|")

        unique_ratio = len(set(tokens)) / max(len(tokens), 1)

        if unique_ratio > 0.7:
            rewards.append(0.1)
        elif unique_ratio > 0.5:
            rewards.append(0.0)
        else:
            rewards.append(-1.0)
    return rewards


# Using Jaccard for Lexical (aligned gloss tokens)
# Using Rouge-L for Semantic (gloss meaning and ordering)
def lexical_and_semantic_reward_function(completions, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    gold_list = kwargs["text_sign"]

    reward = []
    for response, gold in zip(responses, gold_list):
        if not has_valid_format(response):
            reward.append(0.0)
            continue

        gold = normalize_thai(gold)
        pred = normalize_thai(extract_xml_answer(response))
        lex = compute_rouge_L_score(gold, pred)
        sem = compute_wer_score(gold, pred)
        reward.append((lex + sem) / 2)

    return reward



# OOV-based reward
vocab = pd.read_excel("/project/lt-user/data/text-to-gloss/vocab.xlsx")["token"].tolist()
VOCAB = [normalize_thai(token) for token in vocab]
def oov_error_reward_function(completions, **kwargs) -> list[float]:
    gold_list = kwargs["text_sign"]

    reward = []
    for response, gold in zip(completions, gold_list):
        if not has_valid_format(response):
            reward.append(0.0)
            continue

        answer = normalize_thai(extract_xml_answer(response))
        reward.append(compute_oov_score(answer, VOCAB))

    return reward
