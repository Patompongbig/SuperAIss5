import re
from datasets import load_dataset, Dataset
import ast
import json
import logging

# Define Prompt for generate responds as a RHLF
SYSTEM_PROMPT_REASONING = """
คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์คดีและกฎหมาย หน้าที่ของคุณคืออ่านเรื่องราวที่ได้รับมาอย่างละเอียด วิเคราะห์พฤติการณ์ของแต่ละบุคคลที่ปรากฏในเรื่อง และพิจารณาความผิดของพวกเขาตาม **ประเภทความผิด (Class)** และ **กฎพิเศษ** ที่กำหนดให้เท่านั้น

**ประเภทความผิด (Class Categories):**
class 1: ไม่มีความผิดใดๆ เลย (No guilt)
class 2: ลักทรัพย์หรือชิงทรัพย์ (Theft or Robbery)
class 3: การกระทำใดๆ ด้วยความประมาท, ละเลย หรือ ไม่ได้ตั้งใจซึ่งก่อให้เกิดความเสียหายต่อชีวิตและทรัพย์สิน (Negligence, omission, or unintentional acts causing damage to life and property)
class 4: บุกรุกเข้าเขตหวงห้ามหรือนอกเวลาทำการ (Trespassing into restricted areas or outside operating hours)
class 5: กระทำการใดๆด้วยเจตนาที่ส่งผลให้เกิดความเสียหายทางด้านร่างกาย, จิตใจหรือทรัพย์สิน (Intentional acts causing physical, psychological, or property damage)
class 6: ฉ้อโกงประชาชน, ปลอมแปลงเอกสาร หรือปลอมแปลงสิ่งของต่างๆเพื่อนำมาใช้ในเชิงการค้า (Public fraud, document forgery, or object forgery for commercial use)
class 7: ทำให้ผู้อื่นเสียชีวิตด้วยวิธีใดๆก็ตาม (Causing another person's death by any means)

**กฎพิเศษสำหรับการพิจารณาความผิด:**
1.  การทำให้ผู้อื่นเสียชีวิต (class 7) ด้วยความตั้งใจ, ทะเลาะวิวาท หรือวิธีที่เหนือธรรมชาติ จะถือเป็นการกระทำที่เจตนนา (class 5) หรือ การกระทำที่ประมาท (class 3) ร่วมด้วย
2.  การทำตนเองให้เสียชีวิตไม่ถือว่าเป็นกระทำการใช้ความรุนแรง (class 5) และไม่ถือว่าทำให้คนตาย (class 7)
3.  การโจรกรรม (class 2) หรือ การบุกรุกใดๆ (class 4) ที่มีการเสียหายของทรัพย์สินต้องมีการทำเจตนาที่ส่งผลให้เกิดความเสียหายทางด้านร่างกาย, จิตใจหรือทรัพย์สิน (class 5)
4.  หากไม่มีคำใดที่สื่อถึงความตาย (No Dead), ฆ่า (No Kill), หรือ วางแผนฆ่า (Planning for Murder) จะถือว่าไม่มีการเสียชีวิต (No class 7)
5.  หากไม่มีการกระทำความผิดใดๆเลยก็ตาม ผู้คนที่มีชื่อในเรื่องราวจะถือว่าเป็น ไม่มีความผิด (class 1)
6.  แม้ผู้กระทำความผิดเสียชีวิตในเรื่องราว ความผิดจะคงอยู่ตามเดิม (Ex. หากมิวทำการชิงทรัพย์แล้วมิวเสียชีวิต มิวจะยังคงมีความผิดในเรื่องของการชิงทรัพย์ (class 2))

**เงื่อนไขสำคัญสำหรับแต่ละเรื่อง:**
1.  แต่ละคดีอาจเกี่ยวข้องกับความผิดหลายประเภท โดยมีบุคคลอย่างน้อยหนึ่งคน และสูงสุดสามคน (ต้องเป็นบุคคลที่มีชื่อเท่านั้น)
2.  แต่ละบุคคลสามารถมีความผิดได้สูงสุด 3 ประเภท (สูงสุด 3 Class)
3.  ทุกๆคนที่เป็นคำตอบมีชื่อที่ไม่ซ้ำกัน
4.  ชื่อที่อยู่ในคำตอบทุกชื่อต้องเป็นชื่อที่มีอยู่ในเรื่องราว ห้ามมีชื่ออื่นๆนอกเหนือจากภายในเรื่องราว


1.  **ขั้นตอนความคิด (<think>):** แสดงลำดับการคิด เหตุผล และการนำ **ประเภทความผิด** และ **กฎพิเศษ** มาใช้ในการวิเคราะห์พฤติการณ์ของแต่ละบุคคลอย่างละเอียด อธิบายว่าทำไมถึงจัดบุคคลนั้นๆ เข้าข่าย Class ใด โดยอ้างอิงจากข้อเท็จจริงในเรื่องราวและกฎที่กำหนด
2.  **ผลลัพธ์สุดท้าย (<answer>):** สรุปผลการวิเคราะห์ความผิดของแต่ละบุคคลในรูปแบบ **JSON Array** ตามที่กำหนดไว้ โดยต้องมาจากขั้นตอนความคิดเท่านั้น และต้องไม่มีข้อความอื่นใดๆ เพิ่มเติมในส่วนนี้

<think>
[แสดงกระบวนการคิดอย่างละเอียด อ้างอิงกฎและประเภทความผิด]
</think>
<answer>
[{'ชื่อ': '[Class]'}, {'ชื่อ': '[Class,Class]'}]
</answer>

**ตัวอย่าง:**

ข้อความ: สมชายโกรธจัดจึงใช้มีดแทงสมหญิงจนเสียชีวิต
ผลลัพธ์:
<think>
จากเรื่อง สมชายมีพฤติการณ์ใช้มีดแทงสมหญิงซึ่งเป็นการกระทำด้วยเจตนาทำร้ายร่างกาย ทำให้เข้าข่ายประเภทความผิด Class 5 (เจตนาทำร้ายร่างกาย/ทรัพย์สิน) นอกจากนี้ การกระทำของสมชายส่งผลให้สมหญิงเสียชีวิตโดยตรง จึงเข้าข่ายประเภทความผิด Class 7 (ทำให้ผู้อื่นเสียชีวิต) และตามกฎพิเศษข้อ 1, การทำให้ผู้อื่นเสียชีวิตด้วยความตั้งใจถือเป็นการกระทำโดยเจตนา (Class 5) ร่วมด้วย ซึ่งสอดคล้องกับการวิเคราะห์เบื้องต้น ดังนั้นสมชายจึงมีความผิด Class 5 และ Class 7 สำหรับสมหญิง เป็นผู้ถูกกระทำจนเสียชีวิต ไม่ได้มีพฤติการณ์กระทำความผิดใดๆ ตามเรื่องราว จึงเข้าข่ายประเภทความผิด Class 1 (ไม่มีความผิด)
</think>
<answer>
[{'สมชาย': '[5,7]'}, {'สมหญิง': '[1]'}]
</answer>

ข้อความ: ขณะขับรถกลับบ้าน ยอดชายประมาททำให้รถชนเสาไฟฟ้าล้มทับนางสาวสวยเสียชีวิต
ผลลัพธ์:
<think>
จากเรื่อง ยอดชายมีพฤติการณ์ขับรถด้วยความประมาท ทำให้เกิดอุบัติเหตุรถชนเสาไฟฟ้า ซึ่งเป็นเหตุให้เสาไฟฟ้าล้มทับนางสาวสวยเสียชีวิต การขับรถประมาทก่อให้เกิดความเสียหายต่อชีวิตและทรัพย์สิน (เสาไฟฟ้า) จึงเข้าข่ายประเภทความผิด Class 3 (ประมาท) และการกระทำของยอดชายเป็นเหตุโดยตรงให้นางสาวสวยเสียชีวิต จึงเข้าข่ายประเภทความผิด Class 7 (ทำให้ผู้อื่นเสียชีวิต) ตามกฎพิเศษข้อ 1, การทำให้ผู้อื่นเสียชีวิตด้วยความประมาทถือเป็นการกระทำประมาท (Class 3) ร่วมด้วย ซึ่งสอดคล้องกับการวิเคราะห์เบื้องต้น ดังนั้นยอดชายจึงมีความผิด Class 3 และ Class 7 สำหรับนางสาวสวย เป็นผู้ประสบอุบัติเหตุจนเสียชีวิต ไม่ได้มีพฤติการณ์กระทำความผิดใดๆ จึงเข้าข่ายประเภทความผิด Class 1 (ไม่มีความผิด)
</think>
<answer>
[{'ยอดชาย': '[3,7]'}, {'นางสาวสวย': '[1]'}]
</answer>

ข้อความ: คนร้ายงัดบ้านของนายมีทรัพย์แล้วขโมยเงินสดไปจำนวนหนึ่ง
ผลลัพธ์:
<think>
จากเรื่อง คนร้ายมีพฤติการณ์ "งัดบ้าน" ซึ่งคือการบุกรุกสถานที่ (บ้านของนายมีทรัพย์) เข้าข่ายประเภทความผิด Class 4 (บุกรุก) นอกจากนี้ คนร้ายยัง "ขโมยเงินสด" ซึ่งเป็นการลักทรัพย์ เข้าข่ายประเภทความผิด Class 2 (ลักทรัพย์) ตามกฎพิเศษข้อ 3, การโจรกรรม (Class 2) หรือบุกรุก (Class 4) ที่มีการเสียหายของทรัพย์สิน (เงินสด/บ้านที่ถูกงัด) ต้องมีการทำเจตนาที่ส่งผลให้เกิดความเสียหาย (Class 5) ร่วมด้วย ดังนั้นคนร้ายจึงมีความผิด Class 2, Class 4, และ Class 5 สำหรับนายมีทรัพย์ เป็นผู้เสียหายจากการบุกรุกและการลักทรัพย์ ไม่ได้มีพฤติการณ์กระทำความผิดใดๆ จึงเข้าข่ายประเภทความผิด Class 1 (ไม่มีความผิด)
</think>
<answer>
[{'คนร้าย': '[2,4,5]'}, {'นายมีทรัพย์': '[1]'}]
</answer>

คุณต้องตอบกลับในรูปแบบที่กำหนดไว้เป๊ะๆ และมั่นใจว่าการวิเคราะห์และการระบุ Class เป็นไปตามกฎและเงื่อนไขทุกประการ
"""

# Remove the <answer> tags from the text
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


# Remove the <think> tags from the text
def extract_xml_think(text: str) -> str:
    answer = text.split("<think>")[-1]
    answer = answer.split("</think>")[0]
    return answer.strip()


# Remove the #### tags from the text
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# Receive the Dataset and then loop for responds
def process_agnews_dataset_for_reasoning(dataset):
  dataset = dataset.map(lambda x: { # type: ignore
      'prompt': [
          {'role': 'system', 'content': SYSTEM_PROMPT_REASONING},
          {'role': 'user', 'content': x['input_train']},
      ]
  }) # type: ignore
  return dataset # type: ignore

def safe_parse_dict_list(s):
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and all(isinstance(d, dict) for d in parsed):
            return parsed
    except Exception as e:
        print(f"[PARSE ERROR] {e} → {s}")
    return None


# Check the responds compare with the answer, if equal then rewards
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    rewards = []

    for resp_str, ans_str in zip(extracted_responses, answer):
        pred_list = safe_parse_dict_list(resp_str)
        ans_list  = safe_parse_dict_list(ans_str)

        if pred_list is None or ans_list is None:
            print("[SKIP] Bad format, skipping this pair.")
            continue

        pred_dict = {k: v for d in pred_list for k, v in d.items()}
        ans_dict  = {k: v for d in ans_list  for k, v in d.items()}

        i = 0  # incorrect
        j = 0  # correct

        for k, v in pred_dict.items():
            if k not in ans_dict:
                i += 1
            elif str(v) == str(ans_dict[k]):
                j += 1
            else:
                i += 1

        reward = j * 4.0 + i * -3.0
        rewards.append(reward)

    return rewards


# Check if responds is a digit or not
# def int_reward_func(completions, **kwargs) -> list[float]:
#     # Extract the response from the completion
#     responses = [completion[0]['content'] for completion in completions]
    
#     # Extract the answer from the completion
#     extracted_responses = [extract_xml_answer(r) for r in responses]
    
#     # Check if the extracted response is a digit
#     return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# CHheck if the responds is have a thinking process
def reasoning_length_reward(completions, **kwargs) -> list[float]:
    # Extract the response from the completion
    responses = [completion[0]['content'] for completion in completions]
    
    # Extract the think from the completion
    think_responses_len = [len(extract_xml_think(r)) for r in responses]
    
    # Check if the length of the think response is between 30 and 2000
    return [0.5 if (L >= 500 and L <= 3000) else -0.3 for L in think_responses_len]


# Check if the responds match to our expectation
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    
    pattern = r"<think>(?:.|\n)*?</think>\s*<answer>(?:.|\n)*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else -0.3 for match in matches]


# Check if the responds is have a thinking process by counting xml tags
def count_xml(text) -> float:
    count = 0.0
    
    if text.count("<think>\n") == 1:
        count += 0.125
    
    if text.count("\n</think>\n") == 1:
        count += 0.125
    
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    
    return count


# Check if the responds is have a thinking process by counting xml tags on each row
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]