# FOR DATASET
system_prompt = """
You are an expert in translating from Thai to Thai-gloss following these rules:

**Rules for Thai-to-Thai Gloss Translation:**
1.  **Base Word Identification:** Identify the smallest meaningful units (root words or isolated signs/single handshapes) based primarily on handshape, without interpreting their meaning based on the Thai linguistic context yet.
2.  **Uncertainty Marker:** Use phrases within brackets `[]` when you are expressed about rare occasion.
3.  **Numbers/Counting:** Handle numbers and counting appropriately.
4.  **Spelling (Finger-spelling):** Indicate finger-spelled words with `#spelling` and individual letters with `#s` (e.g., from TNN to `#s T|#s N|#s N`, from โรคฮีตสโตรก to `ชื่อ|สะกดนิ้วมือ|ชื่อ|#s ฮ|#s -ี|#s ต|#s ส|#s โ-|#s ต|#s ร|#s ก|`).
5.  **Directional Handshapes:** For the same word with different directional handshapes, mark with `#direction` and `#d` (e.g., #d อ่าวไทย, #d อันดามัน).
6.  **Compound/Continuous Handshapes:** Use `#compound` and `#c` for continuous or unclear handshape sequences (e.g., `#c 4-17` or `#c 10+3`).
7.  **Unclear Compound Spelling:** If a handshape represents spelling but is unclear or a compound handshape without clear segmentation, use `#compound` and `#c` (e.g., `#c #s ก+รม(กรม)`).

---
**Examples:**

### Example 1:
Input: ก็ต้องดูแลรักษาสุขภาพ หลีกเลี่ยงการทํางาน หรือว่าทํากิจกรรมกลางแจ้งเป็นระยะเวลานาน ๆ ป้องกันการป่วยเป็นโรคลมแดด หรือว่าโรคฮีตสโตรกนะคะ
Output: <answer> ดูแล_รักษา|สุขภาพ|ทํางาน|ข้างนอก|แสงอาทิตย์|เวลา|นาน_ช้า|ระวัง|แสงอาทิตย์|เป็นลม|ชื่อ|สะกดนิ้วมือ|ชื่อ|#s ฮ|#s -ี|#s ต|#s ส|#s โ-|#s ต|#s ร|#s ก|กระทบ|ระวัง </answer>

### Example 2:
Input: ทะเลทั้งฝั่งอันดามันและฝั่งอ่าวไทย คลื่นสูงประมาณ 1 เมตร บริเวณที่มีฝนฟ้าคะนอง คลื่นสูงได้มากกว่า 2 เมตร
Output: <answer> ทั้งสองฝั่ง|ทะเล|สูง|1|#s ม|สมมติ_ถ้า|ฝนตก|ทะเล|สูง|2|#s ม|ขึ้นไป </answer>

---
**Your Task:**

Translate the following Thai input into Thai-gloss following the rules with thai word only.  
Output **Thai only**, inside `<answer> … </answer>` tags.
"""


def get_agnews_questions_for_sft(dataset, tokenizer):
    dataset = dataset.map(
        lambda x: {
            "prompt": tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x["text_raw"]},
                ],
                tokenize=False,  
                add_generation_prompt=True  
            ),
            "completion": "<answer> " + x["text_sign"] + " </answer>" + tokenizer.eos_token,
        }
    )
    # text, gloss_sequence, text_raw, text_sign
    return dataset

