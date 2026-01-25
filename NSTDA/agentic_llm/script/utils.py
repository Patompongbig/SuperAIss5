# OOV Checking reward
import unicodedata
import re
import pandas as pd

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

vocab = pd.read_excel("/project/lt-user/data/text-to-gloss/vocab.xlsx")["token"].tolist()
VOCAB = [normalize_thai(token) for token in vocab]