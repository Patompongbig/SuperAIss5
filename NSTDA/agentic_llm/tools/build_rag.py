import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

MODEL_PATH = "/project/lt-user/model/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = "/project/lt-user/data/text-to-gloss"
RAG_DIR = "/project/lt-user/agentic_llm/rag_database"
os.makedirs(RAG_DIR, exist_ok=True)


def extract_sentence_pairs(dataset):
    records = []
    for row in dataset:
        records.append({
            "thai": row["text_raw"].strip(),
            "gloss": row["text_sign"].strip()
        })
    return records


def build_and_save():
    # load dataset
    dataset = load_dataset(DATA_PATH, cache_dir=None)
    
    # extract sentence
    rag_records = (
        extract_sentence_pairs(dataset["sft_train"]) +
        extract_sentence_pairs(dataset["sft_eval"])
    )
    print(f"Total records: {len(rag_records)}")

    # Encode
    model = SentenceTransformer(MODEL_PATH, device="cuda")
    sentences = [record["thai"] for record in rag_records]
    embeddings = model.encode(
        sentences,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Build FAISS index
    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))

    # Save RAG database
    faiss.write_index(index, os.path.join(RAG_DIR, "faiss_index.faiss"))

    with open(os.path.join(RAG_DIR, "rag_records.pkl"), "wb") as f:
        pickle.dump(rag_records, f)
    print("FAISS index saved.")

if __name__ == "__main__":
    build_and_save()

