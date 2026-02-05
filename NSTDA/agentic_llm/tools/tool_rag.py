from typing import TypedDict, List
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss
import pickle

from script.utils import normalize_thai, VOCAB

# RAG finding similar sentences
MODEL_PATH = "/project/lt-user/model/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RAG_DIR = "/project/lt-user/agentic_llm/rag_database"
INDEX_PATH = f"{RAG_DIR}/faiss_index.faiss"
RECORDS_PATH = f"{RAG_DIR}/rag_records.pkl"

_RAG_CACHE = {
    "index": None,
    "records": None,
    "model": None
}

def _load_resource():
    global _RAG_CACHE

    # Load Model
    if _RAG_CACHE["model"] is None:
        _RAG_CACHE["model"] = SentenceTransformer(MODEL_PATH, device="cpu")

    # Load FAISS index
    if _RAG_CACHE["index"] is None:
        _RAG_CACHE["index"] = faiss.read_index(INDEX_PATH)
    
    # Load RAG records
    if _RAG_CACHE["records"] is None:
        with open(RECORDS_PATH, "rb") as f:
            _RAG_CACHE["records"] = pickle.load(f)

    return _RAG_CACHE

def retrieve_similar_thai_gloss_sentences(thai_sentence: str) -> list:
    """
    Retrieve semantically similar Thai sentences and their corresponding Thai glosses.

    This tool searches a database of Thai sentences and returns full-sentence
    Thai–Thai gloss example pairs that are semantically similar to the input sentence.
    The results can be used as references to help generate consistent Thai gloss output.

    Args:
        thai_sentence: A full Thai sentence to search for similar examples.

    Returns:
        A list of up to 2 examples. Each example contains:
        - thai: A Thai sentence similar in meaning
        - gloss: The corresponding Thai gloss sentence
    """
    resource = _load_resource()
    index = resource["index"]
    records = resource["records"]
    model = resource["model"]

    # Search
    query_embedding = model.encode([thai_sentence], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, k=2)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(records):
            record = records[idx]
            results.append({
                "thai": record["thai"],
                "gloss": record["gloss"],
                "score": float(score)
            })
    
    return results


# Check gloss in dict
def get_oov_gloss_tokens(gloss_sentence: str) -> list:
    """
    Returns gloss tokens that are NOT present in the vocabulary.

    Args:
        gloss_sentence: A Thai gloss sentence separated by '|'.

    Returns:
        A list of out-of-vocabulary gloss tokens.
        Empty list means all tokens are valid.
    """
    gloss_sentence = normalize_thai(gloss_sentence)
    tokens = gloss_sentence.split("|")
    return [token for token in tokens if token not in VOCAB]
