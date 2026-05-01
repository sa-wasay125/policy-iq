import os
import pickle
import time
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

# Local model — downloads once (~90MB), runs offline after that

import faiss
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# LOAD ENV VARIABLES
# ──────────────────────────────────────────────
load_dotenv()
_model = SentenceTransformer("all-MiniLM-L6-v2")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
TOP_K = int(os.getenv("TOP_K", 5))



INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "index.faiss")
METADATA_FILE = os.path.join(FAISS_INDEX_DIR, "metadata.pkl")


# ──────────────────────────────────────────────
# EMBED SINGLE TEXT
# ──────────────────────────────────────────────
def embed_text(text: str) -> List[float]:
    return _model.encode(text, normalize_embeddings=True).tolist()


# ──────────────────────────────────────────────
# EMBED CHUNKS
# ──────────────────────────────────────────────
def embed_chunks(chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict]]:
    embeddings = []
    metadata = []

    print(f"[✔] Embedding {len(chunks)} chunks...")

    for chunk in tqdm(chunks):
        try:
            vector = embed_text(chunk["text"])
            embeddings.append(vector)

            metadata.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "chunk_on_page": chunk["chunk_on_page"]
            })

            time.sleep(0.05)

        except Exception as e:
            print(f"[!] Chunk {chunk['chunk_index']} failed: {e}")

    if len(embeddings) == 0:
        raise ValueError("❌ No embeddings created. Check API key or model.")

    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"[✔] Embeddings shape: {embeddings_array.shape}")

    return embeddings_array, metadata


# ──────────────────────────────────────────────
# BUILD FAISS INDEX
# ──────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray):

    if embeddings.size == 0:
        raise ValueError("❌ Empty embeddings")

    dimension = embeddings.shape[1]

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"[✔] FAISS index built ({index.ntotal} vectors)")

    return index


# ──────────────────────────────────────────────
# SAVE INDEX
# ──────────────────────────────────────────────
def save_index(index, metadata):
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print("[✔] Index + metadata saved")


# ──────────────────────────────────────────────
# LOAD INDEX
# ──────────────────────────────────────────────
def load_index():
    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


# ──────────────────────────────────────────────
# SEARCH
# ──────────────────────────────────────────────
def search_index(query: str, index, metadata, top_k=TOP_K):

    query_vector = embed_text(query)
    query_array = np.array([query_vector], dtype=np.float32)

    faiss.normalize_L2(query_array)

    scores, indices = index.search(query_array, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        result = metadata[idx].copy()
        result["score"] = float(score)
        results.append(result)

    return results


# ──────────────────────────────────────────────
# FULL PIPELINE
# ──────────────────────────────────────────────
def process_and_index(chunks):
    embeddings, metadata = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    save_index(index, metadata)

    print("\n[✔] DONE — Ready for queries")


# ──────────────────────────────────────────────
# TEST RUN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from rag_pipeline import process_pdf

    TEST_PDF = "docs/sample_policy.pdf"

    print("\nSTEP 1 — PDF Processing")
    chunks = process_pdf(TEST_PDF)

    print("\nSTEP 2 — Embedding")
    process_and_index(chunks)

    print("\nSTEP 3 — Search Test")

    index, metadata = load_index()

    results = search_index("claim limit", index, metadata)

    for r in results:
        print("\n---")
        print("Page:", r["page_number"])
        print("Score:", r["score"])
        print("Text:", r["text"][:100])