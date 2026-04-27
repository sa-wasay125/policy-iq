import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 5))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def ensure_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)


def save_uploaded_file(file_bytes: bytes, filename: str) -> str:
    ensure_dirs()
    safe_name = Path(filename).name
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(save_path, "wb") as f:
        f.write(file_bytes)
    print(f"[✔] File saved: {save_path}")
    return save_path


def get_upload_path(filename: str) -> str:
    return os.path.join(UPLOAD_DIR, Path(filename).name)


def file_exists(filename: str) -> bool:
    return os.path.exists(get_upload_path(filename))


if __name__ == "__main__":
    ensure_dirs()
    print(f"[✔] Upload dir     : {UPLOAD_DIR}")
    print(f"[✔] FAISS index dir: {FAISS_INDEX_DIR}")
    print(f"[✔] Chunk size     : {CHUNK_SIZE}")
    print(f"[✔] Top-K          : {TOP_K}")
    print(f"[✔] Confidence     : {CONFIDENCE_THRESHOLD}")
    print(f"[✔] API key loaded : {'Yes' if GOOGLE_API_KEY else 'NO - CHECK .env'}")