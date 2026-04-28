import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")

# RAG settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 5))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))

# API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def ensure_dirs():
    """Create required directories if they don't exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)


def save_uploaded_file(file_bytes: bytes, filename: str) -> str:
    """
    Save raw bytes of an uploaded file to the uploads directory.
    Returns the full path where the file was saved.
    """
    ensure_dirs()
    safe_name = Path(filename).name
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    print(f"[✔] File saved: {save_path}")
    return save_path


def get_upload_path(filename: str) -> str:
    """Return the full path for a given filename in uploads dir."""
    return os.path.join(UPLOAD_DIR, Path(filename).name)


def file_exists(filename: str) -> bool:
    """Check if a file already exists in uploads dir."""
    return os.path.exists(get_upload_path(filename))


if __name__ == "__main__":
    ensure_dirs()
    print(f"[✔] Upload dir     : {UPLOAD_DIR}")
    print(f"[✔] FAISS index dir: {FAISS_INDEX_DIR}")
    print(f"[✔] Chunk size     : {CHUNK_SIZE}")
    print(f"[✔] Top-K          : {TOP_K}")
    print(f"[✔] Confidence     : {CONFIDENCE_THRESHOLD}")
    print(f"[✔] API key loaded : {'Yes' if GOOGLE_API_KEY else 'NO — CHECK .env'}")