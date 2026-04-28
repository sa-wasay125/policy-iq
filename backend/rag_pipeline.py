"""
rag_pipeline.py
---------------
Handles:
  - PDF loading (page by page with PyPDF)
  - Text cleaning
  - Chunking with metadata (page number, chunk index, source filename)

Day 2 scope: load_pdf() and chunk_documents() only.
Embeddings and FAISS will be added Day 3.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.utils.file_handler import CHUNK_SIZE, CHUNK_OVERLAP


# ──────────────────────────────────────────────
# STEP 1 — PDF LOADER
# ──────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a PDF and extract text page by page.

    Returns a list of page dicts:
    [
        {
            "page_number": 1,
            "text": "raw text from page 1...",
            "source": "sample_policy.pdf"
        },
        ...
    ]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    reader = PdfReader(str(path))
    pages = []

    print(f"[✔] Loading PDF: {path.name} ({len(reader.pages)} pages)")

    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text()

        if not raw_text or not raw_text.strip():
            print(f"    [!] Page {i + 1} is empty or unreadable — skipping")
            continue

        cleaned = clean_text(raw_text)

        pages.append({
            "page_number": i + 1,        # human-readable, 1-indexed
            "text": cleaned,
            "source": path.name
        })

    print(f"[✔] Extracted {len(pages)} non-empty pages")
    return pages


# ──────────────────────────────────────────────
# STEP 2 — TEXT CLEANER
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw PDF text:
    - Remove excessive whitespace and newlines
    - Remove non-printable characters
    - Normalize multiple spaces to single space
    - Keep punctuation intact (important for policy docs)
    """
    # Remove non-printable / control characters except newline and tab
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Replace multiple newlines with a single space
    text = re.sub(r"\n+", " ", text)

    # Replace multiple spaces/tabs with single space
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ──────────────────────────────────────────────
# STEP 3 — CHUNKER
# ──────────────────────────────────────────────

def chunk_documents(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split pages into overlapping chunks for embedding.

    Each chunk carries full metadata:
    {
        "text": "chunk content...",
        "source": "sample_policy.pdf",
        "page_number": 3,
        "chunk_index": 12,      ← global index across all chunks
        "chunk_on_page": 2      ← which chunk on that page (1-indexed)
    }

    Why overlap?
    A single sentence answering a query might sit at the
    boundary of two chunks. Overlap ensures it appears in
    at least one full chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # try paragraph → sentence → word
        length_function=len
    )

    all_chunks = []
    global_chunk_index = 0

    for page in pages:
        # Split this page's text into sub-chunks
        raw_chunks = splitter.split_text(page["text"])

        for local_i, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue  # skip blank chunks

            all_chunks.append({
                "text": chunk_text.strip(),
                "source": page["source"],
                "page_number": page["page_number"],
                "chunk_index": global_chunk_index,
                "chunk_on_page": local_i + 1
            })

            global_chunk_index += 1

    print(f"[✔] Total chunks created: {len(all_chunks)}")
    return all_chunks


# ──────────────────────────────────────────────
# STEP 4 — COMBINED PIPELINE ENTRY POINT
# ──────────────────────────────────────────────

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Full Day-2 pipeline:
    PDF file → pages → chunks with metadata

    This function will be called by main.py on /upload.
    Day 3 will extend this to also generate embeddings + store in FAISS.
    """
    pages = load_pdf(file_path)
    chunks = chunk_documents(pages)
    return chunks


# ──────────────────────────────────────────────
# QUICK TEST (run this file directly)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Change this path to your actual sample PDF
    TEST_PDF = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs",
        "sample_policy.pdf"
    )

    chunks = process_pdf(TEST_PDF)

    print("\n" + "="*60)
    print(f"TOTAL CHUNKS: {len(chunks)}")
    print("="*60)

    # Print first 3 chunks in detail
    for chunk in chunks[:3]:
        print(f"\n--- Chunk {chunk['chunk_index']} ---")
        print(f"  Source     : {chunk['source']}")
        print(f"  Page       : {chunk['page_number']}")
        print(f"  On page    : chunk {chunk['chunk_on_page']}")
        print(f"  Text length: {len(chunk['text'])} chars")
        print(f"  Preview    : {chunk['text'][:120]}...")

    print("\n[✔] Day 2 pipeline working correctly")