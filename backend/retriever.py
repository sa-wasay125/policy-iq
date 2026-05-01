"""
retriever.py
------------
Handles:
  - Taking a user query
  - Searching FAISS for top K relevant chunks
  - Filtering by confidence threshold
  - Formatting context for LLM consumption
  - Returning structured results with source attribution
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Tuple
from backend.embedder import search_index, load_index
from backend.utils.file_handler import CONFIDENCE_THRESHOLD, TOP_K


# ──────────────────────────────────────────────
# GLOBAL INDEX CACHE
# ──────────────────────────────────────────────

# Load index once at module level — avoids reloading on every query
_index = None
_metadata = None


def get_index():
    """
    Lazy-load the FAISS index.
    First call loads from disk, subsequent calls return cached version.
    """
    global _index, _metadata

    if _index is None or _metadata is None:
        print("[✔] Loading FAISS index into memory...")
        _index, _metadata = load_index()

    return _index, _metadata


def reload_index():
    """
    Force reload the index from disk.
    Called after a new PDF is uploaded and indexed.
    """
    global _index, _metadata
    _index, _metadata = load_index()
    print("[✔] FAISS index reloaded")


# ──────────────────────────────────────────────
# CORE RETRIEVER
# ──────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = TOP_K,
    confidence_threshold: float = CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    """
    Main retrieval function.

    Steps:
    1. Load FAISS index
    2. Search for top K similar chunks
    3. Filter by confidence threshold
    4. Format context string for LLM
    5. Return structured result

    Returns:
    {
        "query": "original query",
        "found": True/False,
        "results": [...],          ← all retrieved chunks with scores
        "context": "...",          ← formatted text for LLM prompt
        "top_score": 0.87,         ← best match score
        "source_pages": [3, 4, 6]  ← pages referenced
    }
    """
    if not query or not query.strip():
        return _empty_result(query, reason="Empty query")

    try:
        index, metadata = get_index()
    except FileNotFoundError:
        return _empty_result(query, reason="No document indexed yet. Please upload a PDF first.")

    # Search FAISS
    raw_results = search_index(query, index, metadata, top_k=top_k)

    if not raw_results:
        return _empty_result(query, reason="No results found")

    # Get best score
    top_score = raw_results[0]["score"] if raw_results else 0.0

    # Filter by confidence threshold
    filtered = [r for r in raw_results if r["score"] >= confidence_threshold]

    if not filtered:
        return {
            "query": query,
            "found": False,
            "results": raw_results,
            "context": "",
            "top_score": top_score,
            "source_pages": [],
            "reason": f"Best match score {top_score:.2f} is below confidence threshold {confidence_threshold}"
        }

    # Build context string for LLM
    context = _build_context(filtered)

    # Collect unique source pages
    source_pages = sorted(list(set(r["page_number"] for r in filtered)))

    return {
        "query": query,
        "found": True,
        "results": filtered,
        "context": context,
        "top_score": top_score,
        "source_pages": source_pages,
        "reason": None
    }


# ──────────────────────────────────────────────
# CONTEXT BUILDER
# ──────────────────────────────────────────────

def _build_context(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a clean context string for the LLM.

    Format:
    [Source: filename.pdf | Page: 3 | Relevance: 87%]
    <chunk text>

    [Source: filename.pdf | Page: 6 | Relevance: 79%]
    <chunk text>

    This format lets the LLM know exactly where each piece of info came from.
    The LLM is then instructed to only answer from this context.
    """
    parts = []

    for r in results:
        score_pct = int(r["score"] * 100)
        header = (
            f"[Source: {r['source']} | "
            f"Page: {r['page_number']} | "
            f"Relevance: {score_pct}%]"
        )
        parts.append(f"{header}\n{r['text']}")

    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────

def _empty_result(query: str, reason: str = "") -> Dict[str, Any]:
    return {
        "query": query,
        "found": False,
        "results": [],
        "context": "",
        "top_score": 0.0,
        "source_pages": [],
        "reason": reason
    }


# ──────────────────────────────────────────────
# SOURCE ATTRIBUTION FORMATTER
# ──────────────────────────────────────────────

def format_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format results for API response — clean source attribution.

    Returns list of:
    {
        "page": 3,
        "source": "sample_policy.pdf",
        "relevance_pct": 87,
        "excerpt": "first 200 chars of chunk..."
    }

    This is what the frontend will display under every answer.
    """
    sources = []
    for r in results:
        sources.append({
            "page": r["page_number"],
            "source": r["source"],
            "relevance_pct": int(r["score"] * 100),
            "excerpt": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
        })
    return sources


# ──────────────────────────────────────────────
# TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What is the claim limit for hospitalization?",
        "What is covered under room rent expenses?",
        "What diseases are excluded from the policy?",
        "What is the ambulance expense coverage?",
        "Is mental illness covered?",
        "What is the grace period for premium payment?",
        "xyzxyzxyz this query should not match anything"
    ]

    print("=" * 65)
    print("RETRIEVER TEST")
    print("=" * 65)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)

        result = retrieve(query)

        if result["found"]:
            print(f"  Found    : YES")
            print(f"  Top score: {result['top_score']:.4f}")
            print(f"  Pages    : {result['source_pages']}")
            print(f"  Context preview:")
            print(f"    {result['context'][:200]}...")
        else:
            print(f"  Found    : NO")
            print(f"  Reason   : {result['reason']}")
            print(f"  Top score: {result['top_score']:.4f}")

    print("\n" + "=" * 65)
    print("[✔] Day 4 retriever working correctly")
    print("=" * 65)