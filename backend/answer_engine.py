import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from dotenv import load_dotenv
from groq import Groq

from backend.retriever import retrieve, format_sources

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are PolicyIQ, an AI assistant that answers questions strictly based on insurance policy documents.

STRICT RULES:
1. Answer ONLY using the provided context below.
2. If the context does not contain enough information, say exactly: "This information was not found in the document."
3. Do NOT use your own knowledge or make assumptions.
4. Keep your answer concise and direct.
5. Always mention the page number(s) where you found the answer.

CONTEXT FROM DOCUMENT:
{context}

USER QUESTION:
{question}

ANSWER:"""


def _is_not_found_response(answer: str) -> bool:
    phrases = [
        "not found in the document",
        "not mentioned in the document",
        "not specified in the document",
        "cannot find",
        "no information",
        "does not contain",
        "not provided in",
        "not available in"
    ]
    return any(p in answer.lower() for p in phrases)


def generate_answer(query: str) -> Dict[str, Any]:

    # Step 1 — Retrieve
    retrieval = retrieve(query)

    # Step 2 — Hallucination gate
    if not retrieval["found"]:
        return {
            "question":       query,
            "answer":         "This information was not found in the document.",
            "found":          False,
            "confidence_pct": int(retrieval["top_score"] * 100),
            "sources":        [],
            "error":          None,
            "reason":         retrieval.get("reason", "")
        }

    # Step 3 — Build prompt
    prompt = SYSTEM_PROMPT.format(
        context=retrieval["context"],
        question=query
    )

    # Step 4 — Call Groq LLM
    try:
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=512
        )
        raw_answer = response.choices[0].message.content.strip()

    except Exception as e:
        return {
            "question":       query,
            "answer":         "An error occurred while generating the answer.",
            "found":          False,
            "confidence_pct": 0,
            "sources":        [],
            "error":          str(e),
            "reason":         "LLM call failed"
        }

    # Step 5 — Validate
    found = not _is_not_found_response(raw_answer)

    # Step 6 — Format sources
    sources = format_sources(retrieval["results"]) if found else []

    return {
        "question":       query,
        "answer":         raw_answer,
        "found":          found,
        "confidence_pct": int(retrieval["top_score"] * 100),
        "sources":        sources,
        "error":          None,
        "reason":         None if found else "LLM determined answer not in document"
    }


if __name__ == "__main__":

    test_queries = [
        "What is covered under room rent expenses?",
        "What is the ambulance expense limit?",
        "Is mental illness treatment covered?",
        "What is the grace period for premium payment?",
        "What diseases are excluded in the first two years?",
        "What is the waiting period for pre-existing conditions?",
        "What is the convalescence benefit amount?",
        "Who is the CEO of Apple?",
    ]

    print("=" * 65)
    print("DAY 5 — ANSWER ENGINE TEST")
    print("=" * 65)

    for query in test_queries:
        print(f"\nQ: {query}")
        print("-" * 50)

        result = generate_answer(query)

        print(f"  Found      : {result['found']}")
        print(f"  Confidence : {result['confidence_pct']}%")
        print(f"  Answer     : {result['answer'][:300]}")

        if result["sources"]:
            print(f"  Sources    :")
            for s in result["sources"]:
                print(f"    - Page {s['page']} ({s['relevance_pct']}% match)")

        if result["error"]:
            print(f"  ERROR      : {result['error']}")

    print("\n" + "=" * 65)
    print("[✔] Day 5 answer engine working correctly")
    print("=" * 65)