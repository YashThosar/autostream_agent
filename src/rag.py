"""
rag.py – RAG pipeline for AutoStream knowledge base.

Uses TF-IDF cosine similarity (sklearn) — lightweight, zero external API calls,
fully local. Retrieves the top-k most relevant document chunks for a given query.
"""
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Load knowledge base
# ---------------------------------------------------------------------------

_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")


def _load_documents() -> Tuple[List[str], List[dict]]:
    """Return (texts, metadata_list) from the knowledge base JSON."""
    with open(_KB_PATH, "r") as f:
        kb = json.load(f)
    texts = [doc["text"] for doc in kb["documents"]]
    meta  = [{"id": doc["id"], "category": doc["category"]} for doc in kb["documents"]]
    return texts, meta


_DOCS, _META = _load_documents()
_VECTORIZER  = TfidfVectorizer(stop_words="english")
_DOC_VECTORS = _VECTORIZER.fit_transform(_DOCS)


# ---------------------------------------------------------------------------
# Retrieval function
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = 2) -> str:
    """
    Retrieve the most relevant knowledge base chunks for a query.

    Args:
        query : Natural language query from the user or agent.
        top_k : Number of top chunks to return.

    Returns:
        A formatted string of retrieved context, ready to inject into an LLM prompt.
    """
    query_vec   = _VECTORIZER.transform([query])
    scores      = cosine_similarity(query_vec, _DOC_VECTORS).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0.05:          # relevance threshold — skip noise
            results.append(
                f"[{_META[idx]['category'].upper()}] {_DOCS[idx]}"
            )

    if not results:
        return "No specific information found in the knowledge base for this query."

    return "\n\n".join(results)


if __name__ == "__main__":
    # Quick sanity test
    print(retrieve("what is the price of the pro plan"))
    print("---")
    print(retrieve("can I get a refund"))
