# retrieval.py

import json
import faiss
import numpy as np
import nltk
import string

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# =====================================================
# Ensure NLTK resources (Cloud-safe)
# =====================================================
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


# =====================================================
# Global state (lazy, cached)
# =====================================================
_processed_chunks = None
_embedder = None
_faiss_index = None
_bm25 = None
_stop_words = None


# =====================================================
# Initializer (idempotent)
# =====================================================
def load_retriever():
    global _processed_chunks, _embedder, _faiss_index, _bm25, _stop_words

    # Already initialized
    if _faiss_index is not None and _bm25 is not None:
        return

    print("🔧 Initializing retriever...")

    ensure_nltk()
    _stop_words = set(stopwords.words("english"))

    # Load corpus
    with open("data/wikipedia_corpus_chunks.json", "r", encoding="utf-8") as f:
        _processed_chunks = json.load(f)

    texts = [c["text"] for c in _processed_chunks]

    # -------------------------
    # Dense retrieval (FAISS)
    # -------------------------
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = _embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False   # ✅ MUST be False on Streamlit Cloud
    )

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dim = embeddings.shape[1]

    _faiss_index = faiss.IndexFlatIP(dim)
    _faiss_index.add(embeddings.astype("float32"))

    # -------------------------
    # Sparse retrieval (BM25)
    # -------------------------
    def preprocess(text):
        tokens = word_tokenize(text.lower())
        return [
            t for t in tokens
            if t not in _stop_words
            and t not in string.punctuation
            and len(t) > 1
        ]

    tokenized_texts = [preprocess(t) for t in texts]

    if not tokenized_texts:
        raise RuntimeError("BM25 initialization failed: no tokenized texts")

    _bm25 = BM25Okapi(tokenized_texts)

    print("✅ Retriever ready")


# =====================================================
# Retrieval methods
# =====================================================
def reciprocal_rank_fusion(dense_ids, sparse_ids, k=60, top_n=5):
    scores = {}

    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    for rank, idx in enumerate(sparse_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:top_n]]


def retrieve_hybrid(query, top_k=5):
    load_retriever()

    q_emb = _embedder.encode([query], convert_to_numpy=True)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)

    _, dense_ids = _faiss_index.search(q_emb.astype("float32"), top_k)

    sparse_scores = _bm25.get_scores(query.split())
    sparse_ids = np.argsort(sparse_scores)[::-1][:top_k]

    fused_ids = reciprocal_rank_fusion(
        dense_ids[0],
        sparse_ids,
        top_n=top_k * 2
    )

    seen = set()
    results = []

    for idx in fused_ids:
        chunk = _processed_chunks[idx]
        if chunk["url"] not in seen:
            results.append(chunk)
            seen.add(chunk["url"])
        if len(results) == top_k:
            break

    return results


def retrieve_dense_only(query, top_k=10):
    load_retriever()

    q_emb = _embedder.encode([query], convert_to_numpy=True)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)

    _, ids = _faiss_index.search(q_emb.astype("float32"), top_k)
    return [_processed_chunks[i]["url"] for i in ids[0]]


def retrieve_sparse_only(query, top_k=10):
    load_retriever()

    scores = _bm25.get_scores(query.split())
    ids = np.argsort(scores)[::-1][:top_k]
    return [_processed_chunks[i]["url"] for i in ids]


def build_context(chunks, max_tokens=512):
    context = []
    total = 0

    for i, c in enumerate(chunks, 1):
        text = f"[Source {i}] {c['text']}"
        tokens = len(word_tokenize(text))

        if total + tokens > max_tokens:
            break

        context.append(text)
        total += tokens

    return "\n".join(context)

def get_processed_chunks():
    load_retriever()
    return _processed_chunks