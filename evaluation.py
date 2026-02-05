# evaluation.py

import math
import time
import re
import string
from collections import Counter
from nltk.util import ngrams
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from retrieval import retrieve_hybrid
from generation import generate_answer

# -------------------------
# Utilities
# -------------------------
def normalize_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred, gold):
    return int(normalize_text(pred) == normalize_text(gold))


def f1_score(pred, gold):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if not pred_tokens or not gold_tokens or num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# -------------------------
# Retrieval metrics
# -------------------------
def get_ranked_urls(query, k=10):
    return [c["url"] for c in retrieve_hybrid(query, top_k=k)]


def reciprocal_rank(ranked_urls, gold_url):
    for i, url in enumerate(ranked_urls, 1):
        if url == gold_url:
            return 1 / i
    return 0.0


def compute_mrr(qa_dataset, k=10):
    return sum(
        reciprocal_rank(get_ranked_urls(q["question"], k), q["source_url"])
        for q in qa_dataset
    ) / len(qa_dataset)


def recall_at_k(qa_dataset, k=10):
    return sum(
        q["source_url"] in get_ranked_urls(q["question"], k)
        for q in qa_dataset
    ) / len(qa_dataset)


def precision_at_k(qa_dataset, k=5):
    return recall_at_k(qa_dataset, k) / k


def ndcg_at_k(qa_dataset, k=10):
    scores = []
    for q in qa_dataset:
        urls = get_ranked_urls(q["question"], k)
        if q["source_url"] in urls:
            rank = urls.index(q["source_url"]) + 1
            scores.append(1 / math.log2(rank + 1))
        else:
            scores.append(0)
    return sum(scores) / len(scores)


# -------------------------
# Answer quality metrics
# -------------------------
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def rouge_l(pred, gold):
    return scorer.score(gold, pred)["rougeL"].fmeasure


def semantic_similarity(pred, gold):
    e1 = embedder.encode(pred, convert_to_tensor=True)
    e2 = embedder.encode(gold, convert_to_tensor=True)
    return util.cos_sim(e1, e2).item()


# -------------------------
# Latency
# -------------------------
def evaluate_latency(qa_dataset, sample_size=20):
    times = []
    for q in qa_dataset[:sample_size]:
        start = time.time()
        generate_answer(q["question"], retrieve_hybrid(q["question"], top_k=3))
        times.append(time.time() - start)
    return {
        "Avg Latency (s)": sum(times) / len(times),
        "Max Latency (s)": max(times),
    }


# -------------------------
# Full evaluation
# -------------------------
def full_evaluation(qa_dataset):
    results = {}

    # Retrieval
    results["MRR@10"] = compute_mrr(qa_dataset)
    results["Recall@10"] = recall_at_k(qa_dataset)
    results["Precision@5"] = precision_at_k(qa_dataset)
    results["NDCG@10"] = ndcg_at_k(qa_dataset)

    # Generation quality
    em, f1, rouge, sem = [], [], [], []

    for q in qa_dataset[30:]:
        chunks = retrieve_hybrid(q["question"], top_k=3)
        pred = generate_answer(q["question"], chunks)
        gold = q["answer"]

        em.append(exact_match(pred, gold))
        f1.append(f1_score(pred, gold))
        rouge.append(rouge_l(pred, gold))
        sem.append(semantic_similarity(pred, gold))

    results["Exact Match"] = sum(em) / len(em)
    results["F1"] = sum(f1) / len(f1)
    results["ROUGE-L"] = sum(rouge) / len(rouge)
    results["SemanticSim"] = sum(sem) / len(sem)

    # Efficiency
    results.update(evaluate_latency(qa_dataset))

    return results