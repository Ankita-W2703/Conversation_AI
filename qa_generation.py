# qa_generation.py

import json
import uuid
import random
from collections import defaultdict
from transformers import pipeline

from retrieval import retrieve_hybrid, build_context, get_processed_chunks

# -------------------------
# Load model ONCE
# -------------------------
qg_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1,
    truncation=True
)

# -------------------------
# Answer generation (USED BY PIPELINE)
# -------------------------
def generate_answer1(query):
    chunks = retrieve_hybrid(query, top_k=3)
    context = build_context(chunks, max_tokens=512)

    prompt = f"""
You are a knowledgeable assistant.

Using ONLY the context below, answer the question clearly.
Do not answer in a single phrase.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    out = qg_model(
        prompt,
        max_new_tokens=200,
        min_new_tokens=60,
        do_sample=False
    )[0]["generated_text"].strip()

    return out, context


# -------------------------
# Question generation
# -------------------------
QUESTION_PROMPTS = {
    "factual": "Generate one factual question answered by the text below.",
    "comparative": "Generate one comparative question based on the text below.",
    "inferential": "Generate one inferential 'why' or 'how' question based on the text below.",
    "multi-hop": "Generate one multi-hop question that connects multiple ideas in the text below."
}

def group_chunks_by_url(chunks):
    grouped = defaultdict(list)
    for c in chunks:
        grouped[c["url"]].append(c)
    return grouped


def generate_questions_from_corpus(chunks_by_url, max_questions=100):
    questions = []
    urls = list(chunks_by_url.keys())
    random.shuffle(urls)

    for url in urls:
        chunks = chunks_by_url[url]
        title = chunks[0]["title"]
        context = chunks[0]["text"][:1000]

        for q_type, prompt in QUESTION_PROMPTS.items():
            if len(questions) >= max_questions:
                return questions

            full_prompt = f"""
{prompt}

Text:
{context}
""".strip()

            try:
                q = qg_model(full_prompt, max_new_tokens=64)[0]["generated_text"].strip()
            except Exception:
                continue

            if len(q) < 10:
                continue

            questions.append({
                "question_id": str(uuid.uuid4()),
                "question": q,
                "answer": None,
                "source_url": url,
                "source_title": title,
                "question_type": q_type
            })

    return questions


def generate_ground_truth_answers(questions, chunks_by_url):
    for q in questions:
        chunks = chunks_by_url[q["source_url"]]
        context = chunks[0]["text"][:1200]

        prompt = f"""
Answer the question using ONLY the text below.

Text:
{context}

Question:
{q["question"]}
""".strip()

        try:
            ans = qg_model(prompt, max_new_tokens=80)[0]["generated_text"].strip()
        except Exception:
            ans = ""

        q["answer"] = ans

    return questions


# -------------------------
# SCRIPT ENTRY POINT (RUN ONCE)
# -------------------------
def build_qa_dataset(output_path="data/qa_dataset_100.json", max_questions=100):
    chunks = get_processed_chunks()
    chunks_by_url = group_chunks_by_url(chunks)

    questions = generate_questions_from_corpus(
        chunks_by_url,
        max_questions=max_questions
    )

    qa_dataset = generate_ground_truth_answers(
        questions,
        chunks_by_url
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, indent=2)

    print(f"✅ QA dataset saved to {output_path}")


if __name__ == "__main__":
    build_qa_dataset()
