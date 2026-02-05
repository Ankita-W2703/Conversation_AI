# qa_generation.py

import json
import uuid
import random
from collections import defaultdict
from transformers import pipeline

from retrieval import retrieve_hybrid, build_context, get_processed_chunks

# =====================================================
# Load model ONCE
# =====================================================
qg_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1
)

# =====================================================
# Answer Generation (LONG, GROUNDED)
# =====================================================
def generate_answer_long(query, chunks):
    context = build_context(chunks, max_tokens=700)

    prompt = f"""
You are a knowledgeable assistant.

Answer the question using ONLY the information in the context.
Write a complete, clear answer in 1–3 full sentences.
Do NOT answer with a single word or short phrase.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    output = qg_model(
        prompt,
        max_new_tokens=220,
        min_new_tokens=80,
        do_sample=False
    )[0]["generated_text"].strip()

    return output


# =====================================================
# Question Generation Prompts (CLEAN)
# =====================================================
QUESTION_PROMPTS = {
    "factual": "Generate one clear factual question that can be answered from the text below.",
    "comparative": "Generate one comparative question that requires comparing information in the text below.",
    "inferential": "Generate one inferential 'why' or 'how' question based on the text below.",
    "multi-hop": "Generate one high-level question asking for the main idea or explanation based on the text below."
}


# =====================================================
# Utilities
# =====================================================
def group_chunks_by_url(chunks):
    grouped = defaultdict(list)
    for c in chunks:
        grouped[c["url"]].append(c)
    return grouped


def clean_question(q):
    """Remove bad or malformed questions"""
    bad_patterns = ["Options:", "(A)", "(B)", "(C)", "Question:"]
    return not any(p in q for p in bad_patterns)


# =====================================================
# Question Generation
# =====================================================
def generate_questions_from_corpus(chunks_by_url, max_questions=100):
    questions = []
    urls = list(chunks_by_url.keys())
    random.shuffle(urls)

    for url in urls:
        chunks = chunks_by_url[url]
        title = chunks[0]["title"]

        # Use richer context
        context = " ".join(c["text"] for c in chunks[:2])[:1200]

        for q_type, prompt in QUESTION_PROMPTS.items():
            if len(questions) >= max_questions:
                return questions

            full_prompt = f"""
{prompt}

Text:
{context}

Question:
""".strip()

            try:
                q = qg_model(
                    full_prompt,
                    max_new_tokens=80,
                    do_sample=False
                )[0]["generated_text"].strip()
            except Exception:
                continue

            if len(q) < 15 or not clean_question(q):
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


# =====================================================
# Ground Truth Answer Generation (LONG)
# =====================================================
def generate_ground_truth_answers(questions, chunks_by_url):
    for q in questions:
        chunks = chunks_by_url[q["source_url"]][:3]

        answer = generate_answer_long(q["question"], chunks)

        # Safety fallback
        if len(answer.split()) < 10:
            answer = generate_answer_long(q["question"], chunks[:1])

        q["answer"] = answer

    return questions


# =====================================================
# Build QA Dataset (RUN ONCE)
# =====================================================
def build_qa_dataset(
    output_path="data/qa_dataset_100.json",
    max_questions=100
):
    print("🔧 Loading Wikipedia chunks...")
    chunks = get_processed_chunks()
    chunks_by_url = group_chunks_by_url(chunks)

    print("Generating questions...")
    questions = generate_questions_from_corpus(
        chunks_by_url,
        max_questions=max_questions
    )

    print("Generating long, grounded answers...")
    qa_dataset = generate_ground_truth_answers(
        questions,
        chunks_by_url
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, indent=2)

    print(f"QA dataset saved to {output_path}")
    print(f"Total Q&A pairs: {len(qa_dataset)}")


# =====================================================
# Entry Point
# =====================================================
if __name__ == "__main__":
    build_qa_dataset(max_questions=100)
