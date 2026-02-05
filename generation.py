from transformers import pipeline
from retrieval import build_context

# -------------------------
# Load model ONCE
# -------------------------
qa = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1,              # CPU (safe for Streamlit Cloud)
    truncation=True
)

# -------------------------
# Optimized generation
# -------------------------
def generate_answer(query, chunks):
    """
    query: str
    chunks: list of retrieved chunks (from retrieve_hybrid)
    """

    # Build compact, grounded context
    context_text = build_context(chunks, max_tokens=1024)

    prompt = f"""
You are an expert educator.

Using ONLY the information in the context below,
write a clear and complete explanation (3-5 sentences).

Do NOT answer in a single phrase.
Do NOT copy text verbatim.
Be concise but informative.
Provide a complete sentence and include relevant context.
Do not answer with a single word unless unavoidable.

Context:
{context_text}

Question:
{query}

Answer:
""".strip()

    outputs = qa(
        prompt,
        max_new_tokens=256,
        min_new_tokens=80,        # forces explanation
        do_sample=False,          # deterministic
        repetition_penalty=1.1,
        length_penalty=1.2,
        num_beams=4               # better structure
    )

    return outputs[0]["generated_text"].strip()