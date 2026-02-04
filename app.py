# app.py

import streamlit as st
import time
import pandas as pd

from retrieval import (
    retrieve_hybrid,
    retrieve_dense_only,
    retrieve_sparse_only
)
from generation import generate_answer


# =====================================================
# Page config (must be FIRST Streamlit call)
# =====================================================
st.set_page_config(
    page_title="Hybrid RAG – FLAN-T5",
    layout="wide"
)


# =====================================================
# Session state
# =====================================================
if "busy" not in st.session_state:
    st.session_state.busy = False


# =====================================================
# Initialize retriever ONCE (cached)
# =====================================================
@st.cache_resource
def init_retriever():
    from retrieval import load_retriever
    load_retriever()


with st.spinner("Initializing retriever (loading documents & indexes)..."):
    init_retriever()


# =====================================================
# UI HEADER
# =====================================================
st.title("Hybrid RAG System")
st.caption("Type a question and click Submit to test input → output")


# =====================================================
# FORM (atomic submit – Streamlit-safe)
# =====================================================
with st.form("rag_form", clear_on_submit=False):
    query = st.text_input(
        "Enter your question",
        placeholder="e.g. What is artificial intelligence?",
        disabled=st.session_state.busy
    )

    submit_clicked = st.form_submit_button(
        "Submit",
        disabled=st.session_state.busy
    )


# =====================================================
# MAIN EXECUTION
# =====================================================
if submit_clicked:

    # Guard against race conditions
    if st.session_state.busy:
        st.stop()

    # Validate AFTER submit (required for forms)
    if not query.strip():
        st.warning("⚠️ Please enter a question before submitting.")
        st.stop()

    st.session_state.busy = True

    # -------------------------
    # Retrieval + Generation
    # -------------------------
    with st.spinner("Retrieving knowledge and generating answer..."):
        start = time.time()

        chunks = retrieve_hybrid(query, top_k=5)
        answer = generate_answer(query, chunks)

        latency = time.time() - start

    st.session_state.busy = False

    # =================================================
    # OUTPUT
    # =================================================
    st.subheader("Answer")
    st.write(answer)

    st.metric("⏱ Response Time (s)", round(latency, 2))


    # =================================================
    # Retrieved Evidence
    # =================================================
    st.subheader("Retrieved Evidence")

    for i, c in enumerate(chunks, 1):
        with st.expander(f"Chunk {i} | {c['title']}"):
            st.write(c["text"][:800] + "…")
            st.markdown(f"**Source:** {c['url']}")


    # =================================================
    # Retrieval Comparison
    # =================================================
    st.subheader("Retrieval Method Comparison")

    with st.spinner("Comparing retrieval methods..."):
        dense_urls = retrieve_dense_only(query)
        sparse_urls = retrieve_sparse_only(query)
        hybrid_urls = [c["url"] for c in chunks]

    def pad_list(lst, length, pad_value=""):
        return lst + [pad_value] * (length - len(lst))

    max_len = 5
    dense_urls = pad_list(dense_urls[:max_len], max_len)
    sparse_urls = pad_list(sparse_urls[:max_len], max_len)
    hybrid_urls = pad_list(hybrid_urls[:max_len], max_len)

    df = pd.DataFrame({
        "Dense": dense_urls,
        "Sparse": sparse_urls,
        "Hybrid (RRF)": hybrid_urls
    })

    st.dataframe(df)
