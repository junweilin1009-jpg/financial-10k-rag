"""Streamlit interface for the final Financial 10-K RAG system."""

from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from financial_rag import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    FinancialRAG,
    RAGConfig,
    validate_model,
)
from financial_rag.document_processing import infer_company
from financial_rag.filings import validate_filing_set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "10k"
CACHE_DIR = PROJECT_ROOT / "cache" / "faiss"
UPLOAD_DIR = PROJECT_ROOT / "cache" / "uploads"
load_dotenv(PROJECT_ROOT / ".env")

CANONICAL_FILENAMES = {
    "Alphabet/Google": "Alphabet_10k_2025.pdf",
    "Amazon": "Amazon_10k_2025.pdf",
    "Microsoft": "Microsoft_10K_2025.pdf",
}

st.set_page_config(page_title="Financial 10-K RAG Assistant", page_icon="📊", layout="wide")
st.title("Financial 10-K RAG Assistant")
st.caption(
    "Filing-grounded analysis of Alphabet, Amazon, and Microsoft 2025 Form 10-Ks "
    "with GPT-5.6 Sol, OpenAI embeddings, and PDF-page citations."
)


def included_pdfs() -> list[Path]:
    return [DATA_DIR / filename for filename in CANONICAL_FILENAMES.values()]


def save_uploaded_pdfs(uploaded_files) -> list[Path]:
    if len(uploaded_files) != 3:
        raise ValueError("Upload exactly three PDFs: one Alphabet, one Amazon, and one Microsoft filing.")

    files_by_company: dict[str, tuple[str, bytes]] = {}
    digest = hashlib.sha256()
    for uploaded in uploaded_files:
        if not uploaded.name.lower().endswith(".pdf"):
            raise ValueError(f"Not a PDF: {uploaded.name}")
        company = infer_company(uploaded.name)
        if company not in CANONICAL_FILENAMES:
            raise ValueError(
                f"Cannot identify the company from {uploaded.name}. Include Alphabet/Google, "
                "Amazon, or Microsoft in each filename."
            )
        if company in files_by_company:
            raise ValueError(f"More than one uploaded file was identified as {company}.")
        data = uploaded.getvalue()
        digest.update(uploaded.name.encode("utf-8"))
        digest.update(data)
        files_by_company[company] = (uploaded.name, data)

    missing = set(CANONICAL_FILENAMES) - set(files_by_company)
    if missing:
        raise ValueError("Missing filing(s): " + ", ".join(sorted(missing)))

    target_dir = UPLOAD_DIR / digest.hexdigest()[:16]
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for company, canonical_name in CANONICAL_FILENAMES.items():
        destination = target_dir / canonical_name
        destination.write_bytes(files_by_company[company][1])
        paths.append(destination)
    return paths


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Retrieved sources ({len(sources)})"):
        for source in sources:
            st.markdown(
                f"**{source['rank']}. {source['company']} - {source['source_file']}, "
                f"PDF page {source['page_number']} ({source['doc_type']})**"
            )
            st.write(source["preview"])
            st.divider()


def reset_chat() -> None:
    st.session_state.pop("rag_engine", None)
    st.session_state.pop("messages", None)


def conversation_csv(messages: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["role", "content", "metadata", "source_count"],
        lineterminator="\n",
    )
    writer.writeheader()
    for message in messages:
        writer.writerow({
            "role": message["role"],
            "content": message["content"],
            "metadata": message.get("metadata", ""),
            "source_count": len(message.get("sources", [])),
        })
    return buffer.getvalue()


with st.sidebar:
    st.header("Runtime configuration")
    openai_key = st.text_input(
        "OpenAI API key",
        type="password",
        help="Used for both text embeddings and answer generation. It is not written to disk.",
    )
    if os.environ.get("OPENAI_API_KEY"):
        st.caption("An OpenAI API key is available from the environment.")

    model_id = st.text_input(
        "OpenAI model ID",
        value=DEFAULT_LLM_MODEL,
        help="The reported final experiment used GPT-5.6 Sol. Change this only if your API key cannot access it.",
    )

    st.subheader("Source filings")
    uploaded_files = st.file_uploader(
        "Optional: replace all three included filings",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload exactly one Alphabet/Google, one Amazon, and one Microsoft PDF.",
    )
    source_mode = "Uploaded replacements" if uploaded_files else "Included course filings"
    st.caption(f"Current source: {source_mode}")

    build_button = st.button("Load / build RAG index", type="primary", use_container_width=True)
    if st.button("Reset session", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()
    st.caption(
        "API calls are billable. The first run embeds the PDFs; later runs reuse a local FAISS cache."
    )


if build_button:
    api_key = openai_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.error("Enter an OpenAI API key or set OPENAI_API_KEY in the environment.")
    else:
        try:
            pdf_paths = save_uploaded_pdfs(uploaded_files) if uploaded_files else included_pdfs()
            missing = [path for path in pdf_paths if not path.exists()]
            if missing:
                raise FileNotFoundError("Missing included PDF(s): " + ", ".join(path.name for path in missing))
            validate_filing_set(pdf_paths)
            config = RAGConfig(llm_model=model_id.strip(), embedding_model=DEFAULT_EMBEDDING_MODEL)
            with st.spinner("Validating the model and loading the financial index..."):
                validate_model(config.llm_model, api_key=api_key)
                engine = FinancialRAG(config, api_key=api_key)
                dimensions = engine.validate_embedding_credentials()
                stats = engine.build_or_load(pdf_paths, cache_root=CACHE_DIR, rebuild=False)
            st.session_state.rag_engine = engine
            st.session_state.messages = []
            action = "loaded from cache" if stats.get("cache_hit") else "built from the PDFs"
            st.success(
                f"RAG index {action}. {stats.get('indexed_documents', 0)} documents; "
                f"{dimensions} embedding dimensions."
            )
        except Exception as exc:
            st.exception(exc)


engine = st.session_state.get("rag_engine")
if engine is None:
    st.info("Enter an API key, confirm the model ID, and load the RAG index to begin.")
    st.stop()

st.success(
    f"Ready: {engine.config.llm_model} + {engine.config.embedding_model} | "
    f"{engine.build_stats.get('indexed_documents', 0)} indexed documents"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("metadata"):
            st.caption(message["metadata"])
        render_sources(message.get("sources", []))

question = st.chat_input("Ask a filing-grounded question about Alphabet, Amazon, or Microsoft...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving filing evidence and generating an answer..."):
            try:
                result = engine.answer(question)
                st.markdown(result["answer"])
                metadata = (
                    f"Model: {result['model']} | Strategy: {result['retrieval_strategy']} | "
                    f"Latency: {result['latency_seconds']}s | Tokens: {result['total_tokens']:,} "
                    f"(input {result['input_tokens']:,}, output {result['output_tokens']:,})"
                )
                st.caption(metadata)
                render_sources(result["sources"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": metadata,
                    "sources": result["sources"],
                })
            except Exception as exc:
                st.exception(exc)

if st.session_state.messages:
    st.download_button(
        "Download conversation as CSV",
        data=conversation_csv(st.session_state.messages),
        file_name="financial_rag_conversation.csv",
        mime="text/csv",
    )
