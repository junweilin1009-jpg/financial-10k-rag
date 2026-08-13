"""Stable configuration for the public Financial 10-K RAG application."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_LLM_MODEL = "gpt-5.6-sol"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
CACHE_VERSION = "2026-07-17-public-v1"
PROJECT_COMPANIES = ("Alphabet/Google", "Amazon", "Microsoft")


@dataclass(frozen=True)
class RAGConfig:
    """Frozen retrieval defaults used for the reported final experiment."""

    llm_model: str = DEFAULT_LLM_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = 1000
    chunk_overlap: int = 150
    fact_k: int = 6
    company_k: int = 3
    fetch_k: int = 24
    qualitative_k: int = 8
    table_supplement_k: int = 2
    max_output_tokens: int = 3000
    max_context_chars: int = 70000
    max_continuation_attempts: int = 1
    reasoning_effort: str = "medium"
    response_verbosity: str = "medium"

