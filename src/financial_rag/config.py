"""Stable configuration for the public Financial 10-K RAG application."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_LLM_MODEL = "gpt-5.6-sol"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
CACHE_VERSION = "2026-08-13-safe-cache-v2"
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

    def __post_init__(self) -> None:
        """Reject invalid settings before they reach APIs or retrieval code."""
        if not self.llm_model.strip():
            raise ValueError("llm_model must not be empty.")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
        for name in (
            "fact_k",
            "company_k",
            "fetch_k",
            "qualitative_k",
            "table_supplement_k",
            "max_output_tokens",
            "max_context_chars",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.max_continuation_attempts < 0:
            raise ValueError("max_continuation_attempts must be non-negative.")
        if self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("reasoning_effort is not supported.")
        if self.response_verbosity not in {"low", "medium", "high"}:
            raise ValueError("response_verbosity must be low, medium, or high.")
