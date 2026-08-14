"""Typed public result contracts shared by interfaces and evaluation."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class SourceReference(TypedDict):
    rank: int
    company: str
    source_file: str
    page_number: int | str
    doc_type: str
    preview: str


class BuildStats(TypedDict):
    files: list[str]
    pages: int
    text_chunks: int
    table_pages: int
    indexed_documents: int
    embedding_backend: str
    build_seconds: float
    config: dict
    cache_hit: NotRequired[bool]
    cache_fingerprint: NotRequired[str]
    cache_path: NotRequired[str]


class AnswerResult(TypedDict):
    question: str
    answer: str
    sources: list[SourceReference]
    retrieval_strategy: str
    target_companies: list[str]
    model: str
    embedding_model: str
    latency_seconds: float
    stop_reason: str
    continuation_attempts: int
    retrieved_document_count: int
    context_document_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int
    cached_input_tokens: int
