"""Public package interface for the Financial 10-K RAG project."""

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, RAGConfig
from .engine import FinancialRAG, list_available_models, validate_model
from .schemas import AnswerResult, BuildStats, SourceReference

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_LLM_MODEL",
    "FinancialRAG",
    "RAGConfig",
    "AnswerResult",
    "BuildStats",
    "SourceReference",
    "list_available_models",
    "validate_model",
]
