"""Context formatting and response normalization for answer generation."""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document


def format_context(
    documents: Sequence[Document],
    max_context_chars: int,
) -> tuple[str, list[Document]]:
    """Format context and return only documents actually sent to the model."""
    blocks = []
    included_documents = []
    total_chars = 0
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata
        label = (
            f"Source {index}: {metadata.get('company', '')}; "
            f"{metadata.get('source_file', metadata.get('source', ''))}; "
            f"PDF page {metadata.get('page_number', '')}; "
            f"type={metadata.get('doc_type', 'text')}"
        )
        block = f"[{label}]\n{doc.page_content.strip()}"
        if total_chars + len(block) > max_context_chars:
            break
        blocks.append(block)
        included_documents.append(doc)
        total_chars += len(block)
    return "\n\n".join(blocks), included_documents


def message_text(message) -> str:
    """Normalize LangChain response content into plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def stop_reason(message) -> str:
    """Normalize provider stop metadata."""
    metadata = getattr(message, "response_metadata", None) or {}
    reason = str(
        metadata.get("finish_reason")
        or metadata.get("stop_reason")
        or metadata.get("stopReason")
        or metadata.get("status")
        or ""
    )
    return "max_tokens" if reason in {"length", "incomplete"} else reason


def token_usage(message) -> dict[str, int]:
    """Normalize token accounting returned by LangChain model wrappers."""
    usage = getattr(message, "usage_metadata", None) or {}
    metadata = getattr(message, "response_metadata", None) or {}
    if not usage:
        usage = metadata.get("token_usage") or metadata.get("usage") or {}

    input_details = usage.get("input_token_details") or {}
    output_details = usage.get("output_token_details") or {}

    def value(*keys) -> int:
        for key in keys:
            raw = usage.get(key)
            if raw is not None:
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    pass
        return 0

    input_tokens = value("input_tokens", "prompt_tokens")
    output_tokens = value("output_tokens", "completion_tokens")
    total_tokens = value("total_tokens") or input_tokens + output_tokens
    reasoning_tokens = output_details.get("reasoning", 0) or usage.get(
        "reasoning_tokens", 0
    )
    cached_input_tokens = (
        input_details.get("cache_read", 0)
        or usage.get("cache_read_input_tokens", 0)
        or usage.get("cached_tokens", 0)
    )
    try:
        reasoning_tokens = int(reasoning_tokens)
    except (TypeError, ValueError):
        reasoning_tokens = 0
    try:
        cached_input_tokens = int(cached_input_tokens)
    except (TypeError, ValueError):
        cached_input_tokens = 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_input_tokens": cached_input_tokens,
    }
