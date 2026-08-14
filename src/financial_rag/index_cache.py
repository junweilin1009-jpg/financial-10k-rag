"""Safe persistence helpers for local FAISS indexes."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def load_vector_store(
    faiss_path: Path,
    payload: dict,
    embeddings: Embeddings,
) -> FAISS:
    """Restore a FAISS store without deserializing executable pickle data."""
    index_path = faiss_path / "index.faiss"
    documents = payload.get("documents")
    raw_mapping = payload.get("index_to_docstore_id")
    if (
        not index_path.is_file()
        or not isinstance(documents, dict)
        or not isinstance(raw_mapping, dict)
    ):
        raise ValueError("Cache is incomplete or uses the retired pickle format.")

    try:
        mapping = {int(index): str(doc_id) for index, doc_id in raw_mapping.items()}
        docstore = InMemoryDocstore(
            {
                str(doc_id): Document(
                    page_content=str(item["page_content"]),
                    metadata=dict(item["metadata"]),
                )
                for doc_id, item in documents.items()
            }
        )
        index = faiss.read_index(str(index_path))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Cache contents are invalid: {exc}") from exc

    expected_indexes = set(range(index.ntotal))
    if set(mapping) != expected_indexes or set(mapping.values()) != set(documents):
        raise ValueError("Cache document mapping does not match the FAISS index.")
    return FAISS(embeddings, index, docstore, mapping)


def cache_payload(
    vector_store: FAISS,
    table_pages: list[Document],
    stats: dict,
) -> dict:
    """Create a JSON-only cache payload for documents and metadata."""
    indexed = getattr(vector_store.docstore, "_dict", {})
    return {
        "cache_format": "faiss-json-v2",
        "build_stats": stats,
        "index_to_docstore_id": {
            str(index): doc_id for index, doc_id in vector_store.index_to_docstore_id.items()
        },
        "documents": {
            doc_id: {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            for doc_id, document in indexed.items()
        },
        "table_pages": [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in table_pages
        ],
    }


def read_cache(
    cache_path: Path,
    embeddings: Embeddings,
) -> tuple[FAISS, dict, list[Document]]:
    """Read and validate the JSON-backed cache at one fingerprint path."""
    faiss_path = cache_path / "faiss"
    metadata_path = cache_path / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("cache_format") != "faiss-json-v2":
        raise ValueError("Cache format is missing or unsupported.")
    vector_store = load_vector_store(faiss_path, payload, embeddings)
    table_pages = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in payload.get("table_pages", [])
    ]
    return vector_store, dict(payload["build_stats"]), table_pages


def write_cache(
    cache_path: Path,
    vector_store: FAISS,
    table_pages: list[Document],
    stats: dict,
) -> None:
    """Persist a native FAISS index and a non-executable JSON document map."""
    faiss_path = cache_path / "faiss"
    faiss_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(vector_store.index, str(faiss_path / "index.faiss"))
    payload = cache_payload(vector_store, table_pages, stats)
    (cache_path / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
