from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from financial_rag import FinancialRAG


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class SafeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = object.__new__(FinancialRAG)
        self.engine.embeddings = FakeEmbeddings()
        self.engine.table_pages = []
        self.engine.vector_store = FAISS.from_documents(
            [
                Document(page_content="alpha", metadata={"company": "Alphabet/Google"}),
                Document(page_content="amazon", metadata={"company": "Amazon"}),
            ],
            self.engine.embeddings,
        )

    def test_round_trip_uses_faiss_and_json_only(self) -> None:
        payload = self.engine._safe_cache_payload({"indexed_documents": 2})
        with tempfile.TemporaryDirectory() as raw_dir:
            cache_dir = Path(raw_dir)
            faiss.write_index(self.engine.vector_store.index, str(cache_dir / "index.faiss"))

            restored = object.__new__(FinancialRAG)
            restored.embeddings = self.engine.embeddings
            restored._load_safe_cache(cache_dir, payload)

            self.assertEqual(restored.vector_store.index.ntotal, 2)
            self.assertEqual(
                {doc.metadata["company"] for doc in restored.vector_store.docstore._dict.values()},
                {"Alphabet/Google", "Amazon"},
            )
            self.assertFalse((cache_dir / "index.pkl").exists())

    def test_rejects_mismatched_document_mapping(self) -> None:
        payload = self.engine._safe_cache_payload({"indexed_documents": 2})
        payload["index_to_docstore_id"].pop("1")
        with tempfile.TemporaryDirectory() as raw_dir:
            cache_dir = Path(raw_dir)
            faiss.write_index(self.engine.vector_store.index, str(cache_dir / "index.faiss"))
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.engine._load_safe_cache(cache_dir, payload)


if __name__ == "__main__":
    unittest.main()
