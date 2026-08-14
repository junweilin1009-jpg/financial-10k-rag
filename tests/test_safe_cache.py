from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from financial_rag import FinancialRAG
from financial_rag.index_cache import cache_payload, load_vector_store, read_cache, write_cache


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
        with tempfile.TemporaryDirectory() as raw_dir:
            cache_dir = Path(raw_dir)
            write_cache(
                cache_dir,
                self.engine.vector_store,
                self.engine.table_pages,
                {"indexed_documents": 2},
            )
            restored, stats, table_pages = read_cache(cache_dir, self.engine.embeddings)

            self.assertEqual(restored.index.ntotal, 2)
            self.assertEqual(
                {doc.metadata["company"] for doc in restored.docstore._dict.values()},
                {"Alphabet/Google", "Amazon"},
            )
            self.assertEqual(stats["indexed_documents"], 2)
            self.assertEqual(table_pages, [])
            self.assertFalse((cache_dir / "index.pkl").exists())

    def test_rejects_mismatched_document_mapping(self) -> None:
        payload = cache_payload(
            self.engine.vector_store,
            self.engine.table_pages,
            {"indexed_documents": 2},
        )
        payload["index_to_docstore_id"].pop("1")
        with tempfile.TemporaryDirectory() as raw_dir:
            cache_dir = Path(raw_dir)
            write_cache(
                cache_dir,
                self.engine.vector_store,
                self.engine.table_pages,
                {"indexed_documents": 2},
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_vector_store(cache_dir / "faiss", payload, self.engine.embeddings)


if __name__ == "__main__":
    unittest.main()
