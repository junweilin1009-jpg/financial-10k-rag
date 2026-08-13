from __future__ import annotations

import unittest
import os
from unittest.mock import patch

from langchain_core.documents import Document

from financial_rag import FinancialRAG, RAGConfig


class ContextFormattingTests(unittest.TestCase):
    def test_explicit_api_key_does_not_mutate_process_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            engine = FinancialRAG(api_key="test-key-not-sent")
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            self.assertEqual(engine.config, RAGConfig())

    def test_returns_only_documents_that_fit_context_limit(self) -> None:
        engine = object.__new__(FinancialRAG)
        engine.config = RAGConfig(max_context_chars=150)
        documents = [
            Document(
                page_content="first evidence",
                metadata={
                    "company": "Amazon",
                    "source_file": "amazon.pdf",
                    "page_number": 1,
                },
            ),
            Document(
                page_content="x" * 200,
                metadata={
                    "company": "Microsoft",
                    "source_file": "microsoft.pdf",
                    "page_number": 2,
                },
            ),
        ]

        context, included = engine._format_context(documents)

        self.assertIn("first evidence", context)
        self.assertEqual(included, documents[:1])

    def test_never_skips_an_oversized_earlier_document(self) -> None:
        engine = object.__new__(FinancialRAG)
        engine.config = RAGConfig(max_context_chars=50)
        documents = [
            Document(page_content="x" * 100, metadata={}),
            Document(page_content="small", metadata={}),
        ]

        context, included = engine._format_context(documents)

        self.assertEqual(context, "")
        self.assertEqual(included, [])


if __name__ == "__main__":
    unittest.main()
