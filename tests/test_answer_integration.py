from __future__ import annotations

import unittest

from langchain_core.documents import Document

from financial_rag import FinancialRAG, RAGConfig


class FakeMessage:
    def __init__(self, content: str, stop_reason: str, input_tokens: int, output_tokens: int):
        self.content = content
        self.response_metadata = {"stop_reason": stop_reason}
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }


class FakeLLM:
    def __init__(self, responses: list[FakeMessage]):
        self.responses = iter(responses)
        self.calls: list[list] = []

    def invoke(self, messages: list) -> FakeMessage:
        self.calls.append(list(messages))
        return next(self.responses)


class AnswerWorkflowIntegrationTests(unittest.TestCase):
    def test_retrieval_context_generation_and_source_contract(self) -> None:
        documents = [
            Document(
                page_content="Revenue was $100 million.",
                metadata={
                    "company": "Amazon",
                    "source_file": "amazon.pdf",
                    "page_number": 12,
                    "doc_type": "page_text",
                },
            ),
            Document(
                page_content="Operating income was $20 million.",
                metadata={
                    "company": "Amazon",
                    "source_file": "amazon.pdf",
                    "page_number": 13,
                    "doc_type": "table_page",
                },
            ),
        ]
        llm = FakeLLM(
            [
                FakeMessage("The margin was", "max_tokens", 30, 4),
                FakeMessage(" 20%.", "stop", 12, 3),
            ]
        )
        engine = object.__new__(FinancialRAG)
        engine.config = RAGConfig(max_context_chars=5_000, max_continuation_attempts=1)
        engine.llm = llm
        engine.retrieve = lambda _question: (documents, "test_strategy", ["Amazon"])

        result = engine.answer("What was the operating margin?")

        self.assertEqual(result["answer"], "The margin was\n20%.")
        self.assertEqual(result["retrieval_strategy"], "test_strategy")
        self.assertEqual(result["target_companies"], ["Amazon"])
        self.assertEqual(result["retrieved_document_count"], 2)
        self.assertEqual(result["context_document_count"], 2)
        self.assertEqual(result["continuation_attempts"], 1)
        self.assertEqual(result["input_tokens"], 42)
        self.assertEqual(result["output_tokens"], 7)
        self.assertEqual(result["total_tokens"], 49)
        self.assertEqual([source["page_number"] for source in result["sources"]], [12, 13])
        self.assertIn("Revenue was $100 million.", llm.calls[0][1][1])
        self.assertIn("Operating income was $20 million.", llm.calls[0][1][1])
        self.assertIn("Continue exactly where", llm.calls[1][-1][1])


if __name__ == "__main__":
    unittest.main()
