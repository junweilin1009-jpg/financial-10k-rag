import unittest

from financial_rag import RAGConfig


class RAGConfigTests(unittest.TestCase):
    def test_default_configuration_is_valid(self) -> None:
        config = RAGConfig()
        self.assertLess(config.chunk_overlap, config.chunk_size)

    def test_rejects_overlap_equal_to_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_overlap"):
            RAGConfig(chunk_size=100, chunk_overlap=100)

    def test_rejects_non_positive_retrieval_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "fact_k"):
            RAGConfig(fact_k=0)

    def test_rejects_invalid_reasoning_effort(self) -> None:
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            RAGConfig(reasoning_effort="extreme")


if __name__ == "__main__":
    unittest.main()
