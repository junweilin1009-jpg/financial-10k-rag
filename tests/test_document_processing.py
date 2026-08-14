import unittest

from financial_rag.document_processing import (
    clean_pdf_text,
    extract_markdown_tables,
    infer_company,
    infer_fiscal_year_end,
)
from financial_rag.engine import infer_fiscal_year_end as engine_fiscal_year_end


class DocumentProcessingTests(unittest.TestCase):
    def test_company_detection_from_public_filenames(self):
        self.assertEqual(infer_company("Alphabet_10k_2025.pdf"), "Alphabet/Google")
        self.assertEqual(infer_company("Amazon_10k_2025.pdf"), "Amazon")
        self.assertEqual(infer_company("Microsoft_10K_2025.pdf"), "Microsoft")

    def test_fiscal_year_end_metadata(self):
        self.assertEqual(infer_fiscal_year_end("Alphabet/Google"), "2025-12-31")
        self.assertEqual(infer_fiscal_year_end("Amazon"), "2025-12-31")
        self.assertEqual(infer_fiscal_year_end("Microsoft"), "2025-06-30")

    def test_engine_can_reconstruct_fiscal_year_metadata(self):
        self.assertEqual(engine_fiscal_year_end("Amazon"), "2025-12-31")

    def test_pdf_noise_cleanup(self):
        text = "Table of Contents\n2026/4/13 16:18\nRevenue $ 100\n"
        self.assertEqual(clean_pdf_text(text), "Revenue $ 100")

    def test_table_extraction_failure_is_logged_and_falls_back(self):
        class BrokenPage:
            @staticmethod
            def extract_tables():
                raise RuntimeError("synthetic parser failure")

        with self.assertLogs("financial_rag.document_processing", level="WARNING") as logs:
            tables = extract_markdown_tables(BrokenPage(), "amazon.pdf", 42)

        self.assertEqual(tables, [])
        self.assertIn("amazon.pdf page 42", logs.output[0])
        self.assertIn("using page text only", logs.output[0])


if __name__ == "__main__":
    unittest.main()
