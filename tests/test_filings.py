from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from financial_rag.filings import inspect_filing, validate_filing_set


class FilingValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.pdf_dir = self.root / "data" / "10k"

    def test_included_filings_are_identified_from_pdf_content(self) -> None:
        filings = validate_filing_set(sorted(self.pdf_dir.glob("*.pdf")))
        by_company = {metadata.company: metadata for metadata in filings.values()}

        self.assertEqual(
            set(by_company),
            {"Alphabet/Google", "Amazon", "Microsoft"},
        )
        self.assertEqual(by_company["Microsoft"].fiscal_year_end, "2025-06-30")
        self.assertEqual(by_company["Amazon"].fiscal_year_end, "2025-12-31")

    def test_filename_does_not_override_pdf_identity(self) -> None:
        amazon = self.pdf_dir / "Amazon_10k_2025.pdf"
        metadata = inspect_filing(amazon)
        self.assertEqual(metadata.company, "Amazon")

    @patch("financial_rag.filings._cover_text")
    def test_rejects_wrong_reporting_period(self, cover_text) -> None:
        cover_text.return_value = (
            "FORM 10-K AMAZON.COM, INC. "
            "For the fiscal year ended December 31, 2024"
        )
        with self.assertRaisesRegex(ValueError, "not the supported Amazon 2025 filing"):
            inspect_filing(self.pdf_dir / "Amazon_10k_2025.pdf")

    @patch("financial_rag.filings.inspect_filing")
    def test_rejects_duplicate_issuer_set(self, inspect) -> None:
        from financial_rag.filings import FilingMetadata

        inspect.return_value = FilingMetadata("Amazon", "2025-12-31", "source.pdf")
        paths = [
            self.pdf_dir / "Alphabet_10k_2025.pdf",
            self.pdf_dir / "Amazon_10k_2025.pdf",
            self.pdf_dir / "Microsoft_10K_2025.pdf",
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate Amazon filings"):
            validate_filing_set(paths)


if __name__ == "__main__":
    unittest.main()
