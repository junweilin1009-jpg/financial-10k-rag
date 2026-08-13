"""Validation and metadata for the supported 2025 Form 10-K corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pdfplumber


@dataclass(frozen=True)
class FilingMetadata:
    """Verified identity and reporting-period metadata for one filing."""

    company: str
    fiscal_year_end: str
    source_file: str


EXPECTED_FILINGS = {
    "Alphabet/Google": {
        "issuer_patterns": (r"\balphabet\s+inc\.?\b",),
        "period_pattern": r"fiscal year ended\s+december\s+31,?\s+2025",
        "fiscal_year_end": "2025-12-31",
    },
    "Amazon": {
        "issuer_patterns": (r"\bamazon\.com,?\s+inc\.?\b",),
        "period_pattern": r"fiscal year ended\s+december\s+31,?\s+2025",
        "fiscal_year_end": "2025-12-31",
    },
    "Microsoft": {
        "issuer_patterns": (r"\bmicrosoft\s+corporation\b",),
        "period_pattern": r"fiscal year ended\s+june\s+30,?\s+2025",
        "fiscal_year_end": "2025-06-30",
    },
}


def _cover_text(pdf_path: Path, page_limit: int = 3) -> str:
    """Extract enough opening-page text to validate filing identity and period."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if not pdf.pages:
                raise ValueError("PDF contains no pages.")
            return "\n".join(
                page.extract_text(layout=True) or page.extract_text() or ""
                for page in pdf.pages[:page_limit]
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot read PDF {pdf_path.name}: {exc}") from exc


def inspect_filing(pdf_path: str | Path) -> FilingMetadata:
    """Validate one PDF as a supported issuer's 2025 Form 10-K."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path.name}")

    cover_text = " ".join(_cover_text(path).lower().split())
    if not re.search(r"\bform\s+10-k\b", cover_text):
        raise ValueError(f"{path.name} is not identifiable as a Form 10-K.")

    matches = [
        company
        for company, spec in EXPECTED_FILINGS.items()
        if any(re.search(pattern, cover_text) for pattern in spec["issuer_patterns"])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not identify exactly one supported issuer in {path.name}; "
            f"found {matches or 'none'}."
        )

    company = matches[0]
    spec = EXPECTED_FILINGS[company]
    if not re.search(str(spec["period_pattern"]), cover_text):
        raise ValueError(
            f"{path.name} is not the supported {company} 2025 filing with period end "
            f"{spec['fiscal_year_end']}."
        )
    return FilingMetadata(
        company=company,
        fiscal_year_end=str(spec["fiscal_year_end"]),
        source_file=path.name,
    )


def validate_filing_set(pdf_paths: Sequence[str | Path]) -> dict[Path, FilingMetadata]:
    """Require exactly one verified filing for every supported company."""
    paths = [Path(path).resolve() for path in pdf_paths]
    if len(paths) != len(EXPECTED_FILINGS):
        raise ValueError(
            f"Expected exactly {len(EXPECTED_FILINGS)} PDFs; found {len(paths)}."
        )

    filings: dict[Path, FilingMetadata] = {}
    by_company: dict[str, Path] = {}
    for path in paths:
        metadata = inspect_filing(path)
        if metadata.company in by_company:
            first = by_company[metadata.company]
            raise ValueError(
                f"Duplicate {metadata.company} filings: {first.name}, {path.name}."
            )
        filings[path] = metadata
        by_company[metadata.company] = path

    missing = set(EXPECTED_FILINGS) - set(by_company)
    if missing:
        raise ValueError("Missing filing(s): " + ", ".join(sorted(missing)))
    return filings
