"""PDF extraction, table preservation, and filing metadata helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document

from .filings import FilingMetadata, inspect_filing


TABLE_MARKERS = (
    "consolidated balance sheets",
    "consolidated statements of income",
    "consolidated statements of operations",
    "segment information",
    "net sales by groups",
    "revenue, classified by significant product",
    "segment revenue",
    "operating income",
    "cash and cash equivalents",
    "effective tax rate",
    "federal statutory rate",
    "provision for income taxes",
    "income taxes paid",
    "cash paid for income taxes",
    "purchases of property and equipment",
    "additions to property and equipment",
)


def infer_company(filename: str) -> str:
    name = filename.lower()
    if "alphabet" in name or "google" in name:
        return "Alphabet/Google"
    if "amazon" in name:
        return "Amazon"
    if "microsoft" in name:
        return "Microsoft"
    return Path(filename).stem


def infer_fiscal_year_end(company: str) -> str:
    if company == "Microsoft":
        return "2025-06-30"
    if company in {"Alphabet/Google", "Amazon"}:
        return "2025-12-31"
    return ""


def clean_pdf_text(text: str) -> str:
    cleaned = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if re.match(r"^\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}", stripped):
            continue
        if re.match(r"^第\d+/\d+", stripped):
            continue
        if stripped == "Table of Contents":
            continue
        if re.match(r"^https://www\.sec\.gov/Archives/", stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _clean_cell(value) -> str:
    return " ".join(str(value or "").split())


def table_to_markdown(table) -> str:
    rows = [[_clean_cell(cell) for cell in row] for row in table]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def looks_like_table_page(text: str) -> bool:
    lower = text.lower()
    marker_match = any(marker in lower for marker in TABLE_MARKERS)
    numeric_density = sum(character.isdigit() for character in text)
    return marker_match and numeric_density >= 40


def load_pdf_documents(
    pdf_path: Path,
    filing: FilingMetadata | None = None,
) -> tuple[list[Document], list[Document]]:
    """Extract a verified filing into page documents and table supplements."""
    filing = filing or inspect_filing(pdf_path)
    company = filing.company
    fiscal_year_end = filing.fiscal_year_end
    page_documents: list[Document] = []
    table_pages: list[Document] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text(layout=True) or page.extract_text() or ""
            text = clean_pdf_text(raw_text)
            metadata = {
                "source_file": pdf_path.name,
                "source": str(pdf_path),
                "company": company,
                "fiscal_year_end": fiscal_year_end,
                "page_number": page_number,
                "doc_type": "page_text",
            }
            page_documents.append(Document(page_content=text, metadata=metadata))

            if looks_like_table_page(text):
                table_blocks = []
                try:
                    for table_index, table in enumerate(page.extract_tables(), start=1):
                        markdown = table_to_markdown(table)
                        if markdown:
                            table_blocks.append(f"Table {table_index}:\n{markdown}")
                except Exception:
                    table_blocks = []
                table_content = text
                if table_blocks:
                    table_content += "\n\n" + "\n\n".join(table_blocks)
                table_metadata = dict(metadata)
                table_metadata["doc_type"] = "table_page"
                table_pages.append(Document(page_content=table_content, metadata=table_metadata))
    return page_documents, table_pages
