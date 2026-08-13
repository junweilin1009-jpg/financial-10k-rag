# Source filings

This folder contains the three Form 10-K PDFs used in the course project:

| Company | Reporting period | File |
|---|---|---|
| Alphabet | Year ended December 31, 2025 | `10k/Alphabet_10k_2025.pdf` |
| Amazon | Year ended December 31, 2025 | `10k/Amazon_10k_2025.pdf` |
| Microsoft | Fiscal year ended June 30, 2025 | `10k/Microsoft_10K_2025.pdf` |

The filings were provided as course materials and are included solely to make the educational RAG experiment reproducible. The repository authors do not claim ownership of the filings.

The application identifies issuers from filenames. Replacement files must contain `Alphabet` or `Google`, `Amazon`, and `Microsoft` in their respective filenames. The Streamlit interface requires all three replacements at once so data versions are not mixed accidentally.

