# Source filings

This folder contains the three Form 10-K PDFs used in the course project:

| Company | Reporting period | File |
|---|---|---|
| Alphabet | Year ended December 31, 2025 | `10k/Alphabet_10k_2025.pdf` |
| Amazon | Year ended December 31, 2025 | `10k/Amazon_10k_2025.pdf` |
| Microsoft | Fiscal year ended June 30, 2025 | `10k/Microsoft_10K_2025.pdf` |

The filings were provided as course materials and are included solely to make the educational RAG experiment reproducible. The repository authors do not claim ownership of the filings.

[`manifest.csv`](manifest.csv) records the page count, byte size, and SHA-256 digest of each
bundled file. The test suite verifies file size and digest so an accidental corpus change cannot
silently alter the reported workflow. The manifest is an integrity record, not a grant of data
reuse rights.

The application validates issuer, Form 10-K identity, and reporting period from each PDF's opening pages. Replacement filenames must still contain `Alphabet` or `Google`, `Amazon`, and `Microsoft` so the Streamlit upload interface can group them before validation. The interface requires all three supported 2025 filings at once so issuers or reporting periods are not mixed accidentally.
