<div align="center">

# Financial 10-K RAG Assistant

[English](README.md) | [简体中文](README_CN.md)

**An evidence-grounded AI research assistant for analyzing the 2025 Form 10-K filings of Alphabet, Amazon, and Microsoft.**

[![CI](https://github.com/junweilin1009-jpg/financial-10k-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/junweilin1009-jpg/financial-10k-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-42%20offline-2EA44F)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/vector%20store-FAISS-0467DF)

[Product](#product-walkthrough) · [Architecture](#system-architecture) · [Engineering](#engineering-highlights) · [Evaluation](#evaluation-and-results) · [Run locally](#run-it-locally) · [Interview guide](docs/portfolio_interview_guide.md)

</div>

![Financial 10-K RAG Streamlit product preview](docs/images/app_preview.jpg)

| Source coverage | Evaluation bank | Offline tests | CI matrix |
|---:|---:|---:|---:|
| **3 filings / 397 pages** | **117 questions** | **42 tests** | **Python 3.11–3.13** |

## Why this project exists

Annual reports contain valuable financial evidence, but finding a precise answer often means searching hundreds of pages, interpreting tables, checking reporting periods, and comparing companies that disclose similar metrics differently.

This project turns those filings into a traceable question-answering workflow:

1. validate the issuer, filing type, and reporting period;
2. retrieve evidence with company-aware and finance-aware rules;
3. generate an answer using only the retrieved context;
4. expose the source PDF, page number, evidence preview, and runtime metadata.

The result is not a generic chatbot. It is a focused RAG system designed to make financial answers easier to inspect and harder to invent.

## Product walkthrough

The Streamlit interface accepts the three included filings or verified replacement copies. It builds or loads a local FAISS index, keeps a citation trail for every answer, preserves the conversation, and exports the session as CSV.

Example questions include:

- **Direct fact:** What was Microsoft's Productivity and Business Processes revenue in fiscal 2024?
- **Comparison:** Compare capital expenditures across Alphabet, Amazon, and Microsoft.
- **Risk analysis:** Identify one material AI-related risk for each company and cite the evidence.
- **Adversarial check:** Ask about a fact that the filings do not support and verify that the system states the evidence boundary.

Each completed answer can display:

```text
Answer
├── model, retrieval strategy, latency, and token usage
└── retrieved sources
    ├── company and source filename
    ├── PDF page number and document type
    └── evidence preview
```

## What makes it portfolio-ready

This repository goes beyond a notebook demonstration. The same tested Python package powers the website, terminal app, Colab workflow, and batch evaluation runner.

| Area | Production-style decision | Why it matters |
|---|---|---|
| Input safety | Validates PDF content, issuer, filing type, and fiscal period | Prevents mislabeled or wrong-period documents from silently entering the index |
| Retrieval | Routes named companies and supplements table-dense pages | Gives comparison and numeric questions evidence for every requested issuer |
| Evidence boundary | Separates generated answers from structured source records | Makes citations inspectable and avoids treating model prose as evidence |
| Cache safety | Stores native FAISS data plus JSON metadata with a source/config fingerprint | Reuses expensive embeddings without deserializing an unsafe Python pickle |
| Reproducibility | Records commit SHA, dirty state, and UTC time for evaluation runs | Ties reported results to an exact code version |
| Quality gates | Lint, formatting, 42 deterministic tests, and installed-CLI smoke checks | Catches regressions without spending API credits |
| Holdout protection | Requires an explicit flag and a clean Git tree for the 15-question holdout | Reduces accidental tuning on the final evaluation set |

## System architecture

![Final RAG architecture](docs/images/architecture.png)

The workflow has four clear stages: document validation and parsing, fingerprinted indexing, finance-aware retrieval, and evidence-constrained generation. One reusable package under `src/financial_rag/` is shared by every interface.

For component boundaries, cache lifecycle, error handling, and design trade-offs, read the [architecture document](docs/architecture.md) and [technical note](docs/tech_note.md).

## Engineering highlights

### Finance-aware retrieval

Generic similarity search was not enough for multi-company 10-K analysis. The final retriever adds company routing, table-page supplements, exact-caption signals, qualitative/risk expansion, deduplication, and context limits. These are reusable evidence rules—not hard-coded answers to benchmark questions.

### Question-driven iteration

![Question-driven iterative optimization](docs/images/question_driven_iteration.png)

The team repeatedly ran diverse questions, reviewed the answers, classified failures, converted recurring failures into general retrieval or prompting rules, and then reran regression questions. A separate holdout remains protected for a cleaner post-freeze check.

### Explicit failure policy

The assistant distinguishes **“not found in the retrieved context”** from **“not disclosed in the filing.”** It must also disclose period or definition differences, and it must not present forecasts or investment recommendations as facts contained in a 10-K.

## Evaluation and results

The public question bank covers direct facts, calculations, comparisons, multilingual questions, qualitative analysis, and adversarial prompts.

| Evaluation stage | Questions | Purpose |
|---|---:|---|
| Development | 62 | Core financial QA and known failure categories |
| Hidden generalization | 30 | Broader wording and retrieval variations |
| Final unseen holdout | 15 | Post-freeze generalization check |
| Cross-group benchmark | 10 | Human-reviewed comparison across model choices |

![Five-model comparison](docs/images/model_tradeoff.png)

With the same retrieval pipeline and embeddings, GPT-5.6 Sol received **20/20** on the ten-question human-reviewed benchmark; the other four tested models received **19.5/20**. Luna was approximately twice as fast and one-fifth of the measured cost in this small experiment, illustrating the quality-versus-cost trade-off.

These ten questions influenced later retrieval improvements, so the post-improvement score is regression evidence—not an unbiased accuracy claim. The recorded costs are experimental snapshots, not current pricing promises. See [results and caveats](results/README.md) for the full methodology.

## Run it locally

Python 3.11–3.13 is supported.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Set your API key for the current terminal session. Never commit a real key; `.env` is ignored.

```bash
export OPENAI_API_KEY="your-key-here"   # PowerShell: $env:OPENAI_API_KEY="your-key-here"
```

Start the website:

```bash
streamlit run app/streamlit_app.py
```

Or ask one question from the terminal:

```bash
financial-rag-chat --question "Compare capital expenditures across the three companies."
```

The first live run creates embeddings and may take several minutes. Later runs reuse `cache/faiss/`. API calls are billable, and model access depends on the user's OpenAI account.

### Other ways to use it

- **Interactive terminal:** run `financial-rag-chat`.
- **Google Colab:** open [the Colab notebook](notebooks/financial_10k_rag_colab.ipynb).
- **Batch evaluation:** run `python evaluation/run_evaluation.py --all`.
- **Short smoke evaluation:** run `python evaluation/run_evaluation.py --all --limit 3`.

`--all` excludes the protected holdout. Running the holdout requires `--acknowledge-holdout` and a clean Git worktree.

## Test it without an API key

```bash
pytest
ruff check .
ruff format --check .
```

The deterministic suite covers configuration, filing identity and period checks, PDF cleanup and table fallbacks, company routing, evidence selection, source boundaries, safe cache round trips, credential handling, and holdout safeguards. GitHub Actions repeats these checks on Python 3.11, 3.12, and 3.13.

## Repository map

```text
app/                     Streamlit product interface
data/                    Three course-provided filings and integrity manifest
docs/                    Architecture, technical notes, guides, and visuals
evaluation/              Question bank and reproducible batch runner
notebooks/               Google Colab workflow
results/                 Curated answers and experiment summaries
src/financial_rag/       Reusable RAG package and terminal interface
tests/                   Deterministic regression tests
.github/workflows/       Multi-version CI quality gates
```

## Project context and my contribution

This portfolio release builds on a six-person Johns Hopkins Carey Business School course project. My responsibilities focused on making the source data usable, the evaluation credible, and the final repository ready for public review.

### My responsibilities — Junwei Lin

- **Source-data preparation:** preprocessed and organized three company filings covering 397 PDF pages, supporting consistent downstream parsing and page-level traceability.
- **Evaluation design:** co-developed the staged 117-question bank across factual retrieval, calculations, cross-company comparisons, multilingual prompts, qualitative analysis, and adversarial cases.
- **Quality validation:** reviewed reference answers, filing/page references, and recurring failure cases so retrieval changes could be checked against evidence rather than answer fluency alone.
- **Portfolio release ownership:** defined acceptance criteria for reproducibility, installation, testing, documentation, CI, and recruiter readability; coordinated the final validation and public GitHub release.

These responsibilities connected raw financial documents to an auditable evaluation process and helped turn a course deliverable into a repository that another reader can inspect, run, and discuss in an interview.

| Team member | Primary contribution |
|---|---|
| Zhewei Hu | Financial retrieval optimization, multi-model evaluation, final repository integration |
| Shuai Yuan | OpenAI LLM/embedding experiments, parameter tests, evaluation support |
| **Junwei Lin** | **Source-data preparation, staged evaluation design, evidence validation, portfolio-release ownership** |
| Yuhan Ding | Baseline financial RAG architecture and domain-evidence design |
| Shuying Chen | Error analysis, question-bank review, iterative improvement |
| Qige Wang | Streamlit/Colab workflow, documentation, presentation support |

For a beginner-friendly project explanation, technical interview questions, a STAR story, and verified résumé bullets, use the [portfolio interview guide](docs/portfolio_interview_guide.md).

## Scope, rights, and limitations

- The assistant answers from the three bundled filings only; it is not a live market-data or web-search system.
- Cross-company definitions and fiscal periods may not be directly comparable.
- Forecasts and investment recommendations are not facts in a 10-K.
- Live embedding and generation tests are excluded from CI because they are billable.
- The bundled PDFs are course-provided source material; confirm redistribution rights before broader reuse.
- **No open-source license is currently granted.** Public visibility lets readers review the work, but reuse rights have not yet been defined by the team.

Release checks and remaining rights decisions are tracked in [release readiness](docs/release_readiness.md).
