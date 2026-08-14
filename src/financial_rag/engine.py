"""OpenAI-based financial RAG engine for Alphabet, Amazon, and Microsoft 10-Ks."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from .config import CACHE_VERSION, PROJECT_COMPANIES, RAGConfig
from .document_processing import clean_pdf_text, infer_fiscal_year_end, load_pdf_documents
from .filings import validate_filing_set
from .generation import format_context, message_text, stop_reason, token_usage
from .index_cache import read_cache, write_cache
from .prompts import SYSTEM_PROMPT
from .retrieval import (
    AI_RISK_FOCUS_QUERIES,
    dedupe,
    expand_query,
    focused_evidence_queries,
    is_ai_risk_synthesis,
    is_comparison,
    is_exact_financial,
    target_companies,
)
from .schemas import AnswerResult, BuildStats, SourceReference

logger = logging.getLogger(__name__)


def list_available_models(api_key: str | None = None) -> list[str]:
    """Return model IDs visible to the supplied OpenAI API key."""
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    return sorted(model.id for model in client.models.list().data)


def validate_model(model: str, api_key: str | None = None) -> str:
    """Verify that the requested OpenAI model is accessible."""
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    return client.models.retrieve(model).id


class FinancialRAG:
    def __init__(
        self,
        config: RAGConfig | None = None,
        api_key: str | None = None,
    ):
        self.config = config or RAGConfig()
        self.vector_store: FAISS | None = None
        self.table_pages: list[Document] = []
        self.evidence_pages: list[Document] = []
        self.build_stats: BuildStats | dict = {}
        openai_api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for both OpenAI embeddings and the answer model."
            )
        self.embedding_backend = "OpenAI Embeddings API"
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            api_key=openai_api_key,
        )
        self.llm_backend = "OpenAI Responses API"
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            api_key=openai_api_key,
            max_completion_tokens=self.config.max_output_tokens,
            reasoning={"effort": self.config.reasoning_effort},
            verbosity=self.config.response_verbosity,
            use_responses_api=True,
            store=False,
        )

    def validate_embedding_credentials(self) -> int:
        """Make one small embedding request before the expensive PDF/index build."""
        vector = self.embeddings.embed_query("Embedding credential check")
        return len(vector)

    def build(self, pdf_paths: Sequence[str | Path]) -> BuildStats:
        started = time.perf_counter()
        page_documents: list[Document] = []
        table_pages: list[Document] = []
        loaded_files = []

        filings = validate_filing_set(pdf_paths)
        for path, filing in filings.items():
            pages, tables = load_pdf_documents(path, filing=filing)
            page_documents.extend(pages)
            table_pages.extend(tables)
            loaded_files.append(path.name)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(page_documents)
        self.table_pages = table_pages
        indexed_documents = chunks + table_pages
        self.vector_store = FAISS.from_documents(indexed_documents, self.embeddings)
        self.build_stats = {
            "files": loaded_files,
            "pages": len(page_documents),
            "text_chunks": len(chunks),
            "table_pages": len(table_pages),
            "indexed_documents": len(indexed_documents),
            "embedding_backend": self.embedding_backend,
            "build_seconds": round(time.perf_counter() - started, 2),
            "config": asdict(self.config),
        }
        return self.build_stats

    def _fingerprint_for_config(
        self,
        pdf_paths: Sequence[str | Path],
        config_payload: dict,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(CACHE_VERSION.encode())
        digest.update(json.dumps(config_payload, sort_keys=True).encode())
        for raw_path in sorted((Path(path).resolve() for path in pdf_paths), key=str):
            digest.update(raw_path.name.encode())
            with raw_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()[:20]

    def _cache_fingerprint(self, pdf_paths: Sequence[str | Path]) -> str:
        # Only settings that change stored vectors belong in the index key.
        # Retrieval counts, LLM/output limits, and context ordering can
        # change without paying to embed the same PDF chunks again.
        index_config = {
            "embedding_provider": "openai",
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }
        return self._fingerprint_for_config(pdf_paths, index_config)

    def build_or_load(
        self,
        pdf_paths: Sequence[str | Path],
        cache_root: str | Path,
        rebuild: bool = False,
    ) -> BuildStats:
        """Load a JSON-backed FAISS cache or build and persist a new one."""
        paths = [Path(path).resolve() for path in pdf_paths]
        # Validate before either a cache hit or a rebuild. Public callers may
        # invoke this method directly instead of going through an interface.
        validate_filing_set(paths)
        fingerprint = self._cache_fingerprint(paths)
        cache_path = Path(cache_root) / fingerprint
        faiss_path = cache_path / "faiss"
        metadata_path = cache_path / "metadata.json"
        index_path = faiss_path / "index.faiss"

        if not rebuild and index_path.is_file() and metadata_path.is_file():
            self.vector_store, self.build_stats, self.table_pages = read_cache(
                cache_path,
                self.embeddings,
            )
            self._refresh_evidence_pages(paths)
            self.build_stats.update(
                {
                    "cache_hit": True,
                    "cache_fingerprint": fingerprint,
                    "cache_path": str(cache_path),
                }
            )
            return self.build_stats

        stats = self.build(paths)
        self._refresh_evidence_pages(paths)
        if self.vector_store is None:
            raise RuntimeError("Vector store was not created during build.")
        write_cache(cache_path, self.vector_store, self.table_pages, stats)
        self.build_stats.update(
            {
                "cache_hit": False,
                "cache_fingerprint": fingerprint,
                "cache_path": str(cache_path),
            }
        )
        return self.build_stats

    @staticmethod
    def _is_priority_evidence_page(text: str, company: str) -> bool:
        """Identify complete pages that contain high-value financial evidence.

        The rules use metric combinations rather than fixed page numbers so the
        retrieval remains useful when a future filing shifts pagination.
        """
        lower = text.lower()
        tax_reconciliation = "federal statutory" in lower and (
            "effective rate" in lower
            or "effective income tax rate" in lower
            or "items accounting for differences" in lower
        )
        cash_tax = (
            "cash paid for income taxes" in lower
            or "income taxes paid, net of refunds" in lower
            or "total cash taxes paid, net of refunds" in lower
        )
        cash_flow_capex = (
            "purchases of property and equipment" in lower
            or "additions to property and equipment" in lower
        ) and ("investing activities" in lower or "cash flows" in lower)
        segment_labels = {
            "Alphabet/Google": ("google services", "google cloud", "other bets"),
            "Amazon": ("north america", "international", "aws"),
            "Microsoft": (
                "productivity and business processes",
                "intelligent cloud",
                "more personal computing",
            ),
        }
        labels = segment_labels.get(company, ())
        complete_segment_table = (
            bool(labels)
            and all(label in lower for label in labels)
            and (
                "segment revenue" in lower
                or "net sales by group" in lower
                or "segment information" in lower
                or ("revenue" in lower and "operating income" in lower)
            )
        )
        return tax_reconciliation or cash_tax or cash_flow_capex or complete_segment_table

    def _refresh_evidence_pages(self, pdf_paths: Sequence[Path]) -> None:
        """Reconstruct a small set of complete pages from the cached chunk index."""
        self.evidence_pages = []
        if self.vector_store is None:
            return
        docstore = getattr(self.vector_store, "docstore", None)
        indexed = getattr(docstore, "_dict", {})
        grouped: dict[tuple[str, int, str], list[str]] = {}
        for doc in indexed.values():
            metadata = doc.metadata
            source = str(metadata.get("source", ""))
            page_number = metadata.get("page_number")
            company = str(metadata.get("company", ""))
            if not source or not page_number or not company:
                continue
            key = (source, int(page_number), company)
            grouped.setdefault(key, []).append(doc.page_content)

        selected: dict[str, list[tuple[int, str]]] = {}
        for (source, page_number, company), chunks in grouped.items():
            combined = "\n".join(chunks)
            if self._is_priority_evidence_page(combined, company):
                selected.setdefault(source, []).append((page_number, company))

        known_paths = {str(path.resolve()): path.resolve() for path in pdf_paths}
        for source, items in selected.items():
            path = known_paths.get(str(Path(source).resolve()), Path(source))
            if not path.exists():
                continue
            try:
                with pdfplumber.open(str(path)) as pdf:
                    for page_number, company in sorted(set(items)):
                        page_index = page_number - 1
                        if not 0 <= page_index < len(pdf.pages):
                            continue
                        page = pdf.pages[page_index]
                        raw_text = page.extract_text(layout=True) or page.extract_text() or ""
                        text = clean_pdf_text(raw_text)
                        self.evidence_pages.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source_file": path.name,
                                    "source": str(path),
                                    "company": company,
                                    "fiscal_year_end": infer_fiscal_year_end(company),
                                    "page_number": page_number,
                                    # Match the stored table-page type so dedupe can
                                    # remove equivalent complete-page duplicates.
                                    "doc_type": "table_page",
                                },
                            )
                        )
            except Exception as exc:
                logger.warning(
                    "Could not refresh priority evidence pages from %s; "
                    "continuing with indexed content: %s",
                    path,
                    exc,
                )
                continue

    def _evidence_supplements(
        self,
        question: str,
        companies: Sequence[str],
    ) -> list[Document]:
        """Select complete evidence pages required by common financial tasks."""
        lower = question.lower()
        needs_tax_rate = bool(
            re.search(
                r"effective tax rate|statutory.*tax rate|tax rate.*reconcil",
                lower,
            )
        )
        needs_cash_tax = bool(
            re.search(
                r"cash[- ]tax gap|taxes paid.*provision|provision.*taxes paid",
                lower,
            )
        )
        needs_segments = bool(
            re.search(
                r"reportable.*segment|largest.*segment|segment.*percentage",
                lower,
            )
        )
        needs_capex = bool(
            re.search(
                r"capital expenditures|\bcapex\b|purchases of property and equipment",
                lower,
            )
        )
        if not any((needs_tax_rate, needs_cash_tax, needs_segments, needs_capex)):
            return []

        selected = []
        for company in companies:
            scored = []
            capex_scored = []
            revenue_scored = []
            for doc in self.evidence_pages:
                if doc.metadata.get("company") != company:
                    continue
                text = doc.page_content.lower()
                score = 0
                if needs_tax_rate:
                    score += 100 * ("federal statutory" in text)
                    score += 100 * ("effective rate" in text or "effective income tax rate" in text)
                if needs_cash_tax:
                    score += 100 * (
                        "cash paid for income taxes" in text
                        or "income taxes paid, net of refunds" in text
                        or "total cash taxes paid, net of refunds" in text
                    )
                    score += 90 * ("provision for income taxes" in text)
                    score += 50 * (
                        "federal statutory" in text
                        and ("effective rate" in text or "effective income tax rate" in text)
                    )
                if needs_segments:
                    segment_labels = {
                        "Alphabet/Google": ("google services", "google cloud", "other bets"),
                        "Amazon": ("north america", "international", "aws"),
                        "Microsoft": (
                            "productivity and business processes",
                            "intelligent cloud",
                            "more personal computing",
                        ),
                    }
                    labels = segment_labels.get(company, ())
                    if labels and all(label in text for label in labels):
                        score += 200
                        score += 20 * min(10, text.count("revenue") + text.count("net sales"))
                        score += 80 * ("operating income" in text)
                        score += 60 * ("total" in text)
                if needs_capex:
                    capex_score = 120 * (
                        "purchases of property and equipment" in text
                        or "additions to property and equipment" in text
                    )
                    capex_score += 80 * ("investing activities" in text or "cash flows" in text)
                    capex_score += 200 * (
                        "consolidated statements of cash flows" in text
                        or "cash flows statements" in text
                    )
                    # A capex/revenue comparison needs a denominator as well as
                    # the cash-flow numerator. Complete segment or income-
                    # statement pages are reliable sources for consolidated
                    # revenue and must survive the context limit.
                    segment_labels = {
                        "Alphabet/Google": ("google services", "google cloud", "other bets"),
                        "Amazon": ("north america", "international", "aws"),
                        "Microsoft": (
                            "productivity and business processes",
                            "intelligent cloud",
                            "more personal computing",
                        ),
                    }
                    labels = segment_labels.get(company, ())
                    revenue_score = 180 * bool(labels and all(label in text for label in labels))
                    revenue_score += 100 * any(
                        marker in text
                        for marker in (
                            "total revenue",
                            "consolidated revenue",
                            "consolidated net sales",
                            "total net sales",
                        )
                    )
                    # Distinguish an actual numeric revenue table from an MD&A
                    # page that merely names all segments and mentions total
                    # revenue in prose.
                    revenue_score += 180 * any(
                        marker in text
                        for marker in (
                            "segment results of operations",
                            "following table presents revenue",
                            "net sales information is as follows",
                            "information on reportable segments and reconciliation",
                        )
                    )
                    revenue_score += 200 * bool(
                        re.search(
                            r"(?:total\s+)?(?:revenues?|net sales)\s+\$\s*[\d,]{4,}",
                            text,
                        )
                    )
                    numeric_score = min(80, sum(character.isdigit() for character in text) // 4)
                    if capex_score:
                        capex_scored.append((capex_score + numeric_score, doc))
                    if revenue_score:
                        revenue_scored.append((revenue_score + numeric_score, doc))
                    score += max(capex_score, revenue_score) + numeric_score
                if score:
                    scored.append((score, doc))
            scored.sort(key=lambda item: item[0], reverse=True)
            if needs_capex:
                # Select by role, not merely the two highest aggregate scores.
                # Otherwise two capex-related pages can displace the required
                # consolidated-revenue denominator page.
                capex_scored.sort(key=lambda item: item[0], reverse=True)
                revenue_scored.sort(key=lambda item: item[0], reverse=True)
                role_docs = []
                if capex_scored:
                    role_docs.append(capex_scored[0][1])
                if revenue_scored:
                    role_docs.append(revenue_scored[0][1])
                if len(role_docs) < 2:
                    for _score, doc in scored:
                        if doc not in role_docs:
                            role_docs.append(doc)
                        if len(role_docs) == 2:
                            break
                selected.extend(role_docs)
                continue
            # These tasks each need two distinct pages per company: paid plus
            # provision for cash tax.
            per_company = 2 if needs_cash_tax else 1
            selected.extend(doc for _score, doc in scored[:per_company])
        return selected

    def _search(
        self, query: str, k: int, company: str | None = None, mmr: bool = False
    ) -> list[Document]:
        if self.vector_store is None:
            raise RuntimeError("Build the vector store before asking questions.")
        metadata_filter = {"company": company} if company else None
        try:
            if mmr:
                return self.vector_store.max_marginal_relevance_search(
                    query,
                    k=k,
                    fetch_k=max(self.config.fetch_k, k * 4),
                    lambda_mult=0.55,
                    filter=metadata_filter,
                )
            return self.vector_store.similarity_search(query, k=k, filter=metadata_filter)
        except (TypeError, ValueError):
            multiplier = 8 if company else 3
            if mmr:
                candidates = self.vector_store.max_marginal_relevance_search(
                    query,
                    k=k * multiplier,
                    fetch_k=max(self.config.fetch_k, k * multiplier * 2),
                    lambda_mult=0.55,
                )
            else:
                candidates = self.vector_store.similarity_search(query, k=k * multiplier)
            if company:
                candidates = [doc for doc in candidates if doc.metadata.get("company") == company]
            return candidates[:k]

    def _table_supplements(self, query: str, companies: Sequence[str]) -> list[Document]:
        query_lower = query.lower()
        query_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", query_lower)
            if len(term) > 2
            and term not in {"what", "which", "from", "with", "that", "this", "year"}
        }
        requested_numbers = {
            match.replace(",", "")
            for match in re.findall(
                r"(?<![\d,.])\d[\d,]*(?:\.\d+)?(?![\d,.])",
                query_lower,
            )
            if not (
                match.replace(",", "").isdigit() and 1900 <= int(match.replace(",", "")) <= 2100
            )
            and len(match.replace(",", "")) >= 4
        }
        requested_segments = {
            "intelligent cloud": ("revenue", "net sales"),
            "google cloud": ("revenue", "net sales"),
            "aws": ("revenue", "net sales"),
            "other bets": ("revenue", "net sales"),
            "productivity and business processes": ("revenue", "net sales"),
            "more personal computing": ("revenue", "net sales"),
        }
        needs_current_ratio_inputs = "current ratio" in query_lower or (
            "current assets" in query_lower and "current liabilities" in query_lower
        )
        needs_revenue = bool(re.search(r"\brevenue\b|\bnet sales\b|\bmargin\b", query_lower))
        needs_operating_income = bool(
            re.search(r"operating income|operating loss|\bmargin\b", query_lower)
        )
        needs_net_income = "net income" in query_lower
        needs_total_revenue = "total revenue" in query_lower
        needs_azure_growth = "azure" in query_lower and bool(
            re.search(r"growth|grew|faster|rate", query_lower)
        )
        needs_subsequent_unrealized_gain = (
            "january 2026" in query_lower and "unrealized gain" in query_lower
        )
        needs_segment_recast = "prior period segment information recast" in query_lower
        supplements = []
        for company in companies:
            candidates = [doc for doc in self.table_pages if doc.metadata.get("company") == company]
            scored = []
            for doc in candidates:
                text_lower = doc.page_content.lower()
                normalized_text = text_lower.replace(",", "")
                text_terms = set(re.findall(r"[a-z0-9]+", text_lower))
                score = len(query_terms & text_terms)
                score += 30 * sum(number in normalized_text for number in requested_numbers)
                if "segment" in query_terms and "segment information" in text_lower:
                    score += 4
                if "cash" in query_terms and "cash and cash equivalents" in text_lower:
                    score += 4
                if "revenue" in query_terms and "revenue" in text_lower:
                    score += 2
                if needs_total_revenue and any(
                    marker in text_lower
                    for marker in (
                        "total revenue",
                        "consolidated revenues",
                        "consolidated net sales",
                    )
                ):
                    score += 50
                if needs_net_income and "net income" in text_lower:
                    score += 15
                    if any(
                        marker in text_lower
                        for marker in (
                            "income statements",
                            "summary results of operations",
                            "cash flows statements",
                        )
                    ):
                        score += 35
                    if "pro forma" in text_lower:
                        score -= 30
                if needs_azure_growth and (
                    "azure and other cloud services revenue grew" in text_lower
                    or "azure and other cloud services revenue growth" in text_lower
                ):
                    score += 60
                if (
                    needs_subsequent_unrealized_gain
                    and "january 2026" in text_lower
                    and "unrealized gains" in text_lower
                    and "non-marketable investments" in text_lower
                ):
                    score += 60
                if (
                    needs_segment_recast
                    and "prior period segment information has been recast" in text_lower
                ):
                    score += 60
                # Prefer complete balance-sheet pages that contain both inputs
                # needed for liquidity calculations. Audit reports often use
                # the phrase "consolidated balance sheets" without containing
                # any of the requested figures and previously outranked them.
                if needs_current_ratio_inputs:
                    has_current_assets = "total current assets" in text_lower
                    has_current_liabilities = "total current liabilities" in text_lower
                    if has_current_assets and has_current_liabilities:
                        score += 40
                    elif has_current_assets or has_current_liabilities:
                        score += 5
                    if "report of independent registered public accounting firm" in text_lower:
                        score -= 10
                # Prefer the numeric table where the requested segment label is
                # immediately followed by its financial row, not a narrative
                # page that merely defines the segment.
                for segment, metrics in requested_segments.items():
                    if segment not in query_lower:
                        continue
                    metric_pattern = "|".join(re.escape(metric) for metric in metrics)
                    if re.search(
                        rf"{re.escape(segment)}[\s\S]{{0,180}}(?:{metric_pattern})",
                        text_lower,
                    ):
                        score += 12
                    # Multi-metric questions need a page containing all inputs,
                    # even when narrative between the segment name and numeric
                    # rows makes the proximity test above too strict.
                    if (
                        segment in text_lower
                        and needs_revenue
                        and needs_operating_income
                        and ("revenue" in text_lower or "net sales" in text_lower)
                        and ("operating income" in text_lower or "operating loss" in text_lower)
                    ):
                        score += 24
                        metric_occurrences = (
                            text_lower.count("revenue")
                            + text_lower.count("net sales")
                            + text_lower.count("operating income")
                            + text_lower.count("operating loss")
                        )
                        score += min(12, metric_occurrences)
                        if re.search(
                            rf"{re.escape(segment)}[\s\S]{{0,600}}"
                            rf"(?:revenue|net sales)[\s\S]{{0,600}}"
                            rf"(?:operating income|operating loss)",
                            text_lower,
                        ):
                            score += 20
                        if re.search(
                            rf"(?m)^\s*{re.escape(segment)}\s*$[\s\S]{{0,300}}"
                            rf"(?:revenue|net sales)[\s\S]{{0,500}}"
                            rf"(?:operating income|operating loss)",
                            text_lower,
                        ):
                            score += 40
                scored.append((score, doc))
            scored.sort(key=lambda item: item[0], reverse=True)
            supplements.extend(
                doc for score, doc in scored[: self.config.table_supplement_k] if score > 0
            )
        return supplements

    @staticmethod
    def _expand_qualitative_pages(
        documents: Sequence[Document],
        per_company_limit: int = 4,
    ) -> list[Document]:
        """Replace selected chunks with a few complete filing pages per company."""
        selected = []
        seen_pages = set()
        company_counts: dict[str, int] = {}
        for doc in documents:
            metadata = doc.metadata
            company = str(metadata.get("company", ""))
            source = str(metadata.get("source", ""))
            page_number = metadata.get("page_number")
            key = (source, page_number)
            if not source or not page_number or key in seen_pages:
                continue
            if company_counts.get(company, 0) >= per_company_limit:
                continue
            selected.append((key, dict(metadata), doc))
            seen_pages.add(key)
            company_counts[company] = company_counts.get(company, 0) + 1

        expanded: dict[tuple, Document] = {}
        by_source: dict[str, list[tuple]] = {}
        for key, metadata, fallback in selected:
            by_source.setdefault(key[0], []).append((key, metadata, fallback))

        for source, items in by_source.items():
            path = Path(source)
            if not path.exists():
                continue
            try:
                with pdfplumber.open(str(path)) as pdf:
                    for key, metadata, _fallback in items:
                        page_index = int(key[1]) - 1
                        if not 0 <= page_index < len(pdf.pages):
                            continue
                        page = pdf.pages[page_index]
                        raw_text = page.extract_text(layout=True) or page.extract_text() or ""
                        metadata["doc_type"] = "full_page_context"
                        expanded[key] = Document(
                            page_content=clean_pdf_text(raw_text),
                            metadata=metadata,
                        )
            except Exception as exc:
                logger.warning(
                    "Could not expand qualitative context from %s; "
                    "continuing with retrieved chunks: %s",
                    path,
                    exc,
                )
                continue

        return [expanded.get(key, fallback) for key, _metadata, fallback in selected]

    def retrieve(self, question: str) -> tuple[list[Document], str, list[str]]:
        targets = target_companies(question)
        comparison = is_comparison(question, targets)
        exact_financial = is_exact_financial(question)
        query = expand_query(question)

        if comparison:
            companies = targets or list(PROJECT_COMPANIES)
            documents = []
            per_company_k = self.config.company_k if exact_financial else self.config.qualitative_k
            risk_synthesis = is_ai_risk_synthesis(question)
            for company in companies:
                if risk_synthesis and not exact_financial:
                    for focus_query in AI_RISK_FOCUS_QUERIES:
                        documents.extend(self._search(focus_query, 2, company=company, mmr=False))
                documents.extend(self._search(query, per_company_k, company=company, mmr=True))
            strategy = "company_balanced_mmr"
            if not exact_financial:
                documents = self._expand_qualitative_pages(documents)
                strategy += "+full_page_context"
                if risk_synthesis:
                    strategy += "+risk_focus"
        elif len(targets) == 1:
            companies = targets
            k = self.config.fact_k if exact_financial else self.config.qualitative_k
            documents = self._search(query, k, company=targets[0], mmr=not exact_financial)
            strategy = "company_filtered_similarity" if exact_financial else "company_filtered_mmr"
        else:
            companies = list(PROJECT_COMPANIES)
            k = self.config.fact_k if exact_financial else self.config.qualitative_k
            documents = self._search(query, k, mmr=not exact_financial)
            strategy = "global_similarity" if exact_financial else "global_mmr"

        if exact_financial:
            focused_documents = []
            for focused_query in focused_evidence_queries(question):
                for company in companies:
                    focused_documents.extend(
                        self._search(focused_query, 2, company=company, mmr=False)
                    )
            supplements = self._table_supplements(query, companies)
            evidence_supplements = self._evidence_supplements(question, companies)
            if focused_documents:
                documents = focused_documents + documents
                strategy += "+focused_evidence"
            if supplements:
                # Put complete table pages before similarity-search chunks.
                # With multi-company questions the context budget can otherwise
                # be consumed before a late company's segment table is reached.
                documents = supplements + documents
                strategy += "+table_supplement"
            if evidence_supplements:
                # Task-complete pages must be first so context truncation cannot
                # discard a late company's required numerator or denominator.
                documents = evidence_supplements + documents
                strategy += "+evidence_pages"

        return dedupe(documents), strategy, companies

    def _format_context(
        self,
        documents: Sequence[Document],
    ) -> tuple[str, list[Document]]:
        return format_context(documents, self.config.max_context_chars)

    @staticmethod
    def _message_text(message) -> str:
        return message_text(message)

    @staticmethod
    def _stop_reason(message) -> str:
        return stop_reason(message)

    @staticmethod
    def _token_usage(message) -> dict[str, int]:
        return token_usage(message)

    def answer(self, question: str) -> AnswerResult:
        started = time.perf_counter()
        documents, strategy, companies = self.retrieve(question)
        context, context_documents = self._format_context(documents)
        if not context_documents:
            raise ValueError("No retrieved evidence fit within the configured context limit.")
        human_message = f"Retrieved filing context:\n\n{context}\n\nQuestion:\n{question}"
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", human_message),
        ]
        response = self.llm.invoke(messages)
        token_usage = self._token_usage(response)
        answer = self._message_text(response)
        stop_reason = self._stop_reason(response)
        continuation_attempts = 0
        while (
            stop_reason == "max_tokens"
            and continuation_attempts < self.config.max_continuation_attempts
        ):
            continuation_attempts += 1
            messages.extend(
                [
                    response,
                    (
                        "human",
                        "Continue exactly where the answer stopped. Do not repeat prior text. "
                        "Finish the requested comparison, conclusion, and citations concisely.",
                    ),
                ]
            )
            response = self.llm.invoke(messages)
            continuation_usage = self._token_usage(response)
            for key in token_usage:
                token_usage[key] += continuation_usage[key]
            continuation = self._message_text(response)
            if continuation:
                answer = answer.rstrip() + "\n" + continuation.lstrip()
            stop_reason = self._stop_reason(response)
        sources: list[SourceReference] = []
        for rank, doc in enumerate(context_documents, start=1):
            metadata = doc.metadata
            sources.append(
                {
                    "rank": rank,
                    "company": metadata.get("company", ""),
                    "source_file": metadata.get("source_file", metadata.get("source", "")),
                    "page_number": metadata.get("page_number", ""),
                    "doc_type": metadata.get("doc_type", ""),
                    "preview": " ".join(doc.page_content.split())[:800],
                }
            )
        result: AnswerResult = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieval_strategy": strategy,
            "target_companies": companies,
            "model": self.config.llm_model,
            "embedding_model": self.config.embedding_model,
            "latency_seconds": round(time.perf_counter() - started, 2),
            "stop_reason": stop_reason,
            "continuation_attempts": continuation_attempts,
            "retrieved_document_count": len(documents),
            "context_document_count": len(context_documents),
            **token_usage,
        }
        return result
