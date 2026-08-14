"""Question routing and query expansion for financial filing retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from langchain_core.documents import Document

COMPANY_ALIASES = {
    "Alphabet/Google": (
        "alphabet",
        "google",
        "google cloud",
        "other bets",
        "googl",
        "goog",
        "谷歌",
        "字母公司",
        "グーグル",
        "アルファベット",
    ),
    "Amazon": ("amazon", "aws", "amzn", "亚马逊", "亞馬遜", "アマゾン"),
    "Microsoft": (
        "microsoft",
        "azure",
        "intelligent cloud",
        "msft",
        "微软",
        "微軟",
        "マイクロソフト",
    ),
}

QUERY_EXPANSIONS = (
    (
        re.compile(r"\bcash\b|liquidity|现金|現金|efectivo|trésorerie|現金同等物", re.I),
        "cash and cash equivalents marketable securities balance sheets liquidity",
    ),
    (
        re.compile(r"current ratio|current assets|current liabilities", re.I),
        "total current assets total current liabilities consolidated balance sheets",
    ),
    (
        re.compile(r"employee|headcount|workforce|员工|僱員|empleados|employés|従業員", re.I),
        "employees human capital full-time part-time headcount",
    ),
    (
        re.compile(r"chief financial officer|\bcfo\b", re.I),
        "Chief Financial Officer executive officers",
    ),
    (
        re.compile(r"total revenue|consolidated revenue|总收入|總收入", re.I),
        "total revenue consolidated revenues consolidated net sales income statements year ended",
    ),
    (
        re.compile(r"revenue|net sales|收入|营收|營收|売上|ingresos|chiffre d['’]affaires", re.I),
        "revenue net sales segment information year ended",
    ),
    (
        re.compile(
            r"operating income|operating loss|\bop\.?\s*inc\.?\b|operating performance|signed values?|margin|营业利润|營業利潤|営業利益|ingreso operativo|résultat d['’]exploitation",
            re.I,
        ),
        "operating income operating loss revenue segment information signed values year ended",
    ),
    (
        re.compile(r"\bnet income\b|净利润|淨利潤|純利益|beneficio neto|résultat net", re.I),
        "net income consolidated income statements summary results of operations year ended",
    ),
    (
        re.compile(r"azure", re.I),
        "Azure and other cloud services revenue grew growth rate server products and cloud services Intelligent Cloud",
    ),
    (
        re.compile(r"january\s+2026|2026年1月|unrealized gains?|未实现收益|未實現收益", re.I),
        "subsequent event January 2026 recognized unrealized gains non-marketable investments observable transactions fair value",
    ),
    (
        re.compile(r"productivity and business processes|\bpbp\b|\brecast\b|reclassif", re.I),
        "Productivity and Business Processes segment composition prior period segment information recast conform",
    ),
    (
        re.compile(r"china|india|regulat|risk|风险|風險|riesgo|risque", re.I),
        "risk factors regulation licensing foreign ownership international operations legal reputation",
    ),
    (
        re.compile(r"artificial intelligence|(?<![a-z0-9])ai(?![a-z0-9])|人工智能|人工智慧", re.I),
        "artificial intelligence competition investment returns infrastructure operations safety ethics copyright intellectual property security harmful content bias legal liability reputation regulation risk",
    ),
    (
        re.compile(r"earnings per share|\beps\b", re.I),
        "diluted net income per share earnings per share",
    ),
    (
        re.compile(r"effective tax rate|statutory.*tax rate|tax rate.*reconcil", re.I),
        "federal statutory rate effective tax rate reconciliation tax credits foreign earnings",
    ),
    (
        re.compile(r"cash[- ]tax gap|taxes paid.*provision|provision.*taxes paid", re.I),
        "cash paid for income taxes net of refunds provision for income taxes",
    ),
    (
        re.compile(r"reportable.*segment|largest.*segment|segment.*percentage", re.I),
        "reportable segments segment revenue total consolidated revenue",
    ),
    (
        re.compile(r"capital expenditures|\bcapex\b|purchases of property and equipment", re.I),
        "cash flows investing activities purchases additions property and equipment",
    ),
)

FOCUSED_EVIDENCE_QUERIES = (
    (
        re.compile(r"total revenue|consolidated revenue|总收入|總收入", re.I),
        "total revenue consolidated revenues consolidated net sales income statements fiscal year 2025",
    ),
    (
        re.compile(r"azure", re.I),
        "Azure and other cloud services revenue grew growth rate Intelligent Cloud server products and cloud services",
    ),
    (
        re.compile(r"january\s+2026|2026年1月|unrealized gains?|未实现收益|未實現收益", re.I),
        "subsequent event January 2026 recognized approximately unrealized gains non-marketable investments observable transactions",
    ),
    (
        re.compile(r"\bnet income\b|净利润|淨利潤|純利益|beneficio neto|résultat net", re.I),
        "net income consolidated income statements summary results of operations fiscal year 2025",
    ),
    (
        re.compile(r"other bets.*operating|operating.*other bets", re.I | re.S),
        "Other Bets operating income operating loss 2024 2025 segment information",
    ),
    (
        re.compile(r"productivity and business processes|\bpbp\b|\brecast\b|reclassif", re.I),
        "Productivity and Business Processes prior period segment information recast segment composition",
    ),
    (
        re.compile(r"effective tax rate|statutory.*tax rate|tax rate.*reconcil", re.I),
        "federal statutory rate effective tax rate reconciliation tax credits foreign earnings",
    ),
    (
        re.compile(r"cash[- ]tax gap|taxes paid.*provision|provision.*taxes paid", re.I),
        "cash paid for income taxes net of refunds provision for income taxes",
    ),
    (
        re.compile(r"reportable.*segment|largest.*segment|segment.*percentage", re.I),
        "reportable segments segment revenue total consolidated revenue",
    ),
    (
        re.compile(r"capital expenditures|\bcapex\b|purchases of property and equipment", re.I),
        "cash flows investing activities purchases additions property and equipment",
    ),
)

AI_RISK_FOCUS_QUERIES = (
    "artificial intelligence competition rapid change investment returns infrastructure costs operations",
    "artificial intelligence legal liability reputation safety bias harmful content copyright security regulation autonomous agentic actions",
)


def target_companies(question: str) -> list[str]:
    lower = question.lower()
    targets = []
    for company, aliases in COMPANY_ALIASES.items():
        if any(alias in lower for alias in aliases):
            targets.append(company)
    return targets


def is_comparison(question: str, targets: Sequence[str]) -> bool:
    lower = question.lower()
    comparison_terms = (
        "compare",
        "comparison",
        "rank",
        "highest",
        "lowest",
        "which company",
        "which of the three",
        "all three",
        "each company",
        "among",
        "versus",
        " vs ",
        "faster",
        "more than",
        "比较",
        "比較",
        "最高",
        "最低",
        "各公司",
        "comparar",
        "comparer",
    )
    return len(targets) >= 2 or any(term in lower for term in comparison_terms)


def is_exact_financial(question: str) -> bool:
    return bool(
        re.search(
            r"\$|how much|what was|quantif|signed values?|per employee|"
            r"\b(?:percentage|percent|margin|growth|grew|faster|rate|change|increase|decrease|cash|revenue|net sales|income|loss|expense|eps|ratio|op\.?\s*inc\.?)\b|"
            r"多少|百分比|增长|增長|增加|减少|減少|收入|营收|營收|利润|利潤|亏损|虧損|比率|"
            r"porcentaje|ingresos|beneficio|pérdida|margen|"
            r"pourcentage|chiffre d['’]affaires|bénéfice|perte|marge|"
            r"売上|利益|損失|比率",
            question,
            re.I,
        )
    )


def focused_evidence_queries(question: str) -> list[str]:
    return [query for pattern, query in FOCUSED_EVIDENCE_QUERIES if pattern.search(question)]


def is_ai_risk_synthesis(question: str) -> bool:
    has_ai = bool(
        re.search(
            r"artificial intelligence|(?<![a-z0-9])ai(?![a-z0-9])|人工智能|人工智慧",
            question,
            re.I,
        )
    )
    has_risk = bool(re.search(r"risk|风险|風險|riesgo|risque", question, re.I))
    return has_ai and has_risk


def expand_query(question: str) -> str:
    additions = [addition for pattern, addition in QUERY_EXPANSIONS if pattern.search(question)]
    if not additions:
        return question
    return question + "\n" + " ".join(dict.fromkeys(" ".join(additions).split()))


def doc_key(doc: Document) -> tuple:
    return (
        doc.metadata.get("source_file", doc.metadata.get("source", "")),
        doc.metadata.get("page_number", ""),
        doc.metadata.get("doc_type", ""),
        doc.page_content[:120],
    )


def dedupe(documents: Iterable[Document]) -> list[Document]:
    seen = set()
    unique = []
    for document in documents:
        key = doc_key(document)
        if key in seen:
            continue
        seen.add(key)
        unique.append(document)
    return unique
