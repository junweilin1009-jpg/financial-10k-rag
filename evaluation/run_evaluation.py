"""Run the public question bank and export auditable CSV and Markdown answers."""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
from pathlib import Path

from financial_rag import DEFAULT_LLM_MODEL, FinancialRAG, RAGConfig, validate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = PROJECT_ROOT / "evaluation" / "question_bank.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation_results.csv"
PDF_DIR = PROJECT_ROOT / "data" / "10k"
CACHE_DIR = PROJECT_ROOT / "cache" / "faiss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_BANK)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Run every public question.")
    selection.add_argument(
        "--questions",
        help="Comma-separated question IDs, for example DEV-001,CLASS-010.",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N selected questions.")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def load_questions(path: Path, requested: str | None, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if requested:
        ids = {item.strip().upper() for item in requested.split(",") if item.strip()}
        rows = [row for row in rows if row["question_id"].upper() in ids]
        missing = ids - {row["question_id"].upper() for row in rows}
        if missing:
            raise ValueError("Question ID(s) not found: " + ", ".join(sorted(missing)))
    return rows[:limit] if limit else rows


def require_api_key() -> None:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return
    key = getpass.getpass("OpenAI API key: ").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY is required.")
    os.environ["OPENAI_API_KEY"] = key


def write_markdown(path: Path, rows: list[dict], model: str) -> None:
    lines = [
        "# Financial 10-K RAG Evaluation Results",
        "",
        f"Model: `{model}`  ",
        f"Completed: {sum(not row['error'] for row in rows)} / {len(rows)}",
        "",
    ]
    for row in rows:
        lines.extend([
            f"## {row['question_id']} - {row['category']}",
            "",
            "### Question",
            "",
            row["question"],
            "",
            "### Reference answer",
            "",
            row["expected_answer"] or "Not provided.",
            "",
            "### Model answer",
            "",
            row["answer"] or f"Error: {row['error']}",
            "",
            f"- Retrieval: `{row['retrieval_strategy']}`",
            f"- Latency: {row['latency_seconds']} seconds",
            f"- Total tokens: {row['total_tokens']}",
            "",
            "---",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    questions = load_questions(args.question_bank, args.questions, args.limit)
    if not questions:
        raise ValueError("No questions were selected.")

    require_api_key()
    validate_model(args.model)
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if len(pdf_paths) != 3:
        raise ValueError(f"Expected exactly three PDFs in {PDF_DIR}; found {len(pdf_paths)}.")

    engine = FinancialRAG(RAGConfig(llm_model=args.model))
    dimensions = engine.validate_embedding_credentials()
    print(f"Embedding credentials valid: {dimensions} dimensions")
    stats = engine.build_or_load(pdf_paths, CACHE_DIR, rebuild=args.rebuild_index)
    print(
        f"Index {'loaded from cache' if stats.get('cache_hit') else 'built'}: "
        f"{stats.get('indexed_documents', 0)} documents"
    )

    output_rows = []
    for number, question_row in enumerate(questions, start=1):
        question_id = question_row["question_id"]
        print(f"[{number}/{len(questions)}] {question_id}")
        result = {}
        error = ""
        try:
            result = engine.answer(question_row["question"])
        except Exception as exc:  # Keep the remaining batch auditable.
            error = f"{type(exc).__name__}: {exc}"
        output_rows.append({
            **question_row,
            "model": args.model,
            "embedding_model": engine.config.embedding_model,
            "answer": result.get("answer", ""),
            "retrieval_strategy": result.get("retrieval_strategy", ""),
            "target_companies": ", ".join(result.get("target_companies", [])),
            "retrieved_sources": json.dumps(result.get("sources", []), ensure_ascii=False),
            "latency_seconds": result.get("latency_seconds", ""),
            "input_tokens": result.get("input_tokens", ""),
            "output_tokens": result.get("output_tokens", ""),
            "reasoning_tokens": result.get("reasoning_tokens", ""),
            "cached_input_tokens": result.get("cached_input_tokens", ""),
            "total_tokens": result.get("total_tokens", ""),
            "stop_reason": result.get("stop_reason", ""),
            "error": error,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    markdown_path = args.output.with_suffix(".md")
    write_markdown(markdown_path, output_rows, args.model)
    print(f"CSV: {args.output}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()

