"""Run the public question bank and export auditable CSV and Markdown answers."""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from financial_rag import DEFAULT_LLM_MODEL, FinancialRAG, RAGConfig, validate_model
from financial_rag.filings import validate_filing_set
from financial_rag.logging_config import configure_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = PROJECT_ROOT / "evaluation" / "question_bank.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation_results.csv"
PDF_DIR = PROJECT_ROOT / "data" / "10k"
CACHE_DIR = PROJECT_ROOT / "cache" / "faiss"
HOLDOUT_STAGE = "Final unseen holdout"
EVALUATION_STAGES = (
    "Development",
    "Hidden generalization",
    HOLDOUT_STAGE,
    "Cross-group benchmark",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_BANK)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every non-holdout question.",
    )
    selection.add_argument(
        "--questions",
        help="Comma-separated question IDs, for example DEV-001,CLASS-010.",
    )
    selection.add_argument("--stage", choices=EVALUATION_STAGES)
    parser.add_argument("--limit", type=int, help="Run only the first N selected questions.")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--acknowledge-holdout",
        action="store_true",
        help="Required when selecting protected holdout questions.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable diagnostic debug logs.")
    return parser.parse_args()


def load_questions(
    path: Path,
    requested: str | None,
    stage: str | None,
    limit: int | None,
) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if requested:
        ids = {item.strip().upper() for item in requested.split(",") if item.strip()}
        rows = [row for row in rows if row["question_id"].upper() in ids]
        missing = ids - {row["question_id"].upper() for row in rows}
        if missing:
            raise ValueError("Question ID(s) not found: " + ", ".join(sorted(missing)))
    elif stage:
        rows = [row for row in rows if row["evaluation_stage"] == stage]
    else:
        rows = [row for row in rows if row["evaluation_stage"] != HOLDOUT_STAGE]
    return rows[:limit] if limit else rows


def require_holdout_acknowledgement(rows: list[dict], acknowledged: bool) -> None:
    """Prevent accidental execution of the post-freeze holdout."""
    contains_holdout = any(row["evaluation_stage"] == HOLDOUT_STAGE for row in rows)
    if contains_holdout and not acknowledged:
        raise ValueError(
            "Protected holdout selected. Freeze and commit the code first, then rerun "
            "with --acknowledge-holdout. Do not tune the code on these results."
        )


def repository_state() -> tuple[str, bool]:
    """Return the commit and dirty state recorded with every evaluation row."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Evaluation requires a Git repository with a readable HEAD.") from exc
    return commit, dirty


def require_api_key() -> str:
    configured = os.environ.get("OPENAI_API_KEY", "").strip()
    if configured:
        return configured
    key = getpass.getpass("OpenAI API key: ").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY is required.")
    return key


def write_markdown(path: Path, rows: list[dict], model: str) -> None:
    lines = [
        "# Financial 10-K RAG Evaluation Results",
        "",
        f"Model: `{model}`  ",
        f"Completed: {sum(not row['error'] for row in rows)} / {len(rows)}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
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
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    questions = load_questions(
        args.question_bank,
        args.questions,
        args.stage,
        args.limit,
    )
    if not questions:
        raise ValueError("No questions were selected.")
    require_holdout_acknowledgement(questions, args.acknowledge_holdout)
    git_commit, git_dirty = repository_state()
    if any(row["evaluation_stage"] == HOLDOUT_STAGE for row in questions) and git_dirty:
        raise ValueError("Protected holdout requires a clean, committed Git worktree.")
    run_timestamp_utc = datetime.now(UTC).isoformat()

    api_key = require_api_key()
    validate_model(args.model, api_key=api_key)
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    validate_filing_set(pdf_paths)

    engine = FinancialRAG(RAGConfig(llm_model=args.model), api_key=api_key)
    dimensions = engine.validate_embedding_credentials()
    logger.info("Embedding credentials valid: %s dimensions", dimensions)
    stats = engine.build_or_load(pdf_paths, CACHE_DIR, rebuild=args.rebuild_index)
    logger.info(
        "Index %s: %s documents",
        "loaded from cache" if stats.get("cache_hit") else "built",
        stats.get("indexed_documents", 0),
    )

    output_rows = []
    for number, question_row in enumerate(questions, start=1):
        question_id = question_row["question_id"]
        logger.info("Evaluating [%s/%s] %s", number, len(questions), question_id)
        result = {}
        error = ""
        try:
            result = engine.answer(question_row["question"])
        except Exception as exc:  # Keep the remaining batch auditable.
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Evaluation question %s failed", question_id)
        output_rows.append(
            {
                **question_row,
                "model": args.model,
                "embedding_model": engine.config.embedding_model,
                "run_timestamp_utc": run_timestamp_utc,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
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
            }
        )

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
