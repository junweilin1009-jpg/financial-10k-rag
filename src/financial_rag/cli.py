"""Interactive and one-shot terminal interface for the Financial 10-K RAG."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from .config import DEFAULT_LLM_MODEL, RAGConfig
from .engine import FinancialRAG, validate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Answer one question and exit; omit for an interactive session.")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="OpenAI model ID.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/10k"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/faiss"))
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--no-sources", action="store_true", help="Hide retrieved-source previews.")
    return parser.parse_args()


def require_api_key() -> None:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return
    key = getpass.getpass("OpenAI API key: ").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY is required.")
    os.environ["OPENAI_API_KEY"] = key


def print_result(result: dict, show_sources: bool) -> None:
    print("\nAnswer\n------")
    print(result["answer"])
    print(
        f"\nModel: {result['model']} | Strategy: {result['retrieval_strategy']} | "
        f"Latency: {result['latency_seconds']}s | Tokens: {result['total_tokens']:,}"
    )
    if show_sources and result.get("sources"):
        print("\nRetrieved sources\n-----------------")
        for source in result["sources"]:
            print(
                f"[{source['rank']}] {source['company']} | {source['source_file']} | "
                f"PDF page {source['page_number']} | {source['doc_type']}"
            )
            print(source["preview"].replace("\n", " "))


def main() -> None:
    args = parse_args()
    require_api_key()
    validate_model(args.model)
    pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
    if len(pdf_paths) != 3:
        raise ValueError(f"Expected exactly three PDFs in {args.pdf_dir}; found {len(pdf_paths)}.")

    engine = FinancialRAG(RAGConfig(llm_model=args.model))
    dimensions = engine.validate_embedding_credentials()
    stats = engine.build_or_load(pdf_paths, args.cache_dir, rebuild=args.rebuild_index)
    action = "loaded from cache" if stats.get("cache_hit") else "built from PDFs"
    print(
        f"Ready: {args.model} + {engine.config.embedding_model}; index {action}; "
        f"{stats.get('indexed_documents', 0)} documents; {dimensions} embedding dimensions."
    )

    if args.question:
        print_result(engine.answer(args.question), show_sources=not args.no_sources)
        return

    print("Type a question, or type 'q' to quit.")
    while True:
        try:
            question = input("\nAsk> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue
        try:
            print_result(engine.answer(question), show_sources=not args.no_sources)
        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
