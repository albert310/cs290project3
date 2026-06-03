from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import UnifiedRAG, UnifiedRAGConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unified SQLite RAG system.")
    parser.add_argument("query", help="User question.")
    parser.add_argument("--db-path", default="data/rag/knowledge.sqlite")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=7200)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--llm-query-keywords", action="store_true")
    parser.add_argument("--query-keyword-max-tokens", type=int, default=256)
    parser.add_argument("--query-keyword-max-terms", type=int, default=12)
    parser.add_argument("--query-keyword-thinking", action="store_true")
    parser.add_argument("--iterative-search", action="store_true")
    parser.add_argument("--max-search-steps", type=int, default=5)
    parser.add_argument("--rollout-decision-max-tokens", type=int, default=512)
    parser.add_argument("--rollout-decision-thinking", action="store_true")
    parser.add_argument("--rollout-hits-per-step", type=int, default=5)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--show-keywords", action="store_true")
    parser.add_argument("--show-rollout", action="store_true")
    parser.add_argument("--show-think", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = UnifiedRAGConfig(
        db_path=Path(args.db_path),
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        max_tokens=args.max_tokens,
        enable_llm_query_keywords=args.llm_query_keywords,
        query_keyword_max_tokens=args.query_keyword_max_tokens,
        query_keyword_max_terms=args.query_keyword_max_terms,
        query_keyword_enable_thinking=args.query_keyword_thinking,
        enable_iterative_search=args.iterative_search,
        max_search_steps=args.max_search_steps,
        rollout_decision_max_tokens=args.rollout_decision_max_tokens,
        rollout_decision_enable_thinking=args.rollout_decision_thinking,
        rollout_hits_per_step=args.rollout_hits_per_step,
    )
    rag = UnifiedRAG(config).open()
    try:
        result = rag.answer(args.query)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0

        if args.show_keywords:
            print("--- query keywords ---")
            print(json.dumps(result.query_keywords, ensure_ascii=False))
            print("search_query:", result.search_query or result.query)
            if result.query_keyword_error:
                print("error:", result.query_keyword_error)
            print()

        if args.show_rollout:
            print("--- search rollout ---")
            print(json.dumps(result.search_rollout, ensure_ascii=False, indent=2))
            print()

        print(result.answer)

        if args.show_think and result.think:
            print("\n--- think ---")
            print(result.think)
        if args.show_context:
            print("\n--- retrieved context ---")
            for index, hit in enumerate(result.hits, start=1):
                print(f"[{index}] rank={hit.rank:.4f} source={hit.source_id}")
                print(hit.snippet)
                print()
        return 0
    finally:
        rag.close()


if __name__ == "__main__":
    raise SystemExit(main())
