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
    parser.add_argument("--retrieval-backend", choices=("sqlite", "tantivy"), default="sqlite")
    parser.add_argument("--tantivy-index-dir", default=".cache/tantivy_rag")
    parser.add_argument("--tantivy-candidates", type=int, default=240)
    parser.add_argument("--lexical-candidates", type=int, default=240)
    parser.add_argument("--structured-candidates", type=int, default=160)
    parser.add_argument("--dense-retrieval", dest="dense_retrieval", action="store_true")
    parser.add_argument("--no-dense-retrieval", dest="dense_retrieval", action="store_false")
    parser.set_defaults(dense_retrieval=True)
    parser.add_argument("--dense-index-dir", default=".cache/dense_rag")
    parser.add_argument("--dense-candidates", type=int, default=160)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--embedding-model", default="qwen3-embedding-4b")
    parser.add_argument("--lexical-rrf-weight", type=float, default=1.0)
    parser.add_argument("--structured-rrf-weight", type=float, default=1.15)
    parser.add_argument("--dense-rrf-weight", type=float, default=1.0)
    parser.add_argument("--hybrid-rrf-k", type=int, default=60)
    parser.add_argument("--neighbor-expansion-window", type=int, default=1)
    parser.add_argument("--neighbor-expansion-limit", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=7200)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--thinking", dest="enable_thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--llm-query-keywords", dest="llm_query_keywords", action="store_true", default=True)
    parser.add_argument("--no-llm-query-keywords", dest="llm_query_keywords", action="store_false")
    parser.add_argument("--query-keyword-max-tokens", type=int, default=256)
    parser.add_argument("--query-keyword-max-terms", type=int, default=12)
    parser.add_argument("--query-keyword-thinking", action="store_true")
    parser.add_argument("--llm-rerank", dest="llm_rerank", action="store_true", default=True)
    parser.add_argument("--no-llm-rerank", dest="llm_rerank", action="store_false")
    parser.add_argument("--llm-rerank-candidates", type=int, default=64)
    parser.add_argument("--llm-rerank-max-tokens", type=int, default=768)
    parser.add_argument("--llm-rerank-thinking", action="store_true")
    parser.add_argument("--llm-rerank-chars-per-hit", type=int, default=500)
    parser.add_argument("--iterative-search", dest="iterative_search", action="store_true", default=True)
    parser.add_argument("--no-iterative-search", dest="iterative_search", action="store_false")
    parser.add_argument("--max-search-steps", type=int, default=5)
    parser.add_argument("--rollout-decision-max-tokens", type=int, default=512)
    parser.add_argument("--rollout-decision-thinking", action="store_true")
    parser.add_argument("--rollout-hits-per-step", type=int, default=5)
    parser.add_argument("--verify-answer", action="store_true")
    parser.add_argument("--verification-keyword-max-tokens", type=int, default=256)
    parser.add_argument("--verification-keyword-max-terms", type=int, default=10)
    parser.add_argument("--verification-keyword-thinking", action="store_true")
    parser.add_argument("--verification-hits", type=int, default=6)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--show-keywords", action="store_true")
    parser.add_argument("--show-rerank", action="store_true")
    parser.add_argument("--show-rollout", action="store_true")
    parser.add_argument("--show-verification", action="store_true")
    parser.add_argument("--show-think", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = UnifiedRAGConfig(
        db_path=Path(args.db_path),
        retrieval_backend=args.retrieval_backend,
        tantivy_index_dir=Path(args.tantivy_index_dir),
        tantivy_candidates=args.tantivy_candidates,
        lexical_candidates=args.lexical_candidates,
        structured_candidates=args.structured_candidates,
        enable_dense_retrieval=args.dense_retrieval,
        dense_index_dir=Path(args.dense_index_dir),
        dense_candidates=args.dense_candidates,
        dense_embedding_base_url=args.embedding_base_url,
        dense_embedding_model=args.embedding_model,
        lexical_rrf_weight=args.lexical_rrf_weight,
        structured_rrf_weight=args.structured_rrf_weight,
        dense_rrf_weight=args.dense_rrf_weight,
        hybrid_rrf_k=args.hybrid_rrf_k,
        neighbor_expansion_window=args.neighbor_expansion_window,
        neighbor_expansion_limit=args.neighbor_expansion_limit,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
        enable_llm_query_keywords=args.llm_query_keywords,
        query_keyword_max_tokens=args.query_keyword_max_tokens,
        query_keyword_max_terms=args.query_keyword_max_terms,
        query_keyword_enable_thinking=args.query_keyword_thinking,
        enable_llm_rerank=args.llm_rerank,
        llm_rerank_candidates=args.llm_rerank_candidates,
        llm_rerank_max_tokens=args.llm_rerank_max_tokens,
        llm_rerank_enable_thinking=args.llm_rerank_thinking,
        llm_rerank_chars_per_hit=args.llm_rerank_chars_per_hit,
        enable_iterative_search=args.iterative_search,
        max_search_steps=args.max_search_steps,
        rollout_decision_max_tokens=args.rollout_decision_max_tokens,
        rollout_decision_enable_thinking=args.rollout_decision_thinking,
        rollout_hits_per_step=args.rollout_hits_per_step,
        enable_answer_verification=args.verify_answer,
        verification_keyword_max_tokens=args.verification_keyword_max_tokens,
        verification_keyword_max_terms=args.verification_keyword_max_terms,
        verification_keyword_enable_thinking=args.verification_keyword_thinking,
        verification_hits=args.verification_hits,
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

        if args.show_rerank:
            print("--- llm rerank ---")
            print(json.dumps(result.llm_rerank, ensure_ascii=False, indent=2))
            print()

        if args.show_rollout:
            print("--- search rollout ---")
            print(json.dumps(result.search_rollout, ensure_ascii=False, indent=2))
            print()

        if args.show_verification:
            print("--- answer verification ---")
            print(json.dumps(result.answer_verification, ensure_ascii=False, indent=2))
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
