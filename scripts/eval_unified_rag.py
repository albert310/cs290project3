from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import UnifiedRAG, UnifiedRAGConfig
from scripts.evaluate_testset import as_json, read_jsonl, score_answer, validate_records


OUTPUT_COLUMNS = [
    "id",
    "query",
    "search_query",
    "llm_query_keywords",
    "llm_query_keyword_raw",
    "llm_query_keyword_error",
    "search_rollout",
    "answer_verification",
    "gt_answer",
    "category",
    "question_type",
    "eval_type",
    "eval_targets",
    "source_urls",
    "sys_resp_before_opt",
    "sys_resp_after_opt",
    "is_correct_before_opt",
    "is_correct_after_opt",
    "latency_sec",
    "retrieved_sources",
    "retrieved_snippets",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the unified SQLite RAG database on the web-verified test set.")
    parser.add_argument("--testset", default="eval/testset_web_verified.jsonl")
    parser.add_argument("--output-csv", default="eval/unified_rag_after_opt.csv")
    parser.add_argument("--db-path", default="data/rag/knowledge.sqlite")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=7200)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--llm-query-keywords", dest="llm_query_keywords", action="store_true", default=True)
    parser.add_argument("--no-llm-query-keywords", dest="llm_query_keywords", action="store_false")
    parser.add_argument("--query-keyword-max-tokens", type=int, default=256)
    parser.add_argument("--query-keyword-max-terms", type=int, default=12)
    parser.add_argument("--query-keyword-thinking", action="store_true")
    parser.add_argument("--iterative-search", action="store_true")
    parser.add_argument("--max-search-steps", type=int, default=5)
    parser.add_argument("--rollout-decision-max-tokens", type=int, default=512)
    parser.add_argument("--rollout-decision-thinking", action="store_true")
    parser.add_argument("--rollout-hits-per-step", type=int, default=5)
    parser.add_argument("--verify-answer", action="store_true")
    parser.add_argument("--verification-keyword-max-tokens", type=int, default=256)
    parser.add_argument("--verification-keyword-max-terms", type=int, default=10)
    parser.add_argument("--verification-keyword-thinking", action="store_true")
    parser.add_argument("--verification-hits", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args(argv)


def record_to_row(
    item: Mapping[str, Any],
    answer: str,
    latency_sec: float,
    hits: Sequence[Any],
    *,
    search_query: str = "",
    query_keywords: Sequence[str] = (),
    query_keyword_raw: str = "",
    query_keyword_error: str = "",
    search_rollout: Sequence[Mapping[str, Any]] = (),
    answer_verification: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    eval_spec = item["eval"]
    is_correct = int(score_answer(answer, eval_spec))
    return {
        "id": item["id"],
        "query": item["query"],
        "search_query": search_query or item["query"],
        "llm_query_keywords": as_json(list(query_keywords)),
        "llm_query_keyword_raw": query_keyword_raw,
        "llm_query_keyword_error": query_keyword_error,
        "search_rollout": as_json(list(search_rollout)),
        "answer_verification": as_json(dict(answer_verification or {})),
        "gt_answer": item["gt_answer"],
        "category": item["category"],
        "question_type": item["question_type"],
        "eval_type": eval_spec["type"],
        "eval_targets": as_json(eval_spec["targets"]),
        "source_urls": as_json(item["source_urls"]),
        "sys_resp_before_opt": "",
        "sys_resp_after_opt": answer,
        "is_correct_before_opt": "",
        "is_correct_after_opt": is_correct,
        "latency_sec": f"{latency_sec:.3f}",
        "retrieved_sources": as_json([hit.source_id for hit in hits]),
        "retrieved_snippets": as_json([hit.snippet for hit in hits]),
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = read_jsonl(Path(args.testset))
    validate_records(records)
    selected = records[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]

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
        enable_answer_verification=args.verify_answer,
        verification_keyword_max_tokens=args.verification_keyword_max_tokens,
        verification_keyword_max_terms=args.verification_keyword_max_terms,
        verification_keyword_enable_thinking=args.verification_keyword_thinking,
        verification_hits=args.verification_hits,
    )
    rag = UnifiedRAG(config).open()
    output_path = Path(args.output_csv)
    rows: List[Dict[str, Any]] = []
    correct = 0
    try:
        total = len(selected)
        print("unified index stats:", json.dumps(rag.index.stats(), ensure_ascii=False, sort_keys=True)[:1000])
        for index, item in enumerate(selected, start=1):
            started = time.perf_counter()
            try:
                result = rag.answer(str(item["query"]))
                answer = result.answer
                hits = result.hits
                search_query = result.search_query
                query_keywords = result.query_keywords
                query_keyword_raw = result.query_keyword_raw
                query_keyword_error = result.query_keyword_error
                search_rollout = result.search_rollout
                answer_verification = result.answer_verification
            except Exception as exc:
                answer = f"ERROR: {type(exc).__name__}: {exc}"
                hits = []
                search_query = str(item["query"])
                query_keywords = []
                query_keyword_raw = ""
                query_keyword_error = f"{type(exc).__name__}: {exc}"
                search_rollout = []
                answer_verification = {}
            latency = time.perf_counter() - started
            row = record_to_row(
                item,
                answer,
                latency,
                hits,
                search_query=search_query,
                query_keywords=query_keywords,
                query_keyword_raw=query_keyword_raw,
                query_keyword_error=query_keyword_error,
                search_rollout=search_rollout,
                answer_verification=answer_verification,
            )
            correct += int(row["is_correct_after_opt"])
            rows.append(row)
            write_rows(output_path, rows)
            print(
                f"[{index}/{total}] {item['id']} "
                f"correct={row['is_correct_after_opt']} latency={latency:.2f}s"
            )
        accuracy = correct / len(rows) if rows else 0.0
        print(f"unified RAG accuracy: {correct}/{len(rows)} = {accuracy:.3f}")
        print(f"wrote {output_path}")
        return 0
    finally:
        rag.close()


if __name__ == "__main__":
    raise SystemExit(main())
