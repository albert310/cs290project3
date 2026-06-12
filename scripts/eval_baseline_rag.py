from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import BaselineRAG, RAGConfig
from scripts.evaluate_testset import as_json, read_jsonl, score_answer, validate_records


OUTPUT_COLUMNS = [
    "id",
    "query",
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
    parser = argparse.ArgumentParser(description="Evaluate the baseline text-only RAG on the web-verified test set.")
    parser.add_argument("--testset", default="eval/testset_web_verified.jsonl")
    parser.add_argument("--output-csv", default="eval/baseline_rag_before_opt.csv")
    parser.add_argument("--texts-dir", default="data/sist/texts")
    parser.add_argument("--cache-path", default=".cache/rag_baseline_texts.sqlite")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-context-chars", type=int, default=5600)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--thinking", dest="enable_thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args(argv)


def record_to_row(item: Mapping[str, Any], answer: str, latency_sec: float, hits: Sequence[Any]) -> Dict[str, Any]:
    eval_spec = item["eval"]
    is_correct = int(score_answer(answer, eval_spec))
    return {
        "id": item["id"],
        "query": item["query"],
        "gt_answer": item["gt_answer"],
        "category": item["category"],
        "question_type": item["question_type"],
        "eval_type": eval_spec["type"],
        "eval_targets": as_json(eval_spec["targets"]),
        "source_urls": as_json(item["source_urls"]),
        "sys_resp_before_opt": answer,
        "sys_resp_after_opt": "",
        "is_correct_before_opt": is_correct,
        "is_correct_after_opt": "",
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

    config = RAGConfig(
        texts_dir=Path(args.texts_dir),
        cache_path=Path(args.cache_path),
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
    )
    rag = BaselineRAG(config).open(rebuild_index=args.rebuild_index)
    output_path = Path(args.output_csv)
    rows: List[Dict[str, Any]] = []
    correct = 0
    try:
        total = len(selected)
        for index, item in enumerate(selected, start=1):
            started = time.perf_counter()
            try:
                result = rag.answer(str(item["query"]))
                answer = result.answer
                hits = result.hits
            except Exception as exc:
                answer = f"ERROR: {type(exc).__name__}: {exc}"
                hits = []
            latency = time.perf_counter() - started
            row = record_to_row(item, answer, latency, hits)
            correct += int(row["is_correct_before_opt"])
            rows.append(row)
            write_rows(output_path, rows)
            print(
                f"[{index}/{total}] {item['id']} "
                f"correct={row['is_correct_before_opt']} latency={latency:.2f}s"
            )
        accuracy = correct / len(rows) if rows else 0.0
        print(f"baseline accuracy: {correct}/{len(rows)} = {accuracy:.3f}")
        print(f"wrote {output_path}")
        return 0
    finally:
        rag.close()


if __name__ == "__main__":
    raise SystemExit(main())
