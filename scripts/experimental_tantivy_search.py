#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.tantivy_index import DEFAULT_TANTIVY_INDEX_DIR, build_index, search_tantivy
from retrieval.keyword_search import make_snippet


DEFAULT_DB_PATH = Path("data/rag/knowledge.sqlite")


def print_hits(query: str, hits: Sequence[Any]) -> None:
    print(f"query: {query}")
    print(f"hits: {len(hits)}")
    for index, hit in enumerate(hits, start=1):
        snippet = make_snippet(hit.text, query, max_chars=260).replace("\n", " ")
        print(f"\n[{index}] rank={hit.rank:.3f} type={hit.source_type} category={hit.category}")
        print(f"title: {hit.title}")
        print(f"url: {hit.url}")
        print(f"path: {hit.path}")
        print(f"snippet: {snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experimental Tantivy BM25 retrieval over the unified RAG SQLite DB.")
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_TANTIVY_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=240)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    if args.build or args.rebuild or not (args.index_dir / "build_meta.json").exists():
        build_index(args.db_path, args.index_dir, rebuild=args.rebuild, limit=args.limit, batch_size=args.batch_size)
    if args.query:
        hits = search_tantivy(args.db_path, args.index_dir, args.query, top_k=args.top_k, candidates=args.candidates)
        print_hits(args.query, hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
