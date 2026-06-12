from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.dense_index import build_dense_index
from rag.unified_index import resolve_db_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the dense embedding index for the unified RAG database.")
    parser.add_argument("--db-path", default="data/rag/knowledge.sqlite")
    parser.add_argument("--index-dir", default=".cache/dense_rag")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--embedding-model", default="qwen3-embedding-4b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-max-chars", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = build_dense_index(
        db_path=resolve_db_path(Path(args.db_path)),
        index_dir=Path(args.index_dir),
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        batch_size=max(1, args.batch_size),
        text_max_chars=max(200, args.text_max_chars),
        resume=args.resume,
        timeout=args.timeout,
    )
    print("dense index metadata:")
    for key, value in metadata.to_dict().items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
