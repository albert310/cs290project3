from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import BaselineRAG, RAGConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline text-only RAG system.")
    parser.add_argument("query", help="User question.")
    parser.add_argument("--texts-dir", default="data/sist/texts")
    parser.add_argument("--cache-path", default=".cache/rag_baseline_texts.sqlite")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-context-chars", type=int, default=5600)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--show-think", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = RAGConfig(
        texts_dir=Path(args.texts_dir),
        cache_path=Path(args.cache_path),
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        max_tokens=args.max_tokens,
    )
    rag = BaselineRAG(config).open(rebuild_index=args.rebuild_index)
    try:
        result = rag.answer(args.query)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0

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
