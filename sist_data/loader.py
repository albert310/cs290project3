from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence


DEFAULT_DATA_ROOT = Path(os.environ.get("SIST_DATA_DIR", "data/sist"))


@dataclass(frozen=True)
class DatasetSummary:
    root: Path
    jsonl_tables: Dict[str, int]
    raw_file_count: int
    text_file_count: int
    raw_extensions: Dict[str, int]


class SISTDataset:
    """Loader for the provided ShanghaiTech/SIST dataset.

    The dataset root is expected to contain:
      - jsonl/*.jsonl: structured records and RAG chunks
      - raw/*: original crawled HTML/PDF/etc. files
      - texts/*: extracted plain text files referenced by documents.jsonl
    """

    def __init__(self, root: os.PathLike[str] | str = DEFAULT_DATA_ROOT) -> None:
        self.root = Path(root)
        self.jsonl_dir = self.root / "jsonl"
        self.raw_dir = self.root / "raw"
        self.texts_dir = self.root / "texts"
        self._documents_by_id: Optional[Dict[int, Dict[str, Any]]] = None

    def require_exists(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.jsonl_dir.exists():
            raise FileNotFoundError(f"Missing JSONL directory: {self.jsonl_dir}")

    def table_names(self) -> List[str]:
        self.require_exists()
        return sorted(path.stem for path in self.jsonl_dir.glob("*.jsonl"))

    def iter_jsonl(self, table: str) -> Iterator[Dict[str, Any]]:
        path = self._table_path(table)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc

    def load_jsonl(self, table: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for row in self.iter_jsonl(table):
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
        return rows

    def iter_chunks(
        self,
        *,
        categories: Optional[Sequence[str]] = None,
        min_chars: int = 1,
        include_document: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        category_set = set(categories) if categories else None
        documents = self.documents_by_id() if include_document else {}

        for chunk in self.iter_jsonl("chunks"):
            text = chunk.get("text") or ""
            if len(text) < min_chars:
                continue
            if category_set and chunk.get("category") not in category_set:
                continue

            if include_document:
                chunk = dict(chunk)
                chunk["document"] = documents.get(chunk.get("document_id"))
            yield chunk

    def iter_rag_records(
        self,
        *,
        categories: Optional[Sequence[str]] = None,
        min_chars: int = 20,
    ) -> Iterator[Dict[str, Any]]:
        """Yield normalized records ready for embedding/indexing."""

        for chunk in self.iter_chunks(categories=categories, min_chars=min_chars):
            yield {
                "id": f"chunk:{chunk.get('id')}",
                "text": chunk.get("text") or "",
                "metadata": {
                    "chunk_id": chunk.get("id"),
                    "document_id": chunk.get("document_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "title": chunk.get("title"),
                    "url": chunk.get("url"),
                    "category": chunk.get("category"),
                    "char_count": chunk.get("char_count"),
                },
            }

    def documents_by_id(self) -> Dict[int, Dict[str, Any]]:
        if self._documents_by_id is None:
            self._documents_by_id = {
                int(row["id"]): row
                for row in self.iter_jsonl("documents")
                if row.get("id") is not None
            }
        return self._documents_by_id

    def get_document(self, document_id: int) -> Optional[Dict[str, Any]]:
        return self.documents_by_id().get(int(document_id))

    def read_document_text(self, document_id: int) -> str:
        document = self.get_document(document_id)
        if not document:
            raise KeyError(f"Unknown document_id: {document_id}")
        text_path = document.get("text_path")
        if not text_path:
            raise ValueError(f"Document {document_id} has no text_path")
        path = self.resolve_data_path(text_path)
        return path.read_text(encoding="utf-8", errors="replace")

    def resolve_data_path(self, relative_path: str) -> Path:
        normalized = relative_path.replace("\\", "/")
        return self.root / normalized

    def summary(self) -> DatasetSummary:
        self.require_exists()
        table_counts = {name: self.count_rows(name) for name in self.table_names()}
        raw_extensions: Dict[str, int] = {}
        raw_file_count = 0
        if self.raw_dir.exists():
            for path in self.raw_dir.rglob("*"):
                if not path.is_file():
                    continue
                raw_file_count += 1
                ext = path.suffix.lower().lstrip(".") or "<none>"
                raw_extensions[ext] = raw_extensions.get(ext, 0) + 1
        text_file_count = sum(1 for path in self.texts_dir.rglob("*") if path.is_file()) if self.texts_dir.exists() else 0
        return DatasetSummary(
            root=self.root,
            jsonl_tables=table_counts,
            raw_file_count=raw_file_count,
            text_file_count=text_file_count,
            raw_extensions=dict(sorted(raw_extensions.items(), key=lambda item: (-item[1], item[0]))),
        )

    def count_rows(self, table: str) -> int:
        return sum(1 for _ in self.iter_jsonl(table))

    def _table_path(self, table: str) -> Path:
        name = table[:-6] if table.endswith(".jsonl") else table
        path = self.jsonl_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Unknown JSONL table: {table} ({path})")
        return path


def _print_summary(dataset: SISTDataset) -> None:
    summary = dataset.summary()
    print(f"root: {summary.root}")
    print("jsonl tables:")
    for table, count in summary.jsonl_tables.items():
        print(f"  {table}: {count}")
    print(f"raw files: {summary.raw_file_count}")
    print(f"text files: {summary.text_file_count}")
    print("raw extensions:")
    for ext, count in summary.raw_extensions.items():
        print(f"  {ext}: {count}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and sample the SIST dataset.")
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--table", default=None, help="JSONL table to sample, e.g. chunks or documents.")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--rag-sample", action="store_true")
    args = parser.parse_args(argv)

    dataset = SISTDataset(args.root)
    if args.summary or (not args.table and not args.rag_sample):
        _print_summary(dataset)

    if args.table:
        for row in dataset.load_jsonl(args.table, limit=args.limit):
            print(json.dumps(row, ensure_ascii=False))

    if args.rag_sample:
        for row in dataset.iter_rag_records(min_chars=20):
            print(json.dumps(row, ensure_ascii=False))
            args.limit -= 1
            if args.limit <= 0:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
