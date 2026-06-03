from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from retrieval.keyword_search import make_snippet, tokenize


SPACE_RE = re.compile(r"\s+")
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,}\d+[A-Z]?\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class TextChunk:
    path: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class TextSearchHit:
    rank: float
    path: str
    chunk_index: int
    text: str
    snippet: str

    @property
    def source_id(self) -> str:
        return f"{self.path}#chunk={self.chunk_index}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "path": self.path,
            "chunk_index": self.chunk_index,
            "source_id": self.source_id,
            "snippet": self.snippet,
        }


def normalize_plain_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> Iterator[str]:
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_chars)
        yield text[start:end].strip()
        if end >= length:
            break
        start = max(end - overlap_chars, start + 1)


def chunk_text(text: str, *, max_chars: int = 900, overlap_chars: int = 120, min_chars: int = 80) -> Iterator[str]:
    normalized = normalize_plain_text(text)
    if not normalized:
        return

    paragraphs = [para.strip() for para in re.split(r"\n{2,}", normalized) if para.strip()]
    current: List[str] = []
    current_len = 0

    def flush_current() -> Iterator[str]:
        nonlocal current, current_len
        if current:
            chunk = "\n".join(current).strip()
            if len(chunk) >= min_chars:
                yield chunk
            current = []
            current_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            yield from flush_current()
            for piece in split_long_text(para, max_chars=max_chars, overlap_chars=overlap_chars):
                if len(piece) >= min_chars:
                    yield piece
            continue

        next_len = current_len + len(para) + (1 if current else 0)
        if next_len > max_chars:
            yield from flush_current()

        current.append(para)
        current_len += len(para) + (1 if current_len else 0)

    yield from flush_current()


def iter_text_chunks(
    texts_dir: Path,
    *,
    max_chars: int = 900,
    overlap_chars: int = 120,
    min_chars: int = 80,
) -> Iterator[TextChunk]:
    for path in sorted(texts_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        rel_path = path.relative_to(texts_dir.parent).as_posix()
        for chunk_index, chunk in enumerate(
            chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars, min_chars=min_chars)
        ):
            yield TextChunk(path=rel_path, chunk_index=chunk_index, text=chunk)


def make_search_text(text: str) -> str:
    tokens = tokenize(text)
    return " ".join(tokens)


def quote_fts_token(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def make_match_query(query: str, *, max_terms: int = 48) -> str:
    tokens = tokenize(query)
    seen = set()
    ordered: List[str] = []
    for token in sorted(tokens, key=lambda item: (len(item), item), reverse=True):
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
        if len(ordered) >= max_terms:
            break
    return " OR ".join(quote_fts_token(token) for token in ordered)


def texts_signature(texts_dir: Path) -> Dict[str, Any]:
    count = 0
    total_size = 0
    max_mtime_ns = 0
    for path in texts_dir.rglob("*.txt"):
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        total_size += stat.st_size
        max_mtime_ns = max(max_mtime_ns, stat.st_mtime_ns)
    return {"file_count": count, "total_size": total_size, "max_mtime_ns": max_mtime_ns}


class TextFTSIndex:
    def __init__(
        self,
        *,
        texts_dir: Path = Path("data/sist/texts"),
        cache_path: Path = Path(".cache/rag_baseline_texts.sqlite"),
        chunk_chars: int = 900,
        overlap_chars: int = 120,
        min_chars: int = 80,
    ) -> None:
        self.texts_dir = texts_dir
        self.cache_path = cache_path
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars
        self.conn: Optional[sqlite3.Connection] = None

    def open(self, *, rebuild: bool = False) -> "TextFTSIndex":
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if rebuild and self.cache_path.exists():
            self.cache_path.unlink()
        self.conn = sqlite3.connect(str(self.cache_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        if rebuild or not self._is_current():
            self.build()
        return self

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _metadata(self) -> Dict[str, Any]:
        signature = texts_signature(self.texts_dir)
        return {
            "texts_dir": str(self.texts_dir),
            "chunk_chars": self.chunk_chars,
            "overlap_chars": self.overlap_chars,
            "min_chars": self.min_chars,
            **signature,
        }

    def _is_current(self) -> bool:
        assert self.conn is not None
        try:
            row = self.conn.execute("SELECT value FROM metadata WHERE key = 'build_metadata'").fetchone()
        except sqlite3.Error:
            return False
        if not row:
            return False
        try:
            return json.loads(row[0]) == self._metadata()
        except json.JSONDecodeError:
            return False

    def build(self) -> None:
        assert self.conn is not None
        self.conn.executescript(
            """
            DROP TABLE IF EXISTS metadata;
            DROP TABLE IF EXISTS chunks_fts;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                search_text,
                content UNINDEXED,
                path UNINDEXED,
                chunk_index UNINDEXED,
                tokenize='unicode61'
            );
            """
        )
        rows = []
        chunk_count = 0
        for chunk in iter_text_chunks(
            self.texts_dir,
            max_chars=self.chunk_chars,
            overlap_chars=self.overlap_chars,
            min_chars=self.min_chars,
        ):
            search_text = make_search_text(chunk.text)
            if not search_text:
                continue
            rows.append((search_text, chunk.text, chunk.path, chunk.chunk_index))
            chunk_count += 1
            if len(rows) >= 500:
                self.conn.executemany(
                    "INSERT INTO chunks_fts(search_text, content, path, chunk_index) VALUES (?, ?, ?, ?)",
                    rows,
                )
                rows = []
        if rows:
            self.conn.executemany(
                "INSERT INTO chunks_fts(search_text, content, path, chunk_index) VALUES (?, ?, ?, ?)",
                rows,
            )
        metadata = self._metadata()
        metadata["chunk_count"] = chunk_count
        self.conn.execute(
            "INSERT INTO metadata(key, value) VALUES ('build_metadata', ?)",
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True),),
        )
        self.conn.commit()

    def stats(self) -> Dict[str, Any]:
        assert self.conn is not None
        row = self.conn.execute("SELECT value FROM metadata WHERE key = 'build_metadata'").fetchone()
        metadata = json.loads(row[0]) if row else {}
        count = self.conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        metadata["chunk_count"] = count
        metadata["cache_path"] = str(self.cache_path)
        return metadata

    def search(self, query: str, *, top_k: int = 6) -> List[TextSearchHit]:
        assert self.conn is not None
        match_query = make_match_query(query)
        if not match_query:
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT bm25(chunks_fts) AS rank, path, chunk_index, content
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        hits: List[TextSearchHit] = []
        for rank, path, chunk_index, content in rows:
            hits.append(
                TextSearchHit(
                    rank=float(rank),
                    path=str(path),
                    chunk_index=int(chunk_index),
                    text=str(content),
                    snippet=make_snippet(str(content), query, max_chars=280),
                )
            )
        return hits
