from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import tantivy

from retrieval.keyword_search import tokenize

from .unified_index import UnifiedRAGIndex, UnifiedSearchHit, _expand_query, _row_to_hit


DEFAULT_TANTIVY_INDEX_DIR = Path(".cache/tantivy_rag")
TOKEN_LIMIT_PER_CHUNK = 2400


def stable_unique(items: Iterable[str], *, limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        item = item.strip().lower()
        if len(item) < 2 or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def _escape_query_token(term: str) -> str:
    return re.sub(r"[^\w\u3400-\u4dbf\u4e00-\u9fff+#.-]+", " ", term).strip()


def query_text(query: str) -> str:
    expanded = _expand_query(query)
    terms = stable_unique([query, expanded, *tokenize(expanded)], limit=256)
    return " ".join(_escape_query_token(term) for term in terms)


def chunk_tokens(row: sqlite3.Row) -> str:
    title = str(row["title"] or "")
    text = str(row["text"] or "")
    source_type = str(row["source_type"] or "")
    category = str(row["category"] or "")
    host = str(row["host"] or "")
    metadata_terms = [source_type, category, host, title]
    terms = stable_unique([*metadata_terms, *tokenize(f"{title}\n{text}")], limit=TOKEN_LIMIT_PER_CHUNK)
    return " ".join(terms)


def build_schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_unsigned_field("chunk_id", stored=True)
    builder.add_text_field("title", stored=False)
    builder.add_text_field("body", stored=False)
    builder.add_text_field("tokens", stored=False)
    return builder.build()


def detect_schema_variant(conn: sqlite3.Connection) -> str:
    columns = {str(row[1]) for row in conn.execute("pragma table_info(chunks)").fetchall()}
    if {"source_tier", "source_url", "quality"}.issubset(columns):
        return "clean_rag_data"
    return "unified"


def select_columns(schema_variant: str) -> str:
    if schema_variant == "clean_rag_data":
        return """
            id, chunk_uid, doc_id, chunk_index, text, title, source_path,
            source_tier as source_type, category, source_url as url, host,
            date, quality as quality_score, metadata_json
        """
    return """
        id, chunk_uid, doc_id, chunk_index, text, title, source_path,
        source_type, category, url, host, date, quality_score, metadata_json
    """


def iter_rows(
    conn: sqlite3.Connection,
    schema_variant: str,
    *,
    limit: Optional[int],
    batch_size: int,
) -> Iterable[sqlite3.Row]:
    select = select_columns(schema_variant)
    offset = 0
    remaining = limit
    while True:
        current_limit = batch_size if remaining is None else min(batch_size, remaining)
        if current_limit <= 0:
            return
        rows = conn.execute(
            f"""
            select {select}
            from chunks
            order by id
            limit ? offset ?
            """,
            (current_limit, offset),
        ).fetchall()
        if not rows:
            return
        yield from rows
        offset += len(rows)
        if remaining is not None:
            remaining -= len(rows)
        if len(rows) < current_limit:
            return


def build_index(
    db_path: Path,
    index_dir: Path = DEFAULT_TANTIVY_INDEX_DIR,
    *,
    rebuild: bool = False,
    limit: Optional[int] = None,
    batch_size: int = 2000,
) -> None:
    if rebuild and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    marker = index_dir / "build_meta.json"
    if marker.exists() and not rebuild:
        return

    schema = build_schema()
    index = tantivy.Index(schema, path=str(index_dir))
    writer = index.writer(heap_size=512_000_000)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    schema_variant = detect_schema_variant(conn)
    total = conn.execute("select count(*) from chunks").fetchone()[0]
    if limit is not None:
        total = min(total, limit)

    started = time.time()
    count = 0
    for row in iter_rows(conn, schema_variant, limit=limit, batch_size=batch_size):
        doc = tantivy.Document()
        doc.add_unsigned("chunk_id", int(row["id"]))
        doc.add_text("title", str(row["title"] or ""))
        doc.add_text("body", str(row["text"] or ""))
        doc.add_text("tokens", chunk_tokens(row))
        writer.add_document(doc)
        count += 1
        if count % 10000 == 0:
            elapsed = time.time() - started
            print(f"indexed {count}/{total} chunks in {elapsed:.1f}s", flush=True)

    writer.commit()
    index.reload()
    conn.close()
    marker.write_text(
        json.dumps(
            {
                "db_path": str(db_path),
                "schema_variant": schema_variant,
                "chunks_indexed": count,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def doc_chunk_id(doc: tantivy.Document) -> int:
    as_dict = doc.to_dict()
    value = as_dict["chunk_id"]
    if isinstance(value, list):
        value = value[0]
    return int(value)


class TantivyRAGIndex:
    def __init__(
        self,
        db_path: Path = Path("data/rag/knowledge.sqlite"),
        index_dir: Path = DEFAULT_TANTIVY_INDEX_DIR,
        *,
        candidate_limit: int = 240,
        auto_build: bool = True,
    ) -> None:
        self.db_path = db_path
        self.index_dir = index_dir
        self.candidate_limit = candidate_limit
        self.auto_build = auto_build
        self.unified = UnifiedRAGIndex(db_path)
        self.index: Optional[tantivy.Index] = None

    def open(self) -> "TantivyRAGIndex":
        if self.auto_build and not (self.index_dir / "build_meta.json").exists():
            build_index(self.db_path, self.index_dir)
        self.unified.open()
        self.index = tantivy.Index.open(str(self.index_dir))
        self.index.reload()
        return self

    def close(self) -> None:
        self.unified.close()
        self.index = None

    def search(self, query: str, *, top_k: int = 8, candidate_limit: Optional[int] = None) -> List[UnifiedSearchHit]:
        assert self.index is not None
        assert self.unified.conn is not None
        target_candidates = candidate_limit or self.candidate_limit
        parsed, _errors = self.index.parse_query_lenient(
            query_text(query),
            ["title", "tokens", "body"],
            field_boosts={"title": 2.6, "tokens": 2.0, "body": 0.6},
        )
        searcher = self.index.searcher()
        result = searcher.search(parsed, target_candidates)
        scored_ids: List[tuple[int, float]] = []
        for score, address in result.hits:
            doc = searcher.doc(address)
            scored_ids.append((doc_chunk_id(doc), float(score)))

        rows_by_id = self._fetch_rows([chunk_id for chunk_id, _ in scored_ids])
        hits: List[UnifiedSearchHit] = []
        for chunk_id, raw_score in scored_ids:
            row = rows_by_id.get(chunk_id)
            if row is None:
                continue
            score = self.unified._rerank(query, row, raw_score)
            hits.append(_row_to_hit(row, query, score))
        hits.sort(key=lambda hit: hit.rank, reverse=True)
        return self.unified._dedupe(hits, top_k=top_k)

    def _fetch_rows(self, chunk_ids: Sequence[int]) -> Dict[int, sqlite3.Row]:
        assert self.unified.conn is not None
        if not chunk_ids:
            return {}
        select = select_columns(self.unified.schema_variant)
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.unified.conn.execute(
            f"""
            select {select}
            from chunks
            where id in ({placeholders})
            """,
            tuple(chunk_ids),
        ).fetchall()
        return {int(row["id"]): row for row in rows}


def search_tantivy(
    db_path: Path,
    index_dir: Path,
    query: str,
    *,
    top_k: int,
    candidates: int,
) -> List[UnifiedSearchHit]:
    index = TantivyRAGIndex(db_path, index_dir, candidate_limit=candidates).open()
    try:
        return index.search(query, top_k=top_k)
    finally:
        index.close()
