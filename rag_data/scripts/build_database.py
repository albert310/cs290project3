from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from common import (
    DEFAULT_DB_PATH,
    RAG_DATA_ROOT,
    PROJECT_ROOT,
    clean_text,
    extract_date,
    host_of,
    infer_category,
    iter_jsonl,
    looks_garbled,
    make_search_text,
    normalize_for_hash,
    open_db,
    quality_score,
    read_json,
    split_text,
    stable_hash,
    write_json,
)


CRAWL_MANIFEST = RAG_DATA_ROOT / "processed" / "official_crawl_manifest.jsonl"
VERIFIED_FACTS = RAG_DATA_ROOT / "config" / "verified_seed_facts.json"
LOCAL_DOCUMENTS = PROJECT_ROOT / "data" / "sist" / "jsonl" / "documents.jsonl"
LOCAL_TEXT_ROOT = PROJECT_ROOT / "data" / "sist"
BUILD_REPORT = RAG_DATA_ROOT / "reports" / "build_report.md"
REVIEW_SAMPLES = RAG_DATA_ROOT / "reports" / "review_samples.md"

SKIP_URL_PATTERNS = (
    re.compile(r"/_upload/tpl/", re.I),
    re.compile(r"/template", re.I),
    re.compile(r"\.(jpg|jpeg|png|gif|svg|ico|css|js|zip|rar|7z|mp4|avi)$", re.I),
)

EXCLUDE_TEXT_PATTERNS = (
    re.compile(r"\bWHO\b|World Health Organization|novel coronavirus|COVID-19 situation reports", re.I),
    re.compile(r"\barxiv\b|proceedings|references\s*$", re.I),
)

OFFICIAL_HOSTS = {
    "www.shanghaitech.edu.cn",
    "shanghaitech.edu.cn",
    "sist.shanghaitech.edu.cn",
    "faculty.sist.shanghaitech.edu.cn",
    "ssist.shanghaitech.edu.cn",
}


@dataclass
class SourceDoc:
    source_tier: str
    source_kind: str
    source_path: str
    source_url: str
    title: str
    category: str
    text: str
    date: str = ""
    priority: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def host(self) -> str:
        return host_of(self.source_url)

    @property
    def doc_uid(self) -> str:
        return stable_hash(f"{self.source_tier}\n{self.source_url}\n{self.source_path}\n{self.title}", length=20)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;

        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS documents;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chunks_fts;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_uid TEXT UNIQUE NOT NULL,
            source_tier TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_url TEXT NOT NULL,
            host TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            date TEXT NOT NULL,
            quality REAL NOT NULL,
            text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_uid TEXT UNIQUE NOT NULL,
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            source_tier TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_url TEXT NOT NULL,
            host TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            date TEXT NOT NULL,
            quality REAL NOT NULL,
            text TEXT NOT NULL,
            search_text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            search_text,
            text UNINDEXED,
            title UNINDEXED,
            category UNINDEXED,
            source_tier UNINDEXED,
            source_url UNINDEXED,
            tokenize='unicode61'
        );

        CREATE INDEX idx_chunks_category ON chunks(category);
        CREATE INDEX idx_chunks_tier ON chunks(source_tier);
        CREATE INDEX idx_chunks_url ON chunks(source_url);
        CREATE INDEX idx_chunks_quality ON chunks(quality);
        """
    )


def html_to_text_and_title(html: str, fallback_title: str = "") -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "form"]):
        tag.decompose()
    title = ""
    if soup.title:
        title = soup.title.get_text(" ", strip=True)
    for selector in ("h1", ".arti_title", ".wp_articlecontent h1", ".title"):
        node = soup.select_one(selector)
        if node:
            value = node.get_text(" ", strip=True)
            if value and len(value) > len(title):
                title = value
                break
    text = soup.get_text("\n", strip=True)
    title = title or fallback_title
    return title.strip(), text


def should_skip_url(url: str) -> bool:
    return any(pattern.search(url) for pattern in SKIP_URL_PATTERNS)


def should_skip_text(text: str, url: str) -> str:
    host = host_of(url)
    if host and host not in OFFICIAL_HOSTS and not host.endswith(".sist.shanghaitech.edu.cn"):
        return f"non_official_host:{host}"
    if should_skip_url(url):
        return "static_or_binary_url"
    if looks_garbled(text):
        return "garbled"
    for pattern in EXCLUDE_TEXT_PATTERNS:
        if pattern.search(text[:5000]):
            return "external_or_academic_noise"
    stripped = re.sub(r"\s+", "", text)
    if len(stripped) < 120:
        return "too_short"
    return ""


def make_verified_fact_docs(path: Path = VERIFIED_FACTS) -> Iterator[SourceDoc]:
    facts = read_json(path)
    for item in facts:
        title = str(item.get("question") or item.get("id") or "verified fact")
        answer = str(item.get("answer") or "")
        expected = "、".join(str(value) for value in item.get("expected_terms") or [])
        text = "\n".join(
            part
            for part in [
                title,
                f"标准答案/核验事实: {answer}",
                f"关键核验词: {expected}" if expected else "",
                f"来源说明: {item.get('source_note', '')}",
                f"来源 URL: {item.get('source_url', '')}",
            ]
            if part
        )
        yield SourceDoc(
            source_tier="verified_seed",
            source_kind="verified_fact",
            source_path=f"rag_data/config/verified_seed_facts.json#{item.get('id')}",
            source_url=str(item.get("source_url") or ""),
            title=title,
            category=str(item.get("category") or infer_category(title, text, str(item.get("source_url") or ""))),
            text=text,
            date="",
            priority=1.0,
            metadata={"id": item.get("id"), "expected_terms": item.get("expected_terms") or []},
        )


def make_crawled_docs(manifest_path: Path = CRAWL_MANIFEST) -> Iterator[SourceDoc]:
    for record in iter_jsonl(manifest_path):
        if record.get("status") != "ok":
            continue
        raw_rel = str(record.get("raw_path") or "")
        raw_path = RAG_DATA_ROOT / raw_rel
        if not raw_path.exists():
            continue
        url = str(record.get("final_url") or record.get("url") or "")
        try:
            raw = raw_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        title, raw_text = html_to_text_and_title(raw)
        cleaned = clean_text(raw_text)
        text = cleaned.text
        reason = should_skip_text(text, url)
        if reason:
            continue
        category = infer_category(title, text, url, fallback=str(record.get("category") or "general"))
        yield SourceDoc(
            source_tier="live_official",
            source_kind="html",
            source_path=raw_rel,
            source_url=url,
            title=title[:180],
            category=category,
            text=text,
            date=extract_date(text, url),
            priority=float(record.get("priority") or 0.8),
            metadata={
                "status_code": record.get("status_code"),
                "content_type": record.get("content_type"),
                "depth": record.get("depth"),
                "parent_url": record.get("parent_url"),
                "cleaning": cleaned.__dict__,
            },
        )


def make_local_mirror_docs(local_documents: Path = LOCAL_DOCUMENTS) -> Iterator[SourceDoc]:
    if not local_documents.exists():
        return
    for record in iter_jsonl(local_documents):
        url = str(record.get("canonical_url") or record.get("url") or "")
        if should_skip_url(url):
            continue
        host = host_of(url)
        if not (host in OFFICIAL_HOSTS or host.endswith(".sist.shanghaitech.edu.cn")):
            continue
        text_rel = str(record.get("text_path") or "").replace("\\", "/")
        if not text_rel:
            continue
        text_path = LOCAL_TEXT_ROOT / text_rel
        if not text_path.exists():
            continue
        try:
            raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        cleaned = clean_text(raw_text)
        text = cleaned.text
        reason = should_skip_text(text, url)
        if reason:
            continue
        title = str(record.get("title") or "").strip() or text.splitlines()[0][:120]
        category = infer_category(title, text, url, fallback=str(record.get("category") or "general"))
        yield SourceDoc(
            source_tier="local_official_mirror",
            source_kind="local_text",
            source_path=f"data/sist/{text_rel}",
            source_url=url,
            title=title[:180],
            category=category,
            text=text,
            date=extract_date(text, str(record.get("source_published_at") or ""), str(record.get("valid_from") or ""), url),
            priority=0.78,
            metadata={
                "source_document_id": record.get("id"),
                "language": record.get("language"),
                "fetched_at": record.get("fetched_at"),
                "source_published_at": record.get("source_published_at"),
                "cleaning": cleaned.__dict__,
            },
        )


def iter_source_docs(*, include_local: bool = True, include_crawl: bool = True) -> Iterator[SourceDoc]:
    yield from make_verified_fact_docs()
    if include_crawl:
        yield from make_crawled_docs()
    if include_local:
        yield from make_local_mirror_docs()


def insert_doc(conn: sqlite3.Connection, doc: SourceDoc) -> Tuple[Optional[int], int, float]:
    q = quality_score(
        source_tier=doc.source_tier,
        category=doc.category,
        url=doc.source_url,
        text=doc.text,
        title=doc.title,
    )
    content_hash = stable_hash(normalize_for_hash(doc.text), length=32)
    metadata_json = json.dumps(doc.metadata, ensure_ascii=False, sort_keys=True)
    try:
        cur = conn.execute(
            """
            INSERT INTO documents(
                doc_uid, source_tier, source_kind, source_path, source_url, host,
                title, category, date, quality, text, content_hash, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.doc_uid,
                doc.source_tier,
                doc.source_kind,
                doc.source_path,
                doc.source_url,
                doc.host,
                doc.title,
                doc.category,
                doc.date,
                q,
                doc.text,
                content_hash,
                metadata_json,
            ),
        )
    except sqlite3.IntegrityError:
        return None, 0, q
    doc_id = int(cur.lastrowid)
    chunk_count = 0
    for chunk_index, chunk in enumerate(split_text(doc.text)):
        chunk_uid = stable_hash(f"{doc.doc_uid}:{chunk_index}:{normalize_for_hash(chunk[:220])}", length=24)
        search_text = make_search_text(doc.title, doc.category, doc.source_url, chunk)
        chunk_metadata = {
            "doc_uid": doc.doc_uid,
            "content_hash": content_hash,
            **doc.metadata,
        }
        conn.execute(
            """
            INSERT OR IGNORE INTO chunks(
                chunk_uid, doc_id, chunk_index, source_tier, source_kind, source_path,
                source_url, host, title, category, date, quality, text, search_text, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_uid,
                doc_id,
                chunk_index,
                doc.source_tier,
                doc.source_kind,
                doc.source_path,
                doc.source_url,
                doc.host,
                doc.title,
                doc.category,
                doc.date,
                q,
                chunk,
                search_text,
                json.dumps(chunk_metadata, ensure_ascii=False, sort_keys=True),
            ),
        )
        if conn.total_changes:
            rowid = conn.execute("SELECT id FROM chunks WHERE chunk_uid=?", (chunk_uid,)).fetchone()[0]
            conn.execute(
                """
                INSERT INTO chunks_fts(rowid, search_text, text, title, category, source_tier, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (rowid, search_text, chunk, doc.title, doc.category, doc.source_tier, doc.source_url),
            )
            chunk_count += 1
    return doc_id, chunk_count, q


def write_reports(conn: sqlite3.Connection, summary: Dict[str, Any]) -> None:
    rows_by_tier = conn.execute(
        "SELECT source_tier, count(*) AS n, round(avg(quality), 3) AS avg_quality FROM chunks GROUP BY source_tier ORDER BY n DESC"
    ).fetchall()
    rows_by_category = conn.execute(
        "SELECT category, count(*) AS n, round(avg(quality), 3) AS avg_quality FROM chunks GROUP BY category ORDER BY n DESC"
    ).fetchall()
    rows_by_host = conn.execute(
        "SELECT host, count(*) AS n FROM chunks GROUP BY host ORDER BY n DESC LIMIT 30"
    ).fetchall()
    no_url = conn.execute("SELECT count(*) FROM chunks WHERE source_url=''").fetchone()[0]

    lines = [
        "# Clean ShanghaiTech/SIST RAG Database Report",
        "",
        f"- Built at UTC: `{summary['built_at']}`",
        f"- Database: `{summary['db_path']}`",
        f"- Documents inserted: **{summary['documents_inserted']}**",
        f"- Chunks inserted: **{summary['chunks_inserted']}**",
        f"- Duplicate documents skipped: **{summary['duplicate_docs']}**",
        f"- Chunks without URL: **{no_url}**",
        "",
        "## Source Tiers",
        "",
        "| source_tier | chunks | avg_quality |",
        "| --- | ---: | ---: |",
    ]
    for row in rows_by_tier:
        lines.append(f"| {row['source_tier']} | {row['n']} | {row['avg_quality']} |")
    lines.extend(["", "## Categories", "", "| category | chunks | avg_quality |", "| --- | ---: | ---: |"])
    for row in rows_by_category:
        lines.append(f"| {row['category']} | {row['n']} | {row['avg_quality']} |")
    lines.extend(["", "## Top Hosts", "", "| host | chunks |", "| --- | ---: |"])
    for row in rows_by_host:
        lines.append(f"| {row['host'] or '(none)'} | {row['n']} |")
    lines.extend(["", "## Build Summary", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), "```"])
    BUILD_REPORT.parent.mkdir(parents=True, exist_ok=True)
    BUILD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sample_lines = ["# Clean RAG Review Samples", "", "Deterministic samples from each source tier and category.", ""]
    for row in conn.execute(
        """
        SELECT id, title, category, source_tier, source_url, quality, text
        FROM chunks
        ORDER BY source_tier, category, id
        LIMIT 120
        """
    ):
        sample_lines.extend(
            [
                f"## chunk {row['id']} | {row['title']}",
                "",
                f"- category: `{row['category']}`",
                f"- source_tier: `{row['source_tier']}`",
                f"- quality: `{row['quality']:.3f}`",
                f"- url: `{row['source_url']}`",
                "",
                "> " + re.sub(r"\s+", " ", row["text"])[:700],
                "",
            ]
        )
    REVIEW_SAMPLES.write_text("\n".join(sample_lines), encoding="utf-8")


def build_database(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    include_local: bool = True,
    include_crawl: bool = True,
    limit_docs: Optional[int] = None,
) -> Dict[str, Any]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    for suffix in ("-wal", "-shm"):
        side = Path(str(db_path) + suffix)
        if side.exists():
            side.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    init_db(conn)
    stats: Counter[str] = Counter()
    by_tier: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    seen_content: set[str] = set()

    for doc in iter_source_docs(include_local=include_local, include_crawl=include_crawl):
        if limit_docs is not None and stats["docs_seen"] >= limit_docs:
            break
        stats["docs_seen"] += 1
        content_key = stable_hash(normalize_for_hash(doc.text), length=32)
        if content_key in seen_content:
            stats["duplicate_docs"] += 1
            continue
        seen_content.add(content_key)
        doc_id, chunk_count, q = insert_doc(conn, doc)
        if doc_id is None:
            stats["duplicate_docs"] += 1
            continue
        if chunk_count == 0:
            stats["docs_without_chunks"] += 1
            continue
        stats["documents_inserted"] += 1
        stats["chunks_inserted"] += chunk_count
        by_tier[doc.source_tier] += chunk_count
        by_category[doc.category] += chunk_count
        if stats["documents_inserted"] % 500 == 0:
            conn.commit()

    summary = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db_path": str(db_path.resolve()),
        "include_local": include_local,
        "include_crawl": include_crawl,
        "limit_docs": limit_docs,
        "docs_seen": stats["docs_seen"],
        "documents_inserted": stats["documents_inserted"],
        "chunks_inserted": stats["chunks_inserted"],
        "duplicate_docs": stats["duplicate_docs"],
        "docs_without_chunks": stats["docs_without_chunks"],
        "chunks_by_tier": dict(by_tier),
        "chunks_by_category": dict(by_category),
    }
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        ("build_summary", json.dumps(summary, ensure_ascii=False, sort_keys=True)),
    )
    conn.commit()
    write_reports(conn, summary)
    conn.close()
    write_json(RAG_DATA_ROOT / "reports" / "build_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean ShanghaiTech/SIST RAG SQLite database.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--no-local", action="store_true", help="Do not include local data/sist official mirror.")
    parser.add_argument("--no-crawl", action="store_true", help="Do not include crawled official pages.")
    parser.add_argument("--limit-docs", type=int)
    args = parser.parse_args()
    summary = build_database(
        db_path=args.db,
        include_local=not args.no_local,
        include_crawl=not args.no_crawl,
        limit_docs=args.limit_docs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
