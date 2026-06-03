from __future__ import annotations

import argparse
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from sist_data import SISTDataset


ASCII_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#.-]*|\d+(?:\.\d+)?")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
CJK_STOP_UNIGRAMS = set("的一是在了和与及或对中为有个们于年月日")
QUERY_EXPANSIONS = {
    "上海科技大学": ["shanghaitech", "shanghai tech"],
    "信息学院": ["sist", "school of information science and technology", "信息科学与技术学院"],
    "计算机科学与技术": ["computer science", "cs"],
    "电子科学与技术": ["electronic science", "ee"],
    "信息与通信工程": ["communication engineering"],
    "课程": ["course", "courses"],
    "任课老师": ["instructor", "teacher", "lecturer", "授课教师"],
    "老师": ["instructor", "teacher", "professor", "教师"],
    "导师": ["advisor", "supervisor", "professor", "faculty", "研究方向"],
    "学分": ["credit", "credits"],
    "毕业": ["graduation", "graduate"],
    "培养方案": ["degree program", "program requirements"],
    "邮箱": ["email", "mail"],
    "邮件": ["email", "mail"],
    "电话": ["phone", "tel"],
    "办公室": ["office"],
    "讲座": ["lecture", "talk", "seminar", "卓越讲座"],
}
DEFAULT_STRUCTURED_TABLES = (
    "contacts",
    "courses",
    "events",
    "facilities",
    "faculty_members",
    "leadership_roles",
    "program_requirements",
    "staff_members",
)
STRUCTURED_TITLE_FIELDS = (
    "name",
    "name_en",
    "course_name",
    "course_name_en",
    "program_name",
    "subject",
    "summary",
    "title",
)
STRUCTURED_URL_FIELDS = ("source_url", "profile_url", "homepage", "detail_url", "url", "list_page_url")
FIELD_LABELS = {
    "email": "邮箱 email",
    "phone": "电话 phone",
    "office": "办公室 office",
    "instructor": "任课老师 instructor",
    "credits": "学分 credits",
    "course_code": "课程编号 course code",
    "course_name": "课程名称 course name",
    "course_name_en": "英文课程名 course name english",
    "program_name": "专业 项目 program",
    "min_credits": "最低学分 min credits",
    "requirement_text": "要求 requirement",
    "requirement_type": "要求类型 requirement type",
    "research_area": "研究方向 research area",
    "role": "角色 职务 role",
    "title": "标题 title",
    "source_url": "来源 source url",
}


@dataclass(frozen=True)
class SearchHit:
    score: float
    record_id: str
    text: str
    metadata: Dict[str, Any]

    def to_dict(self, snippet_chars: int = 240, query: str = "") -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "id": self.record_id,
            "title": self.metadata.get("title"),
            "url": self.metadata.get("url"),
            "category": self.metadata.get("category"),
            "chunk_id": self.metadata.get("chunk_id"),
            "document_id": self.metadata.get("document_id"),
            "snippet": make_snippet(self.text, query, max_chars=snippet_chars),
        }


def tokenize(text: str) -> List[str]:
    """Tokenize mixed Chinese/English text for lexical retrieval."""

    if not text:
        return []

    tokens: List[str] = []
    lowered = text.lower()

    tokens.extend(match.group(0) for match in ASCII_RE.finditer(lowered))

    for match in CJK_RE.finditer(text):
        segment = match.group(0)
        if len(segment) <= 8:
            tokens.append(segment)
        for char in segment:
            if char not in CJK_STOP_UNIGRAMS:
                tokens.append(char)
        for n in (2, 3):
            if len(segment) >= n:
                tokens.extend(segment[i : i + n] for i in range(len(segment) - n + 1))

    return tokens


def expand_query(query: str) -> str:
    additions: List[str] = []
    lowered = query.lower()
    for trigger, values in QUERY_EXPANSIONS.items():
        if trigger.lower() in lowered:
            additions.extend(values)
    if not additions:
        return query
    return query + " " + " ".join(additions)


def make_snippet(text: str, query: str, max_chars: int = 240) -> str:
    if len(text) <= max_chars:
        return text

    candidates = [tok for tok in tokenize(query) if len(tok) >= 2]
    start = -1
    for tok in sorted(candidates, key=len, reverse=True):
        start = text.lower().find(tok.lower())
        if start >= 0:
            break

    if start < 0:
        return text[:max_chars].strip()

    left = max(0, start - max_chars // 3)
    right = min(len(text), left + max_chars)
    snippet = text[left:right].strip()
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    return snippet


class BM25FIndex:
    def __init__(
        self,
        *,
        k1: float = 1.5,
        field_weights: Optional[Mapping[str, float]] = None,
        field_b: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.k1 = k1
        self.field_weights = dict(field_weights or {"title": 3.0, "text": 1.0, "category": 0.8, "url": 0.2})
        self.field_b = dict(field_b or {"title": 0.2, "text": 0.75, "category": 0.0, "url": 0.0})
        self.records: List[Dict[str, Any]] = []
        self.field_counters: List[Dict[str, Counter[str]]] = []
        self.field_lengths: List[Dict[str, int]] = []
        self.avg_field_lengths: Dict[str, float] = {}
        self.postings: DefaultDict[str, List[int]] = defaultdict(list)
        self.doc_freq: Dict[str, int] = {}
        self._built = False

    @classmethod
    def from_dataset(
        cls,
        dataset: SISTDataset,
        *,
        categories: Optional[Sequence[str]] = None,
        min_chars: int = 20,
        limit: Optional[int] = None,
        include_structured: bool = True,
        structured_tables: Sequence[str] = DEFAULT_STRUCTURED_TABLES,
    ) -> "BM25FIndex":
        index = cls()
        for record in dataset.iter_rag_records(categories=categories, min_chars=min_chars):
            index.add_record(record)
            if limit is not None and len(index.records) >= limit:
                index.build()
                return index
        if include_structured:
            for record in iter_structured_rag_records(dataset, tables=structured_tables, min_chars=min_chars):
                index.add_record(record)
                if limit is not None and len(index.records) >= limit:
                    index.build()
                    return index
        index.build()
        return index

    def add_record(self, record: Mapping[str, Any]) -> None:
        if self._built:
            raise RuntimeError("Cannot add records after build().")

        metadata = dict(record.get("metadata") or {})
        text = str(record.get("text") or "")
        fields = {
            "title": str(metadata.get("title") or ""),
            "category": str(metadata.get("category") or ""),
            "url": str(metadata.get("url") or ""),
            "text": text,
        }

        counters: Dict[str, Counter[str]] = {}
        lengths: Dict[str, int] = {}
        for field, value in fields.items():
            tokens = tokenize(value)
            counters[field] = Counter(tokens)
            lengths[field] = len(tokens)

        self.records.append({"id": str(record.get("id")), "text": text, "metadata": metadata})
        self.field_counters.append(counters)
        self.field_lengths.append(lengths)

    def build(self) -> None:
        if self._built:
            return

        total_lengths: DefaultDict[str, int] = defaultdict(int)
        for lengths in self.field_lengths:
            for field in self.field_weights:
                total_lengths[field] += lengths.get(field, 0)
        doc_count = max(len(self.records), 1)
        self.avg_field_lengths = {
            field: max(total_lengths[field] / doc_count, 1e-9)
            for field in self.field_weights
        }

        doc_sets: DefaultDict[str, set[int]] = defaultdict(set)
        for doc_id, counters in enumerate(self.field_counters):
            seen = set()
            for field in self.field_weights:
                seen.update(counters.get(field, {}).keys())
            for token in seen:
                doc_sets[token].add(doc_id)

        for token, doc_ids in doc_sets.items():
            ordered = sorted(doc_ids)
            self.postings[token] = ordered
            self.doc_freq[token] = len(ordered)

        self._built = True

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        categories: Optional[Sequence[str]] = None,
        deduplicate: bool = True,
        expand: bool = False,
    ) -> List[SearchHit]:
        if not self._built:
            self.build()

        query_text = expand_query(query) if expand else query
        query_terms = Counter(tokenize(query_text))
        if not query_terms:
            return []

        category_set = set(categories) if categories else None
        scores: DefaultDict[int, float] = defaultdict(float)
        doc_count = len(self.records)

        for token, query_tf in query_terms.items():
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = math.log(1.0 + (doc_count - self.doc_freq[token] + 0.5) / (self.doc_freq[token] + 0.5))

            for doc_id in postings:
                metadata = self.records[doc_id]["metadata"]
                if category_set and metadata.get("category") not in category_set:
                    continue
                weighted_tf = self._weighted_tf(doc_id, token)
                if weighted_tf <= 0:
                    continue
                bm25_tf = ((self.k1 + 1.0) * weighted_tf) / (self.k1 + weighted_tf)
                scores[doc_id] += query_tf * idf * bm25_tf

        if not scores:
            return []

        ranked = heapq.nlargest(len(scores), scores.items(), key=lambda item: item[1])
        hits: List[SearchHit] = []
        seen = set()
        for doc_id, score in ranked:
            record = self.records[doc_id]
            metadata = record["metadata"]
            text = record["text"]
            if deduplicate:
                key = metadata.get("dedup_key") or (metadata.get("url"), text[:500])
                if key in seen:
                    continue
                seen.add(key)
            hits.append(SearchHit(score=score, record_id=record["id"], text=text, metadata=metadata))
            if len(hits) >= top_k:
                break
        return hits

    def _weighted_tf(self, doc_id: int, token: str) -> float:
        value = 0.0
        counters = self.field_counters[doc_id]
        lengths = self.field_lengths[doc_id]

        for field, weight in self.field_weights.items():
            tf = counters.get(field, Counter()).get(token, 0)
            if tf <= 0:
                continue
            avg_len = self.avg_field_lengths.get(field, 1.0)
            b = self.field_b.get(field, 0.75)
            norm = 1.0 - b + b * (lengths.get(field, 0) / avg_len)
            value += weight * tf / max(norm, 1e-9)
        return value


def _print_hits(query: str, hits: Sequence[SearchHit]) -> None:
    print(f"query: {query}")
    for rank, hit in enumerate(hits, start=1):
        row = hit.to_dict(query=query)
        print(f"[{rank}] score={row['score']} title={row['title']} category={row['category']}")
        print(f"    url={row['url']}")
        print(f"    chunk_id={row['chunk_id']} document_id={row['document_id']}")
        print(f"    {row['snippet']}")


def iter_structured_rag_records(
    dataset: SISTDataset,
    *,
    tables: Sequence[str] = DEFAULT_STRUCTURED_TABLES,
    min_chars: int = 20,
) -> Iterator[Dict[str, Any]]:
    for table in tables:
        for row in dataset.iter_jsonl(table):
            text = stringify_structured_row(table, row)
            if len(text) < min_chars:
                continue
            title = first_nonempty(row, STRUCTURED_TITLE_FIELDS) or f"{table}:{row.get('id')}"
            url = first_nonempty(row, STRUCTURED_URL_FIELDS)
            yield {
                "id": f"{table}:{row.get('id')}",
                "text": text,
                "metadata": {
                    "chunk_id": None,
                    "document_id": row.get("source_document_id"),
                    "chunk_index": None,
                    "title": title,
                    "url": url,
                    "category": table,
                    "source_table": table,
                    "source_id": row.get("id"),
                    "dedup_key": structured_dedup_key(table, row, title),
                    "char_count": len(text),
                },
            }


def stringify_structured_row(table: str, row: Mapping[str, Any]) -> str:
    lines = [f"table: {table}", f"表: {table}"]
    for key, value in row.items():
        text = format_structured_value(value)
        if not text:
            continue
        label = FIELD_LABELS.get(key, key)
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def format_structured_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def first_nonempty(row: Mapping[str, Any], fields: Sequence[str]) -> Optional[str]:
    for field in fields:
        value = format_structured_value(row.get(field))
        if value:
            return value
    return None


def structured_dedup_key(table: str, row: Mapping[str, Any], title: str) -> str:
    if table == "courses":
        fields = ("course_code", "course_name", "course_name_en", "cohort", "term", "instructor")
    elif table == "contacts":
        fields = ("name", "email", "phone", "office", "context")
    elif table == "faculty_members":
        fields = ("name", "name_en", "email", "profile_url", "homepage")
    elif table == "leadership_roles":
        fields = ("name", "role", "org", "status")
    elif table == "program_requirements":
        fields = ("program_name", "degree", "cohort", "requirement_type", "requirement_text")
    elif table == "events":
        fields = ("title", "event_date", "published_at", "source_url")
    else:
        fields = ("name", "title", "source_url", "evidence")

    parts = [table, title]
    for field in fields:
        value = format_structured_value(row.get(field))
        if value:
            parts.append(value[:240])
    return "|".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BM25F keyword search over SIST RAG chunks.")
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--data-root", default="data/sist")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="Index only the first N chunks for quick debugging.")
    parser.add_argument("--chunks-only", action="store_true", help="Index only chunks.jsonl, excluding structured tables.")
    parser.add_argument("--no-dedup", action="store_true", help="Keep duplicate chunks in the top-k results.")
    parser.add_argument("--expand", action="store_true", help="Enable the small Chinese/English query expansion map.")
    parser.add_argument("--json", action="store_true", help="Print hits as JSON lines.")
    args = parser.parse_args(argv)

    if not args.query:
        parser.error("query is required")

    dataset = SISTDataset(args.data_root)
    index = BM25FIndex.from_dataset(
        dataset,
        categories=args.category,
        min_chars=args.min_chars,
        limit=args.limit,
        include_structured=not args.chunks_only,
    )
    hits = index.search(
        args.query,
        top_k=args.top_k,
        categories=args.category,
        deduplicate=not args.no_dedup,
        expand=args.expand,
    )

    if args.json:
        for hit in hits:
            print(json.dumps(hit.to_dict(query=args.query), ensure_ascii=False))
    else:
        print(f"indexed_chunks: {len(index.records)}")
        _print_hits(args.query, hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
