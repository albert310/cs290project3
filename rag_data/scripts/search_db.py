from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from common import DEFAULT_DB_PATH, make_match_query, make_snippet, tokenize


SOURCE_TIER_BOOST = {
    "verified_seed": 8.0,
    "live_official": 5.5,
    "local_official_mirror": 4.0,
    "llm_curated": 3.5,
}

YEAR_RE = re.compile(r"(20\d{2})")
PROGRAM_CODE_RE = re.compile(r"(?<![A-Za-z0-9])(CS|EE|IE|CSEE|SI)(?![A-Za-z0-9])", re.I)

INTENT_RULES = {
    "degree": {
        "triggers": ("培养方案", "degree program", "degree programmes", "program requirements", "bachelor", "master", "ph.d", "phd", "博士", "硕士", "本科", "毕业要求", "学位"),
        "preferred": {"sist_degree_programs"},
        "discouraged": {"sist_courses", "sist_news_events"},
        "url_markers": ("degree", "programme", "program", "pyfa", "培养方案"),
    },
    "course": {
        "triggers": ("课程", "course", "courses", "任课", "老师", "instructor", "学分", "课程代码", "course code", "semester", "spring", "fall"),
        "preferred": {"sist_courses"},
        "discouraged": {"sist_news_events"},
        "url_markers": ("course", "courses", "schedule"),
    },
    "faculty": {
        "triggers": ("教授", "教师", "导师", "faculty", "professor", "邮箱", "email", "办公室", "office", "研究方向", "research interests", "个人主页"),
        "preferred": {"sist_faculty"},
        "discouraged": {"sist_news_events"},
        "url_markers": ("faculty", "main.htm"),
    },
    "contact": {
        "triggers": ("地址", "contact", "联系我们", "huaxia", "华夏中路", "campus", "校区"),
        "preferred": {"university_contact"},
        "discouraged": {"sist_news_events"},
        "url_markers": ("1059", "contact"),
    },
}


def exact_term_boost(query: str, row: sqlite3.Row) -> float:
    haystack = f"{row['title']}\n{row['category']}\n{row['text']}\n{row['source_url']}".lower()
    score = 0.0
    for token in set(tokenize(query)):
        if len(token) < 2:
            continue
        token_l = token.lower()
        if token_l in str(row["title"]).lower():
            score += min(len(token), 10) * 0.9
        elif token_l in haystack:
            score += min(len(token), 10) * 0.22
    return score


def intent_boost(query: str, row: sqlite3.Row) -> float:
    lowered = query.lower()
    category = str(row["category"] or "")
    url = str(row["source_url"] or "").lower()
    title = str(row["title"] or "").lower()
    text_head = str(row["text"] or "")[:800].lower()
    score = 0.0
    for rule in INTENT_RULES.values():
        if not any(trigger.lower() in lowered for trigger in rule["triggers"]):
            continue
        if category in rule["preferred"]:
            score += 24.0
        if category in rule["discouraged"]:
            score -= 18.0
        if any(marker.lower() in url for marker in rule["url_markers"]):
            score += 16.0
        if any(marker.lower() in title for marker in rule["triggers"]):
            score += 6.0

    query_years = set(YEAR_RE.findall(query))
    if query_years:
        if any(year in url or year in title or year in text_head for year in query_years):
            score += 18.0
        else:
            score -= 10.0

    program_codes = {match.group(1).upper() for match in PROGRAM_CODE_RE.finditer(query)}
    if program_codes:
        for code in program_codes:
            code_l = code.lower()
            encoded_markers = (
                f"in%20{code_l}",
                f"in_{code_l}",
                f"{code_l}.htm",
                f"{code_l}.pdf",
                f"{code_l}%e5",
            )
            if any(marker in url for marker in encoded_markers):
                score += 36.0
            if code_l == "cs" and ("计算机科学与技术" in row["text"] or "computer science" in text_head):
                score += 20.0
            if code_l == "ee" and ("电子信息工程" in row["text"] or "electrical" in text_head or "electronic" in text_head):
                score += 20.0
            if category == "sist_degree_programs" and any(term in lowered for term in ("bachelor", "本科", "培养方案")):
                if "degree%20program" in url or "degree programmes" in url:
                    score += 22.0

    if category == "sist_news_events" and any(term in title for term in ("询价", "采购", "公告", "家具", "报名启动")):
        score -= 12.0
    return score


def query_intents(query: str) -> set[str]:
    lowered = query.lower()
    intents = set()
    for name, rule in INTENT_RULES.items():
        if any(trigger.lower() in lowered for trigger in rule["triggers"]):
            intents.add(name)
    return intents


def supplemental_candidates(conn: sqlite3.Connection, query: str, *, category: str = "", limit: int = 80) -> List[sqlite3.Row]:
    """Recall high-precision official pages that FTS can miss for long pages.

    SQLite FTS may rank navigation-heavy list pages above long official degree
    pages. These supplemental clauses are intentionally narrow and only add
    records with direct URL/year/program-code evidence.
    """

    intents = query_intents(query)
    years = sorted(set(YEAR_RE.findall(query)))
    program_codes = {match.group(1).upper() for match in PROGRAM_CODE_RE.finditer(query)}
    clauses: List[str] = []
    params: List[Any] = []

    if "degree" in intents:
        base = ["category = 'sist_degree_programs'"]
        if category:
            base.append("category = ?")
            params.append(category)
        if years:
            year_parts = []
            for year in years:
                year_parts.append("(source_url LIKE ? OR title LIKE ? OR text LIKE ?)")
                params.extend([f"%{year}%", f"%{year}%", f"%{year}%"])
            base.append("(" + " OR ".join(year_parts) + ")")
        if program_codes:
            code_parts = []
            for code in program_codes:
                code_l = code.lower()
                if code == "CS":
                    code_parts.append(
                        "(lower(source_url) LIKE ? OR lower(source_url) LIKE ? OR text LIKE ? OR lower(text) LIKE ?)"
                    )
                    params.extend([f"%in%20{code_l}.htm%", f"%in%20{code_l}%", "%计算机科学与技术%", "%computer science%"])
                elif code == "EE":
                    code_parts.append(
                        "(lower(source_url) LIKE ? OR lower(source_url) LIKE ? OR text LIKE ? OR lower(text) LIKE ? OR lower(text) LIKE ?)"
                    )
                    params.extend([f"%in%20{code_l}.htm%", f"%in%20{code_l}%", "%电子信息工程%", "%electrical%", "%electronic%"])
                else:
                    code_parts.append("(lower(source_url) LIKE ? OR title LIKE ? OR text LIKE ?)")
                    params.extend([f"%{code_l}%", f"%{code}%", f"%{code}%"])
            base.append("(" + " OR ".join(code_parts) + ")")
        if any(term in query.lower() for term in ("bachelor", "本科")):
            base.append("(lower(source_url) LIKE '%undergraduate%' OR lower(source_url) LIKE '%bachelor%')")
        clauses.append("(" + " AND ".join(base) + ")")

    if "course" in intents:
        base = ["category = 'sist_courses'"]
        if category:
            base.append("category = ?")
            params.append(category)
        if years:
            year_parts = []
            for year in years:
                year_parts.append("(source_url LIKE ? OR title LIKE ? OR text LIKE ?)")
                params.extend([f"%{year}%", f"%{year}%", f"%{year}%"])
            base.append("(" + " OR ".join(year_parts) + ")")
        code_terms = re.findall(r"(?<![A-Za-z0-9])([A-Z]{2,}\d{2,4}[A-Z]?)(?![A-Za-z0-9])", query, flags=re.I)
        if code_terms:
            code_parts = []
            for code in sorted({item.upper() for item in code_terms}):
                code_parts.append("(title LIKE ? OR text LIKE ? OR source_url LIKE ?)")
                params.extend([f"%{code}%", f"%{code}%", f"%{code}%"])
            base.append("(" + " OR ".join(code_parts) + ")")
        clauses.append("(" + " AND ".join(base) + ")")

    if "faculty" in intents:
        name_terms = [term for term in re.findall(r"[\u4e00-\u9fff]{2,4}", query) if term not in {"教授", "教师", "导师", "邮箱", "办公室", "研究方向"}]
        if name_terms:
            base = ["category = 'sist_faculty'"]
            if category:
                base.append("category = ?")
                params.append(category)
            name_parts = []
            for name in name_terms[:4]:
                name_parts.append("(title LIKE ? OR text LIKE ? OR source_url LIKE ?)")
                params.extend([f"%{name}%", f"%{name}%", f"%{name}%"])
            base.append("(" + " OR ".join(name_parts) + ")")
            clauses.append("(" + " AND ".join(base) + ")")

    if not clauses:
        return []

    rows = conn.execute(
        f"""
        SELECT *
        FROM chunks
        WHERE {' OR '.join(clauses)}
        ORDER BY source_tier = 'verified_seed' DESC,
                 source_tier = 'live_official' DESC,
                 quality DESC,
                 id ASC
        LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    return rows


def search(db_path: Path, query: str, *, top_k: int = 8, category: str = "") -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    match_query = make_match_query(query)
    if not match_query:
        return []
    params: List[Any] = [match_query]
    category_clause = ""
    if category:
        category_clause = "AND c.category = ?"
        params.append(category)
    params.append(max(top_k * 10, 60))
    rows = conn.execute(
        f"""
        SELECT bm25(chunks_fts) AS bm25_rank, c.*
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ? {category_clause}
        ORDER BY bm25_rank
        LIMIT ?
        """,
        params,
    ).fetchall()
    supplemental = supplemental_candidates(conn, query, category=category, limit=max(top_k * 8, 40))
    hits: List[Dict[str, Any]] = []
    seen = set()
    by_chunk: Dict[int, Tuple[sqlite3.Row, float]] = {}
    for row in rows:
        by_chunk[int(row["id"])] = (row, -float(row["bm25_rank"] or 0.0))
    for row in supplemental:
        current = by_chunk.get(int(row["id"]))
        if current is None:
            by_chunk[int(row["id"])] = (row, 18.0)
        else:
            by_chunk[int(row["id"])] = (current[0], max(current[1], 18.0))

    for row, base_rank in by_chunk.values():
        text_key = re.sub(r"\s+", "", str(row["text"])[:260])
        if text_key in seen:
            continue
        seen.add(text_key)
        rank = base_rank
        rank += SOURCE_TIER_BOOST.get(str(row["source_tier"]), 1.0)
        rank += float(row["quality"] or 0.0) * 5.0
        rank += exact_term_boost(query, row)
        rank += intent_boost(query, row)
        hits.append(
            {
                "rank": round(rank, 4),
                "chunk_id": row["id"],
                "title": row["title"],
                "category": row["category"],
                "source_tier": row["source_tier"],
                "quality": row["quality"],
                "date": row["date"],
                "url": row["source_url"],
                "path": row["source_path"],
                "snippet": make_snippet(row["text"], query),
            }
        )
    hits.sort(key=lambda item: item["rank"], reverse=True)
    conn.close()
    return hits[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the clean ShanghaiTech/SIST RAG database.")
    parser.add_argument("query")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--category", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    hits = search(args.db, args.query, top_k=args.top_k, category=args.category)
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return
    for i, hit in enumerate(hits, start=1):
        print(f"[{i}] rank={hit['rank']} tier={hit['source_tier']} category={hit['category']} quality={hit['quality']:.3f}")
        print(f"    title: {hit['title']}")
        print(f"    url: {hit['url']}")
        print(f"    snippet: {hit['snippet']}")


if __name__ == "__main__":
    main()
