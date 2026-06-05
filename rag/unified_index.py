from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from retrieval.keyword_search import make_snippet, tokenize

from .text_index import make_match_query


COURSE_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,}\d{2,4}[A-Z]?)(?![A-Za-z0-9])", flags=re.IGNORECASE)
PROGRAM_CODE_RE = re.compile(r"(?<![A-Za-z0-9])(CS|EE|IE|CSEE|SI)(?![A-Za-z0-9])", flags=re.IGNORECASE)
YEAR_RE = re.compile(r"(20\d{2})")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
CJK_PHRASE_RE = re.compile(r"[\u4e00-\u9fff]{2,8}")

SOURCE_PRIORITY = {
    "verified_seed": 7.0,
    "live_official": 6.2,
    "local_official_mirror": 5.0,
    "training": 5.0,
    "structured_json": 4.6,
    "structured_table": 4.4,
    "sist_text": 4.0,
    "web": 2.8,
    "text_pages": 2.4,
    "sist_raw": 2.0,
    "pdf_md": 1.2,
}

STRUCTURED_CATEGORIES = {
    "courses",
    "sist_courses",
    "courses_clean",
    "courses_unified",
    "course_teacher_map",
    "faculty_members",
    "sist_faculty",
    "contacts",
    "program_requirements",
    "program_sources",
    "events",
    "leadership_roles",
    "university_overview",
    "university_contact",
    "sist_overview",
    "sist_degree_programs",
    "sist_news_events",
    "sist_research",
}

QUERY_EXPANSIONS = {
    "类型的高校": ["全日制普通高等学校", "研究型", "创新型", "小规模", "高水平", "国际化"],
    "什么类型的高校": ["全日制普通高等学校", "研究型", "创新型", "小规模", "高水平", "国际化"],
    "日常管理": ["上海市人民政府负责日常管理", "负责日常管理", "上海市人民政府"],
    "第四单元": ["考试科目", "业务课二", "专业课"],
    "先修": ["先修课程", "Prerequisites"],
    "任课": ["任课教师", "Instructor"],
    "老师": ["任课教师", "Instructor"],
    "学分": ["Credit", "credits"],
    "研究中心": ["Research Center", "研究中心"],
    "研究方向": ["research interests", "研究兴趣", "研究方向"],
    "招生方式": ["招生方式", "申请考核", "硕博连读", "直接攻博"],
    "占地": ["校园占地", "亩"],
    "建筑面积": ["总建筑面积", "平方米"],
    "开放时间": ["开馆时间", "馆舍开放时间", "Hours"],
}


@dataclass(frozen=True)
class UnifiedSearchHit:
    rank: float
    chunk_id: int
    chunk_uid: str
    path: str
    chunk_index: int
    text: str
    snippet: str
    title: str
    source_type: str
    category: str
    url: str
    host: str
    date: str
    quality_score: float

    @property
    def source_id(self) -> str:
        title = self.title or self.category or self.source_type
        return f"{self.source_type}:{self.path}#chunk={self.chunk_index}; title={title}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "chunk_uid": self.chunk_uid,
            "path": self.path,
            "chunk_index": self.chunk_index,
            "source_id": self.source_id,
            "title": self.title,
            "source_type": self.source_type,
            "category": self.category,
            "url": self.url,
            "host": self.host,
            "date": self.date,
            "quality_score": self.quality_score,
            "snippet": self.snippet,
        }


def _normalize_query_tokens(query: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for token in tokenize(query):
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _query_course_codes(query: str) -> List[str]:
    return sorted({match.group(0).upper() for match in COURSE_CODE_RE.finditer(query)})


def _query_years(query: str) -> List[str]:
    return sorted({match.group(1) for match in YEAR_RE.finditer(query)})


def _query_program_codes(query: str) -> List[str]:
    return sorted({match.group(1).upper() for match in PROGRAM_CODE_RE.finditer(query)})


def _has_degree_intent(query: str) -> bool:
    lowered = query.lower()
    return any(
        term in query or term in lowered
        for term in (
            "培养方案",
            "学位",
            "毕业要求",
            "本科",
            "硕士",
            "博士",
            "degree program",
            "degree programmes",
            "program requirements",
            "bachelor",
            "master",
            "phd",
            "ph.d",
        )
    )


def _has_course_intent(query: str) -> bool:
    lowered = query.lower()
    return any(
        term in query or term in lowered
        for term in (
            "课程",
            "课表",
            "任课",
            "老师",
            "教师",
            "学分",
            "课程代码",
            "course",
            "courses",
            "instructor",
            "teacher",
            "semester",
            "spring",
            "fall",
        )
    )


def _has_sist_overview_intent(query: str) -> bool:
    lowered = query.lower()
    has_entity = (
        "sist" in lowered
        or "信息学院" in query
        or "信息科学与技术学院" in query
        or "school of information science and technology" in lowered
    )
    if not has_entity:
        return False
    overview_terms = (
        "overview",
        "about",
        "vision",
        "mission",
        "at a glance",
        "概况",
        "简介",
        "介绍",
        "是什么",
        "about sist",
    )
    if any(term in lowered or term in query for term in overview_terms):
        return True
    return not any(
        intent(query)
        for intent in (_has_degree_intent, _has_course_intent, _profileish_query)
    )


def _clean_intent_boost(query: str, row: sqlite3.Row) -> float:
    text = str(row["text"] or "")
    title = str(row["title"] or "")
    title_lower = title.lower()
    text_head_lower = text[:900].lower()
    url = str(row["url"] or "").lower()
    category = str(row["category"] or "")
    score = 0.0

    if _has_sist_overview_intent(query):
        if category == "sist_overview":
            score += 70.0
        elif category in {"sist_news_events", "sist_courses", "sist_degree_programs", "sist_faculty"}:
            score -= 26.0
        if url.rstrip("/") in {
            "https://sist.shanghaitech.edu.cn",
            "https://sist.shanghaitech.edu.cn/sist_en",
            "https://faculty.sist.shanghaitech.edu.cn",
        }:
            score += 60.0
        if any(marker in url for marker in ("sist_en", "about", "overview")):
            score += 30.0
        if any(term in text_head_lower for term in ("about sist", "sist at a glance", "vision and mission")):
            score += 28.0
        if any(term in title_lower for term in ("school of information science and technology", "信息科学与技术学院")):
            score += 12.0

    if _has_degree_intent(query):
        if category == "sist_degree_programs":
            score += 24.0
        elif category in {"sist_courses", "sist_news_events"}:
            score -= 18.0
        if any(marker in url for marker in ("degree", "programme", "program", "pyfa")):
            score += 16.0
        if any(term in title_lower for term in ("培养方案", "degree", "program")):
            score += 6.0

    if _has_course_intent(query):
        if category == "sist_courses":
            score += 24.0
        elif category == "sist_news_events":
            score -= 18.0
        if any(marker in url for marker in ("course", "courses", "schedule")):
            score += 16.0
        if any(term in title_lower for term in ("course", "课程", "课表", "schedule")):
            score += 6.0

    if _profileish_query(query):
        if category == "sist_faculty":
            score += 24.0
        elif category == "sist_news_events":
            score -= 18.0
        if any(marker in url for marker in ("faculty", "main.htm")):
            score += 16.0
        if any(term in text for term in ("邮箱", "办公室", "研究方向", "博士毕业院校")):
            score += 10.0

    years = _query_years(query)
    if years:
        if any(year in url or year in title_lower or year in text_head_lower for year in years):
            score += 18.0
        else:
            score -= 10.0

    for code in _query_program_codes(query):
        code_lower = code.lower()
        encoded_markers = (
            f"in%20{code_lower}",
            f"in_{code_lower}",
            f"{code_lower}.htm",
            f"{code_lower}.pdf",
            f"{code_lower}%e5",
        )
        if any(marker in url for marker in encoded_markers):
            score += 36.0
        if code == "CS" and ("计算机科学与技术" in text or "computer science" in text_head_lower):
            score += 20.0
        if code == "EE" and ("电子信息工程" in text or "electrical" in text_head_lower or "electronic" in text_head_lower):
            score += 20.0
        if category == "sist_degree_programs" and any(term in query.lower() for term in ("bachelor", "本科", "培养方案")):
            if "degree%20program" in url or "degree programmes" in url:
                score += 22.0

    if category == "sist_news_events" and any(term in title for term in ("询价", "采购", "公告", "家具", "报名启动")):
        score -= 12.0
    return score


def _profileish_query(query: str) -> bool:
    lowered = query.lower()
    return any(
        term in query or term in lowered
        for term in (
            "主页",
            "个人主页",
            "教师",
            "教授",
            "研究方向",
            "研究兴趣",
            "邮箱",
            "办公室",
            "profile",
            "homepage",
            "faculty",
            "research interests",
        )
    )


def _query_exact_cjk_phrases(query: str) -> List[str]:
    stop = {
        "上海科技大学",
        "信息科学",
        "技术学院",
        "信息科学与技术学院",
        "研究方向",
        "研究兴趣",
        "个人主页",
        "任课教师",
        "上海市人民政府",
        "中国科学院",
    }
    out: List[str] = []
    seen = set()
    for match in CJK_PHRASE_RE.finditer(query):
        phrase = match.group(0)
        if phrase in stop:
            continue
        if not (2 <= len(phrase) <= 4):
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= 5:
            break
    return out


def _is_cjk_char(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _contains_exact_cjk_phrase(text: str, phrase: str) -> bool:
    start = 0
    while True:
        index = text.find(phrase, start)
        if index < 0:
            return False
        before = text[index - 1] if index > 0 else ""
        after_index = index + len(phrase)
        after = text[after_index] if after_index < len(text) else ""
        if (not before or not _is_cjk_char(before)) and (not after or not _is_cjk_char(after)):
            return True
        start = index + 1


def _expand_query(query: str) -> str:
    additions: List[str] = []
    lowered = query.lower()
    for trigger, values in QUERY_EXPANSIONS.items():
        if trigger.lower() in lowered:
            additions.extend(values)
    if not additions:
        return query
    return query + " " + " ".join(additions)


def _row_to_hit(row: sqlite3.Row, query: str, score: float) -> UnifiedSearchHit:
    return UnifiedSearchHit(
        rank=score,
        chunk_id=int(row["id"]),
        chunk_uid=str(row["chunk_uid"]),
        path=str(row["source_path"]),
        chunk_index=int(row["chunk_index"]),
        text=str(row["text"]),
        snippet=make_snippet(str(row["text"]), query, max_chars=320),
        title=str(row["title"] or ""),
        source_type=str(row["source_type"] or ""),
        category=str(row["category"] or ""),
        url=str(row["url"] or ""),
        host=str(row["host"] or ""),
        date=str(row["date"] or ""),
        quality_score=float(row["quality_score"] or 0.0),
    )


class UnifiedRAGIndex:
    def __init__(self, db_path: Path = Path("data/rag/knowledge.sqlite")) -> None:
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.schema_variant = "unified"

    def open(self) -> "UnifiedRAGIndex":
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        columns = {str(row[1]) for row in self.conn.execute("pragma table_info(chunks)").fetchall()}
        if {"source_tier", "source_url", "quality"}.issubset(columns):
            self.schema_variant = "clean_rag_data"
        else:
            self.schema_variant = "unified"
        return self

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def stats(self) -> Dict[str, Any]:
        assert self.conn is not None
        out: Dict[str, Any] = {"db_path": str(self.db_path)}
        tables = ["documents", "chunks", "chunks_fts"]
        if self.schema_variant != "clean_rag_data":
            tables.append("structured_records")
        for table in tables:
            out[table] = self.conn.execute(f"select count(*) from {table}").fetchone()[0]
        row = self.conn.execute("select value from metadata where key='build_summary'").fetchone()
        if row:
            out["build_summary"] = json.loads(row[0])
        out["schema_variant"] = self.schema_variant
        return out

    def search(self, query: str, *, top_k: int = 8, candidate_limit: int = 180) -> List[UnifiedSearchHit]:
        assert self.conn is not None
        course_codes = _query_course_codes(query)
        candidates: Dict[int, Tuple[sqlite3.Row, float]] = {}
        for row, raw_score in self._exact_candidates(query, limit=max(top_k * 8, 40)):
            candidates[int(row["id"])] = (row, raw_score)
        if self.schema_variant == "clean_rag_data":
            for row, raw_score in self._clean_supplemental_candidates(query, limit=max(top_k * 10, 60)):
                current = candidates.get(int(row["id"]))
                if current is None or raw_score > current[1]:
                    candidates[int(row["id"])] = (row, raw_score)
        for row, raw_score in self._fts_candidates(query, limit=candidate_limit):
            current = candidates.get(int(row["id"]))
            if current is None or raw_score > current[1]:
                candidates[int(row["id"])] = (row, raw_score)

        if course_codes:
            candidates = {
                chunk_id: (row, raw_score)
                for chunk_id, (row, raw_score) in candidates.items()
                if any(code in f"{row['title']}\n{row['text']}".upper() for code in course_codes)
            }
            if not candidates:
                return []

        hits = [
            _row_to_hit(row, query, self._rerank(query, row, raw_score))
            for row, raw_score in candidates.values()
        ]
        hits.sort(key=lambda hit: hit.rank, reverse=True)
        return self._dedupe(hits, top_k=top_k)

    def _fts_candidates(self, query: str, *, limit: int) -> Iterable[Tuple[sqlite3.Row, float]]:
        assert self.conn is not None
        match_query = make_match_query(_expand_query(query), max_terms=72)
        if not match_query:
            return []
        if self.schema_variant == "clean_rag_data":
            select_columns = """
                c.id, c.chunk_uid, c.doc_id, c.chunk_index, c.text, c.title,
                c.source_path, c.source_tier as source_type, c.category,
                c.source_url as url, c.host, c.date, c.quality as quality_score,
                c.metadata_json
            """
        else:
            select_columns = "c.*"
        try:
            rows = self.conn.execute(
                f"""
                select bm25(chunks_fts) as bm25_rank, {select_columns}
                from chunks_fts
                join chunks c on c.id = chunks_fts.rowid
                where chunks_fts match ?
                order by bm25_rank
                limit ?
                """,
                (match_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(row, -float(row["bm25_rank"] or 0.0)) for row in rows]

    def _exact_candidates(self, query: str, *, limit: int) -> Iterable[Tuple[sqlite3.Row, float]]:
        assert self.conn is not None
        clauses: List[str] = []
        params: List[Any] = []
        url_column = "source_url" if self.schema_variant == "clean_rag_data" else "url"
        quality_column = "quality" if self.schema_variant == "clean_rag_data" else "quality_score"
        if self.schema_variant == "clean_rag_data":
            select_columns = """
                id, chunk_uid, doc_id, chunk_index, text, title, source_path,
                source_tier as source_type, category, source_url as url, host,
                date, quality as quality_score, metadata_json
            """
        else:
            select_columns = "*"

        for code in _query_course_codes(query):
            clauses.append("(upper(title)=? or upper(text) like ? or upper(text) like ?)")
            params.extend([code, f"%课程代码: {code}%", f"%{code}%"])

        emails = EMAIL_RE.findall(query)
        for email in emails:
            clauses.append("lower(text) like ?")
            params.append(f"%{email.lower()}%")

        years = _query_years(query)
        for year in years:
            clauses.append("(date like ? or text like ? or title like ?)")
            params.extend([f"{year}%", f"%{year}%", f"%{year}%"])

        if _profileish_query(query):
            for phrase in _query_exact_cjk_phrases(query):
                clauses.append(f"(title like ? or text like ? or {url_column} like ? or source_path like ?)")
                params.extend([f"%{phrase}%", f"%{phrase}%", f"%{phrase}%", f"%{phrase}%"])

        if not clauses:
            return []

        rows = self.conn.execute(
            f"""
            select {select_columns}
            from chunks
            where {' or '.join(clauses)}
            order by {quality_column} desc, id
            limit ?
            """,
            (*params, limit),
        ).fetchall()
        return [(row, 20.0) for row in rows]

    def _clean_supplemental_candidates(self, query: str, *, limit: int) -> Iterable[Tuple[sqlite3.Row, float]]:
        assert self.conn is not None
        clauses: List[str] = []
        params: List[Any] = []
        years = _query_years(query)
        program_codes = _query_program_codes(query)

        if _has_degree_intent(query):
            base = ["category = 'sist_degree_programs'"]
            if years:
                year_clauses = []
                for year in years:
                    year_clauses.append("(source_url like ? or title like ? or text like ?)")
                    params.extend([f"%{year}%", f"%{year}%", f"%{year}%"])
                base.append("(" + " or ".join(year_clauses) + ")")
            if program_codes:
                code_clauses = []
                for code in program_codes:
                    code_lower = code.lower()
                    if code == "CS":
                        code_clauses.append(
                            "(lower(source_url) like ? or lower(source_url) like ? or text like ? or lower(text) like ?)"
                        )
                        params.extend([f"%in%20{code_lower}.htm%", f"%in%20{code_lower}%", "%计算机科学与技术%", "%computer science%"])
                    elif code == "EE":
                        code_clauses.append(
                            "(lower(source_url) like ? or lower(source_url) like ? or text like ? or lower(text) like ? or lower(text) like ?)"
                        )
                        params.extend([f"%in%20{code_lower}.htm%", f"%in%20{code_lower}%", "%电子信息工程%", "%electrical%", "%electronic%"])
                    else:
                        code_clauses.append("(lower(source_url) like ? or title like ? or text like ?)")
                        params.extend([f"%{code_lower}%", f"%{code}%", f"%{code}%"])
                base.append("(" + " or ".join(code_clauses) + ")")
            if any(term in query.lower() for term in ("bachelor", "本科")):
                base.append("(lower(source_url) like '%undergraduate%' or lower(source_url) like '%bachelor%')")
            clauses.append("(" + " and ".join(base) + ")")

        if _has_course_intent(query):
            base = ["category = 'sist_courses'"]
            codes = _query_course_codes(query)
            if codes:
                code_clauses = []
                for code in codes:
                    code_clauses.append("(title like ? or text like ? or source_url like ?)")
                    params.extend([f"%{code}%", f"%{code}%", f"%{code}%"])
                base.append("(" + " or ".join(code_clauses) + ")")
            if years:
                year_clauses = []
                for year in years:
                    year_clauses.append("(source_url like ? or title like ? or text like ?)")
                    params.extend([f"%{year}%", f"%{year}%", f"%{year}%"])
                base.append("(" + " or ".join(year_clauses) + ")")
            clauses.append("(" + " and ".join(base) + ")")

        if _profileish_query(query):
            names = [
                phrase
                for phrase in _query_exact_cjk_phrases(query)
                if phrase not in {"教授", "教师", "导师", "邮箱", "办公室", "研究方向"}
            ]
            if names:
                base = ["category = 'sist_faculty'"]
                name_clauses = []
                for name in names:
                    name_clauses.append("(title like ? or text like ? or source_url like ?)")
                    params.extend([f"%{name}%", f"%{name}%", f"%{name}%"])
                base.append("(" + " or ".join(name_clauses) + ")")
                clauses.append("(" + " and ".join(base) + ")")

        if _has_sist_overview_intent(query):
            clauses.append(
                "("
                "category = 'sist_overview' "
                "or source_url in ('https://sist.shanghaitech.edu.cn/', 'https://sist.shanghaitech.edu.cn/sist_en/', 'https://faculty.sist.shanghaitech.edu.cn/') "
                "or lower(text) like '%sist at a glance%' "
                "or lower(text) like '%about sist%' "
                "or lower(text) like '%vision and mission%'"
                ")"
            )

        if not clauses:
            return []

        rows = self.conn.execute(
            f"""
            select
                id, chunk_uid, doc_id, chunk_index, text, title, source_path,
                source_tier as source_type, category, source_url as url, host,
                date, quality as quality_score, metadata_json
            from chunks
            where {' or '.join(clauses)}
            order by source_tier = 'verified_seed' desc,
                     source_tier = 'live_official' desc,
                     quality desc,
                     id asc
            limit ?
            """,
            (*params, limit),
        ).fetchall()
        return [(row, 26.0) for row in rows]

    def _rerank(self, query: str, row: sqlite3.Row, raw_score: float) -> float:
        text = str(row["text"] or "")
        title = str(row["title"] or "")
        haystack = f"{title}\n{text}"
        haystack_lower = haystack.lower()
        source_type = str(row["source_type"] or "")
        category = str(row["category"] or "")
        score = raw_score
        score += SOURCE_PRIORITY.get(source_type, 1.0)
        score += float(row["quality_score"] or 0.0) * 4.0

        if category in STRUCTURED_CATEGORIES:
            score += 1.5
        if source_type == "pdf_md":
            score -= 2.0

        tokens = _normalize_query_tokens(_expand_query(query))
        for token in tokens:
            token_lower = token.lower()
            if token_lower in title.lower():
                score += min(len(token), 8) * 1.2
            elif token_lower in haystack_lower:
                score += min(len(token), 8) * 0.35

        for code in _query_course_codes(query):
            if code == title.upper():
                score += 35.0
            if f"课程代码: {code}" in text or f"# {code}" in text:
                score += 45.0
            elif code in haystack.upper():
                score += 18.0
            else:
                score -= 30.0
            if any(term in query for term in ("任课", "老师", "教师", "学分", "课程")):
                if "任课教师:" in text or "课程代码:" in text or category in {"courses", "sist_courses", "courses_clean", "courses_unified"}:
                    score += 10.0
                else:
                    score -= 8.0

        years = _query_years(query)
        if years:
            row_years = set(YEAR_RE.findall(haystack))
            date = str(row["date"] or "")
            has_query_year = any(date.startswith(year) or year in haystack for year in years)
            if has_query_year:
                score += 16.0
                if any(date.startswith(year) for year in years):
                    score += 8.0
            else:
                score -= 22.0
            if row_years and not any(year in row_years for year in years):
                score -= 14.0
            title_years = set(YEAR_RE.findall(title))
            if title_years and not any(year in title_years for year in years):
                score -= 18.0

        if any(term in query for term in ("邮箱", "email", "邮件", "电话", "office", "办公室")):
            if category in {"contacts", "faculty_members", "sist_faculty", "professors_enriched", "leadership_roles"}:
                score += 8.0
            if "邮箱:" in text or "email" in haystack_lower or "办公室" in text or "office" in haystack_lower:
                score += 4.0

        if _profileish_query(query):
            phrases = _query_exact_cjk_phrases(query)
            for phrase in phrases:
                if title == phrase:
                    score += 90.0
                elif _contains_exact_cjk_phrase(title, phrase):
                    score += 18.0
                if (
                    _contains_exact_cjk_phrase(text[:800], phrase)
                    and (
                        text.startswith(phrase + " ")
                        or text.startswith(phrase + "\n")
                        or f"姓名:{phrase}" in text
                        or f"姓名: {phrase}" in text
                    )
                ):
                    score += 42.0
                elif _contains_exact_cjk_phrase(haystack, phrase):
                    score += 8.0
            if any(marker in text for marker in ("个人主页:", "研究方向:", "邮箱:", "办公室:", "博士毕业院校:")):
                score += 12.0
            if category in {"faculty", "faculty_members", "sist_faculty", "professors_enriched"}:
                score += 8.0

        if "日常管理" in query and "上海科技大学" in query:
            if "上海市人民政府负责日常管理" in haystack or "负责日常管理的全日制普通高等学校" in haystack:
                score += 45.0
            if any(term in title + text[:800] for term in ("仪器设备", "科研经费", "基建项目", "采购")):
                score -= 28.0

        if any(term in query for term in ("开放时间", "图书馆")) and "library" in str(row["host"] or ""):
            score += 10.0

        if self.schema_variant == "clean_rag_data":
            score += _clean_intent_boost(query, row)

        return score

    def _dedupe(self, hits: Sequence[UnifiedSearchHit], *, top_k: int) -> List[UnifiedSearchHit]:
        out: List[UnifiedSearchHit] = []
        seen_text = set()
        seen_path_title = set()
        for hit in hits:
            text_key = re.sub(r"\s+", "", hit.text[:260])
            path_title_key = (hit.path, hit.title, hit.chunk_index)
            if text_key in seen_text or path_title_key in seen_path_title:
                continue
            seen_text.add(text_key)
            seen_path_title.add(path_title_key)
            out.append(hit)
            if len(out) >= top_k:
                break
        return out
