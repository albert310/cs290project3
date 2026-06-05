from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DATA_ROOT = PROJECT_ROOT / "rag_data"
DEFAULT_DB_PATH = RAG_DATA_ROOT / "db" / "shanghaitech_sist.sqlite"

ASCII_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#.@/-]*|\d+(?:\.\d+)?")
CJK_SEGMENT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
CJK_STOP_UNIGRAMS = set("的一是在了和与及或对中为有个们于年月日")
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
DATE_RE = re.compile(r"(20\d{2})[-_/年.](\d{1,2})(?:[-_/月.](\d{1,2}))?")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")

NOISE_EXACT = {
    "导航",
    "网站导航",
    "站内搜索",
    "首 页",
    "首页",
    "返回",
    "返回顶部",
    "中文",
    "English",
    "EN",
    "Menu",
    "Newsletter",
    "更多+",
    "Copyright",
    "版权所有",
    "打印",
    "关闭",
    "分享到",
    "扫一扫",
    "官方微信",
    "学校首页",
    "学院首页",
    "学院概况",
    "使命和愿景",
    "学院介绍",
    "院长寄语",
    "顾问委员会",
    "院务委员会",
    "科学研究",
    "研究中心",
    "联合实验室",
    "省部级研究中心",
    "规章制度",
    "采购公告",
    "EHS",
    "最新动向",
    "常用资料",
    "科研设备",
    "师资队伍",
    "常任教授",
    "特聘教授",
    "访问教授",
    "研究人员",
    "支撑人员",
    "行政人员",
    "学生培养",
    "本科生培养",
    "研究生培养",
    "毕业答辩流程",
    "培养方案",
    "培养与学位授予相关细则",
    "远程答辩公示",
    "本硕博课程体系",
    "本科生教学教务",
    "研究生教学教务",
    "技能培训",
    "学生风采",
    "资料下载",
    "招生工作",
    "本科生招生",
    "研究生招生介绍一览",
    "硕士研究生招生",
    "博士研究生招生",
    "通知公告",
    "报考指南",
    "招生宣传",
    "师生风采",
    "招生Q&A",
    "暑期项目",
    "实习就业",
    "实习信息",
    "就业信息",
    "国际交流",
    "人才招聘",
    "Faculty Recruitment",
    "Researcher Recruitment",
    "Job Opportunities",
    "Undergraduate",
    "Graduate",
}

NOISE_PATTERNS = (
    re.compile(r"^\s*浏览次数[:：]?\s*\d*\s*$"),
    re.compile(r"^\s*发布时间[:：]?\s*$"),
    re.compile(r"^\s*当前位置[:：]?\s*$"),
    re.compile(r"^\s*沪公网安备"),
    re.compile(r"^\s*Copyright\b", flags=re.I),
    re.compile(r"^\s*地址[:：].*邮编[:：]?\s*\d+"),
    re.compile(r"^\s*(上一条|下一条|打印|关闭|分享)\s*$"),
    re.compile(r"^\s*!?\[[^\]]{0,40}\]\([^)]*\)\s*$"),
    re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"),
)

EXTERNAL_NOISE_HOSTS = {
    "www.who.int",
    "who.int",
    "arxiv.org",
    "doi.org",
}

CATEGORY_KEYWORDS = {
    "university_overview": (
        "学校简介",
        "about shanghaitech",
        "shanghaitech university",
        "上海科技大学",
        "found",
        "established",
        "中国科学院",
        "上海市人民政府",
    ),
    "university_contact": (
        "contact",
        "联系我们",
        "地址",
        "huaxia",
        "华夏中路",
        "campus map",
    ),
    "sist_overview": (
        "信息科学与技术学院",
        "school of information science and technology",
        "sist",
    ),
    "sist_faculty": (
        "faculty",
        "师资",
        "教授",
        "副教授",
        "助理教授",
        "邮箱",
        "email",
        "research interests",
        "研究方向",
    ),
    "sist_courses": (
        "course",
        "courses",
        "课程",
        "课程代码",
        "学分",
        "任课",
        "instructor",
        "credit",
    ),
    "sist_degree_programs": (
        "培养方案",
        "degree program",
        "program requirements",
        "undergraduate",
        "graduate",
        "bachelor",
        "master",
        "ph.d",
        "博士",
        "硕士",
        "本科",
    ),
    "sist_research": (
        "research",
        "科研",
        "实验室",
        "课题组",
        "publication",
        "论文",
    ),
    "sist_news_events": (
        "news",
        "event",
        "notice",
        "新闻",
        "通知",
        "公告",
        "讲座",
        "seminar",
    ),
}


@dataclass(frozen=True)
class CleanedText:
    text: str
    raw_lines: int
    kept_lines: int
    removed_lines: int


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def normalize_space(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_for_hash(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_allowed_host(host: str, allowed_hosts: Sequence[str], allowed_suffixes: Sequence[str]) -> bool:
    host = host.lower()
    if host in {item.lower() for item in allowed_hosts}:
        return True
    return any(host.endswith(suffix.lower()) for suffix in allowed_suffixes)


def line_is_noise(line: str) -> bool:
    line = normalize_space(line).strip(" #|*-_")
    if not line:
        return True
    if line in NOISE_EXACT:
        return True
    if set(line) <= {"-", "_", "*", "=", "#", "|", " ", "!", "."}:
        return True
    if len(line) <= 2 and not re.search(r"\w|[\u4e00-\u9fff]", line):
        return True
    for pattern in NOISE_PATTERNS:
        if pattern.search(line):
            return True
    return False


def strip_markdown_and_urls(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = URL_RE.sub(" ", text)
    return normalize_space(text)


def clean_text(text: str, *, min_line_chars: int = 2) -> CleanedText:
    text = normalize_space(text)
    text = re.sub(r"\[\]\([^)]*\)", " ", text)
    lines = text.splitlines()
    kept: List[str] = []
    local_counts: Dict[str, int] = {}
    for raw_line in lines:
        line = strip_markdown_and_urls(raw_line)
        line = line.strip()
        if line_is_noise(line):
            continue
        if len(line) < min_line_chars:
            continue
        if line.count("](") >= 2:
            continue
        local_counts[line] = local_counts.get(line, 0) + 1
        if local_counts[line] > 2:
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return CleanedText(
        text=cleaned,
        raw_lines=len(lines),
        kept_lines=len(kept),
        removed_lines=max(len(lines) - len(kept), 0),
    )


def looks_garbled(text: str) -> bool:
    probe = text[:5000]
    if "\ufffd" in probe or "â" in probe or "Ã" in probe:
        return True
    cjk = sum(1 for ch in probe if "\u4e00" <= ch <= "\u9fff")
    mojibake = sum(probe.count(item) for item in ("ä", "å", "ç", "è", "æ", "é", "Ϣѧ", "רҵ", "ѧλ"))
    if len(probe) >= 300 and cjk < 30 and mojibake > 30:
        return True
    return False


def extract_date(text: str, *fallbacks: str) -> str:
    for value in list(fallbacks) + [text[:6000]]:
        if not value:
            continue
        for match in DATE_RE.finditer(str(value)):
            year, month, day = match.groups()
            month_i = int(month)
            day_i = int(day) if day else None
            if not 1 <= month_i <= 12:
                continue
            if day_i is not None and not 1 <= day_i <= 31:
                continue
            if day_i is None:
                return f"{int(year):04d}-{month_i:02d}"
            return f"{int(year):04d}-{month_i:02d}-{day_i:02d}"
    return ""


def infer_category(title: str, text: str, url: str = "", fallback: str = "general") -> str:
    haystack = f"{title}\n{text[:5000]}\n{url}".lower()
    path = urlparse(url).path.lower()
    host = urlparse(url).netloc.lower()
    is_profileish_path = (
        "/faculty/" in path
        or re.search(r"/[a-z][a-z0-9]{2,12}/(?:main\.htm)?$", path)
        or re.search(r"/[a-z][a-z0-9]{2,12}/?$", path)
    )
    if (host == "faculty.sist.shanghaitech.edu.cn" or host == "sist.shanghaitech.edu.cn") and is_profileish_path:
        if any(term in haystack for term in ("email", "邮箱", "research", "研究方向", "professor", "教授", "副教授", "助理教授", "博导")):
            return "sist_faculty"
    if any(marker in path for marker in ("course", "courses", "course%20schedule")):
        return "sist_courses"
    if any(marker in path for marker in ("degree", "degree%20program", "pyfa", "programmes", "programs")):
        return "sist_degree_programs"
    scores: Dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in haystack:
                score += 1
        if score:
            scores[category] = score
    if scores:
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return fallback


def quality_score(*, source_tier: str, category: str, url: str, text: str, title: str = "") -> float:
    score = 0.45
    if source_tier == "verified_seed":
        score += 0.45
    elif source_tier == "live_official":
        score += 0.36
    elif source_tier == "local_official_mirror":
        score += 0.30
    elif source_tier == "llm_curated":
        score += 0.26

    host = host_of(url)
    if host in {"www.shanghaitech.edu.cn", "shanghaitech.edu.cn"}:
        score += 0.08
    if host in {"sist.shanghaitech.edu.cn", "faculty.sist.shanghaitech.edu.cn"} or host.endswith(".sist.shanghaitech.edu.cn"):
        score += 0.10
    if url:
        score += 0.05
    if category in {"university_overview", "sist_overview", "sist_faculty", "sist_courses", "sist_degree_programs"}:
        score += 0.05
    if EMAIL_RE.search(text):
        score += 0.02
    if len(text) < 120:
        score -= 0.08
    if looks_garbled(text):
        score -= 0.25
    if host in EXTERNAL_NOISE_HOSTS:
        score -= 0.40
    return max(0.0, min(score, 0.99))


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    lowered = text.lower()
    tokens.extend(match.group(0) for match in ASCII_TOKEN_RE.finditer(lowered))
    for match in CJK_SEGMENT_RE.finditer(text):
        segment = match.group(0)
        if len(segment) <= 12:
            tokens.append(segment)
        for char in segment:
            if char not in CJK_STOP_UNIGRAMS:
                tokens.append(char)
        for n in (2, 3, 4):
            if len(segment) >= n:
                tokens.extend(segment[i : i + n] for i in range(len(segment) - n + 1))
    return tokens


def make_search_text(*values: str) -> str:
    return " ".join(tokenize("\n".join(value for value in values if value)))


def quote_fts_token(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def make_match_query(query: str, *, max_terms: int = 64) -> str:
    seen = set()
    ordered: List[str] = []
    for token in sorted(tokenize(query), key=lambda item: (len(item), item), reverse=True):
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
        if len(ordered) >= max_terms:
            break
    return " OR ".join(quote_fts_token(token) for token in ordered)


def split_text(text: str, *, max_chars: int = 900, overlap_chars: int = 120, min_chars: int = 120) -> Iterator[str]:
    text = normalize_space(text)
    if not text:
        return
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    current: List[str] = []
    current_len = 0

    def flush() -> Iterator[str]:
        nonlocal current, current_len
        if current:
            chunk = "\n".join(current).strip()
            if len(chunk) >= min_chars:
                yield chunk
            current = []
            current_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            yield from flush()
            start = 0
            while start < len(para):
                end = min(len(para), start + max_chars)
                piece = para[start:end].strip()
                if len(piece) >= min_chars:
                    yield piece
                if end >= len(para):
                    break
                start = max(start + 1, end - overlap_chars)
            continue
        next_len = current_len + len(para) + (1 if current else 0)
        if next_len > max_chars:
            yield from flush()
        current.append(para)
        current_len += len(para) + (1 if current_len else 0)
    yield from flush()


def make_snippet(text: str, query: str, *, max_chars: int = 320) -> str:
    if len(text) <= max_chars:
        return text
    candidates = [token for token in tokenize(query) if len(token) >= 2]
    start = -1
    lowered = text.lower()
    for token in sorted(candidates, key=len, reverse=True):
        start = lowered.find(token.lower())
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


def open_db(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
