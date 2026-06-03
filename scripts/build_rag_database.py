from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.keyword_search import tokenize


URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
DATE_RE = re.compile(r"(20\d{2})[-_/年.](\d{1,2})(?:[-_/月.](\d{1,2}))?")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
MARKDOWN_LINK_ONLY_RE = re.compile(r"^\s*\[?\s*\]?\([^)]*\)\s*$")
MARKDOWN_LINK_RE = re.compile(r"(?P<prefix>\*\s*)?\[(?P<label>[^\]]{0,60})\]\([^)]*\)")
HTML_ENTITY_RE = re.compile(r"&(?:nbsp|amp|lt|gt|quot);", flags=re.IGNORECASE)
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,}\d{2,4}[A-Z]?\b")
CID_RE = re.compile(r"\(cid:\d+\)")
STATIC_ASSET_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".css",
    ".js",
    ".webp",
    ".bmp",
}


GLOBAL_NOISE_LINES = {
    "导航",
    "导航菜单",
    "### 导航",
    "### 网站导航",
    "### 语言",
    "网站导航",
    "站内搜索",
    "Menu",
    "EN",
    "中文",
    "English",
    "Newsletter",
    "新闻",
    "通知公告",
    "友情链接",
    "更多+",
    "官方微信",
    "学校首页",
    "首 页",
    "首页",
    "返回",
    "返回顶部",
    "None",
    "!",
    "! !",
    "版权所有",
    "Copyright © 上海科技大学 版权所有",
    "Copyright © 2019 上海科技大学免疫化学研究所 版权所有",
    "沪公网安备 31011502006855号",
    "地址：上海市浦东新区华夏中路393号 邮编：201210",
    "地址：上海市浦东新区华夏中路393号",
    "邮编：201210",
    "上海市浦东新区华夏中路393号 201210（浦东校区）",
    "上海市徐汇区岳阳路319号8号楼 200031（岳阳路校区）",
    "[](javascript:;)",
    "[ ](javascript:;)",
    "[返回](javascript:;)原图",
    "---",
    "| --- |",
    "| --- | --- |",
}

NOISE_LINK_LABELS = {
    "!",
    "En",
    "中文",
    "English",
    "官方微信",
    "新闻",
    "公告",
    "学校概况",
    "校务委员会",
    "校领导",
    "院所设置",
    "科学研究",
    "科研进展",
    "联系我们",
    "学校简介",
    "机构设置",
    "大事记",
    "影像报刊",
    "学校首页",
}

NOISE_PATTERNS = (
    re.compile(r"^\s*浏览次数[:：]?\s*\d*\s*$"),
    re.compile(r"^\s*发布时间[:：]?\s*$"),
    re.compile(r"^\s*当前位置[:：]?\s*$"),
    re.compile(r"^\s*Copyright\b", flags=re.IGNORECASE),
    re.compile(r"^\s*沪公网安备"),
    re.compile(r"^\s*地址[:：].*邮编"),
    re.compile(r"^\s*(上一条|下一条|打印|关闭|分享|扫码分享)\s*$"),
    re.compile(r"^\s*\*?\s*\[[^\]]{0,24}\]\(javascript:;?\)\s*$"),
    re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$"),
    re.compile(r"^\s*<img\b", flags=re.IGNORECASE),
)

PDF_KEEP_TERMS = (
    "招生",
    "培养方案",
    "课程",
    "学位",
    "通知",
    "公告",
    "公示",
    "办法",
    "章程",
    "指南",
    "政策",
    "申请",
    "就业",
    "招聘",
    "采购",
    "招标",
    "报名",
    "奖学金",
    "规章",
)

PDF_RELEVANCE_TERMS = (
    "上海科技大学",
    "上科大",
    "ShanghaiTech",
    "SIST",
    "School of Information Science",
    "Information Science and Technology",
    "信息科学与技术学院",
    "生命科学与技术学院",
    "物质科学与技术学院",
    "创意与艺术学院",
    "创业与管理学院",
    "上海科技大学图书馆",
    "华夏中路393",
    "Huaxia Middle Road",
)

PDF_ACADEMIC_TERMS = (
    "abstract",
    "references",
    "arxiv",
    "doi",
    "ieee",
    "acm",
    "proceedings",
    "conference",
    "workshop",
    "slides",
    "poster",
    "acl",
    "emnlp",
    "iclr",
    "neurips",
    "cvpr",
    "iccv",
    "aaai",
    "ijcai",
    "iecon",
    "apec",
    "ecce",
    "ecceasia",
    "tpel",
    "tpe",
    "tie",
    "tia",
    "tvt",
    "mtt",
    "mwscas",
    "iscas",
    "icassp",
    "access",
    "jestpe",
    "ietpe",
    "iros",
    "icra",
)

TEXT_SOURCE_DIRS = (
    ("course_sist", Path("data/sist/jsonl/documents.jsonl"), "sist_text", 0.82),
    ("self_crawl", Path("data/shanghaitech_data/web"), "web", 0.76),
    ("self_crawl", Path("data/shanghaitech_data/text_pages"), "text_pages", 0.72),
    ("self_crawl", Path("data/shanghaitech_data/sist_raw"), "sist_raw", 0.70),
    ("self_crawl", Path("data/shanghaitech_data/training"), "training", 0.88),
    ("self_crawl", Path("data/shanghaitech_data/pdf_md"), "pdf_md", 0.48),
)

STRUCTURED_TABLES = {
    "faculty_members": 1.00,
    "courses": 0.92,
    "program_requirements": 0.90,
    "events": 0.86,
    "contacts": 0.86,
    "facilities": 0.86,
    "staff_members": 0.84,
    "program_sources": 0.78,
    "facts": 0.72,
    "leadership_roles": 0.68,
}

STRUCTURED_CONFIDENCE_THRESHOLDS = {
    "facts": 0.80,
    "leadership_roles": 0.82,
    "courses": 0.55,
    "events": 0.62,
    "program_requirements": 0.55,
    "program_sources": 0.60,
}

SELECTED_SELF_JSON = {
    "professors_enriched.json": 0.88,
    "sist_faculty.json": 0.92,
    "slst_faculty.json": 0.82,
    "courses_unified.json": 0.86,
    "courses_clean.json": 0.82,
    "sist_courses.json": 0.88,
    "course_teacher_map.json": 0.84,
    "prof_courses_full.json": 0.76,
}

FIELD_LABELS = {
    "school": "学院/单位",
    "name": "姓名",
    "name_en": "英文名",
    "title": "职称/标题",
    "role": "职务",
    "faculty_category": "教师类别",
    "category": "类别",
    "research_area": "研究方向",
    "email": "邮箱",
    "phone": "电话",
    "office": "办公室",
    "homepage": "主页",
    "profile_url": "个人主页",
    "course_code": "课程代码",
    "code": "课程代码",
    "course_name": "课程名称",
    "name_cn": "课程名称",
    "course_name_en": "英文课程名",
    "credits": "学分",
    "hours": "学时",
    "term": "学期",
    "semester": "学期",
    "instructor": "任课教师",
    "teacher": "任课教师",
    "teachers": "任课教师",
    "program_name": "项目/专业",
    "program": "项目/专业",
    "cohort": "年级",
    "degree": "学位层次",
    "requirement_type": "要求类型",
    "requirement_text": "要求",
    "min_credits": "最低学分",
    "event_type": "事件类型",
    "event_date": "事件日期",
    "published_at": "发布日期",
    "source_url": "来源 URL",
    "url": "URL",
    "evidence": "证据片段",
}

TEACHER_FIELD_KEYS = {"instructor", "teacher", "teachers"}
BAD_TEACHER_VALUE_RE = re.compile(r"(矩阵分析|下一年|教学中心|星期[一二三四五六日天]|课程表)")


@dataclass
class CleanResult:
    text: str
    raw_lines: int
    kept_lines: int
    removed_lines: int
    removed_blacklist: int


@dataclass
class CandidateDoc:
    source_dataset: str
    source_path: str
    source_type: str
    category: str
    title: str
    url: str
    host: str
    date: str
    text: str
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def doc_key(self) -> str:
        return f"{self.source_dataset}:{self.source_path}"


@dataclass
class BuildStats:
    docs_seen: Counter = field(default_factory=Counter)
    docs_kept: Counter = field(default_factory=Counter)
    docs_skipped: Counter = field(default_factory=Counter)
    chunks_inserted: Counter = field(default_factory=Counter)
    structured_records: Counter = field(default_factory=Counter)
    duplicate_chunks: int = 0
    short_docs: int = 0
    low_quality_docs: int = 0
    dynamic_blacklist_count: int = 0
    top_blacklist_lines: List[Tuple[str, int]] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def normalize_for_hash(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(item, dict):
                yield item


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "、".join(clean_scalar(item) for item in value if clean_scalar(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_placeholder_value(value: str) -> bool:
    text = normalize_line(value)
    if not text:
        return True
    return text in {"中文", "EN", "English", "Menu", "None", "站内搜索", "网站导航"}


def normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    line = line.replace("\u3000", " ")
    line = HTML_ENTITY_RE.sub(" ", line)
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def strip_markdown_inline(value: str) -> str:
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", value)
    value = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", value)
    value = URL_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" #|*-_\t")


def strip_noise_markdown_links(line: str) -> str:
    def replace(match: re.Match[str]) -> str:
        label = normalize_line(match.group("label")).strip(" !#|*-_\t")
        if not label:
            return " "
        if label in NOISE_LINK_LABELS or label.startswith("微信扫一扫"):
            return " "
        return match.group(0)

    return MARKDOWN_LINK_RE.sub(replace, line)


def clean_structured_markdown_value(value: str, key: str = "") -> str:
    text = re.sub(r"!\[.*?\]\([^)]*\)", " ", value)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)\s;]*\)?", " ", text)
    text = re.sub(r"\[\s*\]\(([^)]*)\)", r"\1", text)
    text = strip_noise_markdown_links(text)
    parts: List[str] = []
    for part in re.split(r"\s*;\s*", text):
        part = normalize_line(part)
        if not part:
            continue
        if key in {"research", "context", "evidence", "description"} and (
            "Copyright" in part
            or "沪公网安备" in part
            or "footlogo" in part
            or "_upload/tpl" in part
        ):
            continue
        parts.append(part)
    return "; ".join(parts).strip()


def clean_teacher_value(value: str) -> str:
    if not value:
        return ""
    parts = []
    for part in re.split(r"\s*[、;；]\s*", value):
        part = normalize_line(part)
        if not part or BAD_TEACHER_VALUE_RE.search(part):
            continue
        parts.append(part)
    return "、".join(parts)


def is_static_asset_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    if "@" in parsed.netloc:
        return True
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in STATIC_ASSET_EXTENSIONS):
        return True
    if "/_upload/tpl/" in path or "/template" in path:
        return True
    return False


def is_noise_line(line: str, dynamic_blacklist: Optional[set[str]] = None) -> bool:
    stripped = normalize_line(line)
    if not stripped:
        return True
    if stripped in GLOBAL_NOISE_LINES:
        return True
    if dynamic_blacklist and stripped in dynamic_blacklist:
        return True
    if TABLE_SEPARATOR_RE.match(stripped):
        return True
    if MARKDOWN_LINK_ONLY_RE.match(stripped):
        return True
    if stripped.startswith(("* [", "- [", "[")) and stripped.count("](") == 1 and not DATE_RE.search(stripped):
        label = strip_markdown_inline(stripped)
        if len(label) <= 24 or label in GLOBAL_NOISE_LINES:
            return True
    if set(stripped) <= {"-", "_", "*", "=", "#", "|", " ", "!"}:
        return True
    for pattern in NOISE_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def is_blacklist_candidate(line: str) -> bool:
    stripped = normalize_line(line)
    if not stripped:
        return False
    if stripped in GLOBAL_NOISE_LINES:
        return True
    if len(stripped) <= 36:
        return True
    if len(stripped) <= 120 and (
        stripped.startswith("* [")
        or stripped.startswith("[")
        or "javascript:;" in stripped
        or "Copyright" in stripped
        or "公网安备" in stripped
        or "邮编" in stripped
    ):
        return True
    return False


def clean_text(text: str, dynamic_blacklist: Optional[set[str]] = None) -> CleanResult:
    text = unicodedata.normalize("NFKC", text or "")
    text = unquote(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"\[\]\([^)]*\)", " ", text)
    text = re.sub(r"\s+(?=\* \[[^\]]{1,80}\]\()", "\n", text)
    text = re.sub(r"\s+(?=#{1,6}\s+)", "\n", text)
    lines = text.splitlines()
    kept: List[str] = []
    line_counts: Counter = Counter()
    removed_blacklist = 0

    for raw_line in lines:
        line = re.sub(r"!\[.*?\]\([^)]*\)", " ", raw_line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)
        line = re.sub(r"!\[[^\]]*\]\([^)\s;]*\)?", " ", line)
        line = re.sub(r"\[\s*\]\([^)]*\)", " ", line)
        line = normalize_line(line)
        line = strip_noise_markdown_links(line)
        if "分享到" in line:
            line = line.split("分享到", 1)[0]
        line = normalize_line(line)
        if not line:
            continue
        if is_noise_line(line, dynamic_blacklist=dynamic_blacklist):
            if dynamic_blacklist and line in dynamic_blacklist:
                removed_blacklist += 1
            continue
        if line.count("](") >= 3 and not DATE_RE.search(line):
            continue
        if line.startswith("Source:") and re.search(r"[0-9a-f]{32,}", line):
            continue
        if line.startswith("URL:") and URL_RE.search(line):
            continue
        if line.startswith("![](") or line.startswith("!["):
            continue
        if line_counts[line] >= 2:
            continue
        line_counts[line] += 1
        kept.append(line)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return CleanResult(
        text=cleaned,
        raw_lines=len(lines),
        kept_lines=len(kept),
        removed_lines=max(len(lines) - len(kept), 0),
        removed_blacklist=removed_blacklist,
    )


def looks_garbled(text: str) -> bool:
    if CID_RE.search(text):
        return True
    if "â" in text or "ä ̧" in text:
        return True
    probe = text[:5000]
    if "Ã" in probe:
        return True
    if "\ufffd" in probe:
        return True
    cjk = sum(1 for ch in probe if "\u4e00" <= ch <= "\u9fff")
    cyrillic = sum(1 for ch in probe if "\u0400" <= ch <= "\u04ff")
    greek = sum(1 for ch in probe if "\u0370" <= ch <= "\u03ff")
    mojibake = sum(probe.count(item) for item in ("ä", "å", "ç", "è", "æ", "é", "ï1⁄4", "3⁄4"))
    if len(probe) >= 300 and cjk < 20 and (cyrillic + greek) > 80:
        return True
    if len(probe) >= 300 and cjk < 30 and mojibake > 50:
        return True
    suspicious_terms = ("ϢѧԺ", "Ϣѧ", "רҵ", "ѧλ", "汾ŵ", "ھ", "ʦ", "ƽ", "о", "Ƹ", "εͳ")
    return sum(1 for term in suspicious_terms if term in probe) >= 1


def extract_title(text: str, fallback: str = "") -> str:
    for line in text.splitlines()[:80]:
        match = HEADING_RE.match(line)
        if match:
            title = normalize_line(strip_markdown_inline(match.group(1)))
            if title and not is_noise_line(title) and "http" not in title.lower():
                return title[:180]
    for line in text.splitlines()[:80]:
        title = normalize_line(strip_markdown_inline(re.sub(r"^#+\s*", "", line)))
        if title and not is_noise_line(title) and len(title) >= 2 and "http" not in title.lower():
            return title[:180]
    return strip_markdown_inline(fallback)[:180]


def extract_url(text: str, fallback: str = "") -> str:
    candidates: List[str] = []
    for line in text.splitlines()[:80]:
        if "URL:" in line or "http" in line:
            candidates.extend(match.group(0).rstrip("。.,，") for match in URL_RE.finditer(line))
    candidates.extend(match.group(0).rstrip("。.,，") for match in URL_RE.finditer(text[:4000]))
    if fallback:
        candidates.append(fallback)
    for candidate in candidates:
        if not is_static_asset_url(candidate):
            return candidate
    return ""


def extract_date(text: str, *fallbacks: str) -> str:
    for value in fallbacks:
        if value:
            match = DATE_RE.search(str(value))
            if match:
                year, month, day = match.groups()
                if day:
                    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
                return f"{int(year):04d}-{int(month):02d}"
    for match in DATE_RE.finditer(text[:8000]):
        year, month, day = match.groups()
        if day:
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        return f"{int(year):04d}-{int(month):02d}"
    return ""


def host_from_url_or_path(url: str, source_path: str) -> str:
    parts = Path(source_path).parts
    if "web" in parts:
        index = parts.index("web")
        if len(parts) > index + 1:
            host = parts[index + 1].lower()
            if "@" not in host:
                return host
    if len(parts) >= 2 and parts[0] == "web":
        host = parts[1].lower()
        if "@" not in host:
            return host
    if url and not is_static_asset_url(url):
        parsed = urlparse(url)
        if parsed.netloc and "@" not in parsed.netloc:
            return parsed.netloc.lower()
    return ""


def infer_category_from_path(source_type: str, path: Path) -> str:
    name = path.as_posix().lower()
    if "faculty" in name or "professor" in name or "师资" in name:
        return "faculty"
    if "course" in name or "课程" in name or "pyfa" in name or "培养方案" in name:
        return "program"
    if "admission" in name or "招生" in name or "apply" in name:
        return "admission"
    if "career" in name or "jobs" in name or "招聘" in name:
        return "career"
    if "news" in name or "新闻" in name:
        return "news"
    if source_type == "pdf_md":
        return "pdf"
    if source_type == "training":
        return "program"
    return source_type


def pdf_should_skip(text: str, path: Path) -> bool:
    probe = (path.name + "\n" + text[:5000]).lower()
    relevance_probe = path.name + "\n" + text[:8000]
    relevance_probe_lower = relevance_probe.lower()
    if not any(term in relevance_probe or term.lower() in relevance_probe_lower for term in PDF_RELEVANCE_TERMS):
        return True
    keep = sum(1 for term in PDF_KEEP_TERMS if term.lower() in probe)
    academic = sum(1 for term in PDF_ACADEMIC_TERMS if term.lower() in probe)
    if keep:
        return False
    return academic >= 2


def document_should_skip_as_academic_pdf(doc: CandidateDoc, cleaned_text: str) -> bool:
    probe_path = (doc.source_path + "\n" + doc.url).lower()
    if ".pdf" not in probe_path and doc.source_type != "pdf_md":
        return False
    probe = probe_path + "\n" + cleaned_text[:5000].lower()
    keep = sum(1 for term in PDF_KEEP_TERMS if term.lower() in probe)
    academic = sum(1 for term in PDF_ACADEMIC_TERMS if term.lower() in probe)
    if keep:
        return False
    return academic >= 1


def adjusted_quality(base: float, text: str, source_type: str, category: str, host: str, path: str) -> float:
    score = base
    lower = (text[:5000] + "\n" + path).lower()
    if host.endswith("shanghaitech.edu.cn"):
        score += 0.04
    if "sist.shanghaitech.edu.cn" in host:
        score += 0.03
    if category in {"faculty", "program", "admission"}:
        score += 0.04
    if source_type == "pdf_md":
        if any(term.lower() in lower for term in PDF_KEEP_TERMS):
            score += 0.18
        if any(term in lower for term in PDF_ACADEMIC_TERMS):
            score -= 0.22
    if len(text) < 300:
        score -= 0.08
    return max(0.05, min(score, 1.0))


def iter_sist_text_candidates(data_root: Path) -> Iterator[CandidateDoc]:
    docs_path = data_root / "sist/jsonl/documents.jsonl"
    texts_root = data_root / "sist"
    for row in read_jsonl(docs_path):
        text_path = clean_scalar(row.get("text_path"))
        if not text_path:
            continue
        rel = Path(text_path.replace("\\", "/"))
        path = texts_root / rel
        if not path.exists():
            continue
        text = read_text(path)
        title = clean_scalar(row.get("title")) or extract_title(text, path.stem)
        url = clean_scalar(row.get("canonical_url")) or clean_scalar(row.get("url")) or extract_url(text)
        host = clean_scalar(row.get("host")) or host_from_url_or_path(url, path.as_posix())
        date = extract_date(
            text,
            clean_scalar(row.get("valid_from")),
            clean_scalar(row.get("source_published_at")),
            clean_scalar(row.get("fetched_at")),
        )
        yield CandidateDoc(
            source_dataset="course_sist",
            source_path=path.relative_to(PROJECT_ROOT).as_posix(),
            source_type="sist_text",
            category=clean_scalar(row.get("category")) or "sist",
            title=title,
            url=url,
            host=host,
            date=date,
            text=text,
            quality_score=0.82,
            metadata={"document_id": row.get("id"), "sha256": row.get("sha256")},
        )


def iter_self_text_candidates(data_root: Path) -> Iterator[CandidateDoc]:
    base = data_root / "shanghaitech_data"
    for _, rel_dir, source_type, base_quality in TEXT_SOURCE_DIRS[1:]:
        root = PROJECT_ROOT / rel_dir
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            lower_name = path.name.lower()
            if lower_name.endswith((".js.md", ".css.md")) or "chunk-vendors" in lower_name:
                continue
            rel_under_base = path.relative_to(base).as_posix() if base in path.parents else path.as_posix()
            if source_type == "web":
                parts = Path(rel_under_base).parts
                if len(parts) >= 2 and "@" in parts[1]:
                    continue
            text = read_text(path)
            if source_type == "pdf_md" and pdf_should_skip(text, path):
                continue
            rel_path = path.relative_to(PROJECT_ROOT).as_posix()
            title = extract_title(text, path.stem)
            url = extract_url(text)
            host = host_from_url_or_path(url, rel_under_base)
            date = extract_date(text, path.name)
            category = infer_category_from_path(source_type, path)
            yield CandidateDoc(
                source_dataset="self_crawl",
                source_path=rel_path,
                source_type=source_type,
                category=category,
                title=title,
                url=url,
                host=host,
                date=date,
                text=text,
                quality_score=base_quality,
                metadata={},
            )


def iter_text_candidates(data_root: Path) -> Iterator[CandidateDoc]:
    yield from iter_sist_text_candidates(data_root)
    yield from iter_self_text_candidates(data_root)


def build_dynamic_line_blacklist(candidates: Sequence[CandidateDoc]) -> Tuple[set[str], List[Tuple[str, int]]]:
    counts: Counter = Counter()
    for doc in candidates:
        seen = set()
        for line in doc.text.splitlines():
            stripped = normalize_line(line)
            if is_blacklist_candidate(stripped):
                seen.add(stripped)
        counts.update(seen)

    threshold = max(20, int(max(len(candidates), 1) * 0.018))
    blacklist = {
        line
        for line, count in counts.items()
        if count >= threshold and (len(line) <= 64 or "javascript" in line or "Copyright" in line or "公网安备" in line)
    }
    blacklist.update(GLOBAL_NOISE_LINES)
    return blacklist, counts.most_common(40)


def split_text_to_chunks(text: str, *, max_chars: int = 950, overlap_chars: int = 120, min_chars: int = 120) -> List[str]:
    units = [unit.strip() for unit in re.split(r"\n{2,}", text) if unit.strip()]
    if len(units) <= 1:
        units = [line.strip() for line in text.splitlines() if line.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunk = "\n".join(current).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        current = []
        current_len = 0

    for unit in units:
        if len(unit) > max_chars:
            flush()
            start = 0
            while start < len(unit):
                end = min(len(unit), start + max_chars)
                piece = unit[start:end].strip()
                if len(piece) >= min_chars:
                    chunks.append(piece)
                if end >= len(unit):
                    break
                start = max(end - overlap_chars, start + 1)
            continue

        next_len = current_len + len(unit) + (1 if current else 0)
        if next_len > max_chars:
            flush()
        current.append(unit)
        current_len += len(unit) + (1 if current_len else 0)
    flush()
    return chunks


def make_search_text(title: str, text: str, metadata: Optional[Mapping[str, Any]] = None) -> str:
    values = [title, text]
    if metadata:
        for key in ("category", "host", "url", "source_path", "date"):
            value = metadata.get(key)
            if value:
                values.append(str(value))
    tokens = tokenize("\n".join(values))
    return " ".join(tokens)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS documents;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS structured_records;
        DROP TABLE IF EXISTS build_events;
        DROP TABLE IF EXISTS chunks_fts;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_key TEXT NOT NULL UNIQUE,
            source_dataset TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            category TEXT,
            title TEXT,
            url TEXT,
            host TEXT,
            date TEXT,
            language TEXT,
            raw_chars INTEGER NOT NULL,
            cleaned_chars INTEGER NOT NULL,
            quality_score REAL NOT NULL,
            content_hash TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_uid TEXT NOT NULL UNIQUE,
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            title TEXT,
            source_dataset TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            category TEXT,
            url TEXT,
            host TEXT,
            date TEXT,
            quality_score REAL NOT NULL,
            content_hash TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        );

        CREATE TABLE structured_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER,
            source_dataset TEXT NOT NULL,
            source_path TEXT NOT NULL,
            table_name TEXT NOT NULL,
            record_id TEXT,
            title TEXT,
            source_url TEXT,
            quality_score REAL NOT NULL,
            raw_json TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id)
        );

        CREATE TABLE build_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            source_path TEXT,
            message TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            search_text,
            chunk_uid UNINDEXED,
            tokenize='unicode61'
        );

        CREATE INDEX idx_chunks_source_type ON chunks(source_type);
        CREATE INDEX idx_chunks_category ON chunks(category);
        CREATE INDEX idx_chunks_host ON chunks(host);
        CREATE INDEX idx_chunks_date ON chunks(date);
        CREATE INDEX idx_chunks_quality ON chunks(quality_score);
        CREATE INDEX idx_documents_source_type ON documents(source_type);
        CREATE INDEX idx_structured_table ON structured_records(table_name);
        """
    )


def insert_document(conn: sqlite3.Connection, doc: CandidateDoc, cleaned: CleanResult) -> int:
    content_hash = stable_hash(normalize_for_hash(cleaned.text), length=40)
    cur = conn.execute(
        """
        INSERT INTO documents(
            doc_key, source_dataset, source_path, source_type, category, title, url, host,
            date, language, raw_chars, cleaned_chars, quality_score, content_hash, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc.doc_key,
            doc.source_dataset,
            doc.source_path,
            doc.source_type,
            doc.category,
            doc.title,
            doc.url,
            doc.host,
            doc.date,
            "",
            len(doc.text),
            len(cleaned.text),
            doc.quality_score,
            content_hash,
            json.dumps(doc.metadata, ensure_ascii=False, sort_keys=True),
        ),
    )
    return int(cur.lastrowid)


def insert_chunk(
    conn: sqlite3.Connection,
    *,
    doc_id: int,
    chunk_index: int,
    text: str,
    doc: CandidateDoc,
    metadata: Optional[Mapping[str, Any]] = None,
) -> int:
    metadata_dict = dict(metadata or {})
    metadata_dict.update(
        {
            "source_path": doc.source_path,
            "source_type": doc.source_type,
            "source_dataset": doc.source_dataset,
            "url": doc.url,
            "host": doc.host,
            "date": doc.date,
        }
    )
    chunk_hash = stable_hash(normalize_for_hash(text), length=40)
    chunk_uid = stable_hash(f"{doc.doc_key}:{chunk_index}:{chunk_hash}", length=32)
    cur = conn.execute(
        """
        INSERT INTO chunks(
            chunk_uid, doc_id, chunk_index, text, title, source_dataset, source_path,
            source_type, category, url, host, date, quality_score, content_hash, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_uid,
            doc_id,
            chunk_index,
            text,
            doc.title,
            doc.source_dataset,
            doc.source_path,
            doc.source_type,
            doc.category,
            doc.url,
            doc.host,
            doc.date,
            doc.quality_score,
            chunk_hash,
            json.dumps(metadata_dict, ensure_ascii=False, sort_keys=True),
        ),
    )
    chunk_id = int(cur.lastrowid)
    search_text = make_search_text(doc.title, text, metadata_dict)
    conn.execute(
        "INSERT INTO chunks_fts(rowid, search_text, chunk_uid) VALUES (?, ?, ?)",
        (chunk_id, search_text, chunk_uid),
    )
    return chunk_id


def build_text_documents(
    conn: sqlite3.Connection,
    candidates: Sequence[CandidateDoc],
    dynamic_blacklist: set[str],
    stats: BuildStats,
    *,
    min_doc_chars: int = 120,
) -> set[str]:
    seen_chunk_hashes: set[str] = set()
    for raw_doc in candidates:
        stats.docs_seen[raw_doc.source_type] += 1
        cleaned = clean_text(raw_doc.text, dynamic_blacklist=dynamic_blacklist)
        if looks_garbled(cleaned.text):
            stats.docs_skipped[f"{raw_doc.source_type}:garbled"] += 1
            continue
        if document_should_skip_as_academic_pdf(raw_doc, cleaned.text):
            stats.docs_skipped[f"{raw_doc.source_type}:academic_pdf"] += 1
            continue
        if len(cleaned.text) < min_doc_chars:
            stats.short_docs += 1
            stats.docs_skipped[f"{raw_doc.source_type}:short"] += 1
            continue
        title = raw_doc.title or extract_title(cleaned.text, Path(raw_doc.source_path).stem)
        url = raw_doc.url or extract_url(cleaned.text)
        host = raw_doc.host or host_from_url_or_path(url, raw_doc.source_path)
        date = raw_doc.date or extract_date(cleaned.text, raw_doc.source_path)
        category = raw_doc.category or infer_category_from_path(raw_doc.source_type, Path(raw_doc.source_path))
        quality = adjusted_quality(raw_doc.quality_score, cleaned.text, raw_doc.source_type, category, host, raw_doc.source_path)
        if quality < 0.30:
            stats.low_quality_docs += 1
            stats.docs_skipped[f"{raw_doc.source_type}:low_quality"] += 1
            continue

        doc = CandidateDoc(
            source_dataset=raw_doc.source_dataset,
            source_path=raw_doc.source_path,
            source_type=raw_doc.source_type,
            category=category,
            title=title,
            url=url,
            host=host,
            date=date,
            text=raw_doc.text,
            quality_score=quality,
            metadata={
                **raw_doc.metadata,
                "raw_lines": cleaned.raw_lines,
                "kept_lines": cleaned.kept_lines,
                "removed_lines": cleaned.removed_lines,
                "removed_blacklist": cleaned.removed_blacklist,
            },
        )
        try:
            doc_id = insert_document(conn, doc, cleaned)
        except sqlite3.IntegrityError:
            stats.docs_skipped[f"{doc.source_type}:duplicate_doc"] += 1
            continue

        chunk_count = 0
        for chunk_index, chunk in enumerate(split_text_to_chunks(cleaned.text)):
            chunk_hash = stable_hash(normalize_for_hash(chunk), length=40)
            if chunk_hash in seen_chunk_hashes:
                stats.duplicate_chunks += 1
                continue
            seen_chunk_hashes.add(chunk_hash)
            insert_chunk(conn, doc_id=doc_id, chunk_index=chunk_index, text=chunk, doc=doc)
            chunk_count += 1
        if chunk_count:
            stats.docs_kept[doc.source_type] += 1
            stats.chunks_inserted[doc.source_type] += chunk_count
        else:
            stats.docs_skipped[f"{doc.source_type}:no_chunks"] += 1
    return seen_chunk_hashes


def row_confidence(row: Mapping[str, Any]) -> float:
    try:
        return float(row.get("confidence") if row.get("confidence") is not None else 1.0)
    except (TypeError, ValueError):
        return 1.0


def meaningful_structured_row(table: str, row: Mapping[str, Any]) -> bool:
    threshold = STRUCTURED_CONFIDENCE_THRESHOLDS.get(table, 0.0)
    if row_confidence(row) < threshold:
        return False
    if table == "courses":
        return any(clean_scalar(row.get(key)) for key in ("course_code", "course_name", "course_name_en", "instructor"))
    if table == "program_requirements":
        title = clean_scalar(row.get("program_name")) or clean_scalar(row.get("title"))
        source_url = source_url_from_row(row)
        requirement = clean_scalar(row.get("requirement_text"))
        probe = f"{title}\n{source_url}\n{requirement}".lower()
        bad_title_terms = ("毕业生故事", "人物专访", "学生活动", "新闻", "讲座", "论坛", "风采")
        if any(term in title for term in bad_title_terms):
            return False
        good_terms = (
            "培养方案",
            "学位申请",
            "答辩",
            "课程",
            "学分",
            "毕业要求",
            "degree",
            "program",
            "academics",
            "graduate",
            "undergraduate",
            "requirement",
            "defence",
            "defense",
            "pyfa",
        )
        if not any(term in probe for term in good_terms):
            return False
    if table == "program_sources":
        title = clean_scalar(row.get("program_name")) or clean_scalar(row.get("title"))
        evidence = clean_scalar(row.get("evidence"))
        probe = f"{title}\n{source_url_from_row(row)}\n{evidence}".lower()
        bad_title_terms = ("招聘", "实习", "宣讲", "讲座", "新闻", "人物专访", "毕业生故事")
        if any(term in title for term in bad_title_terms):
            return False
        good_terms = (
            "培养方案",
            "课程",
            "学位",
            "本科",
            "硕士",
            "博士",
            "degree",
            "program",
            "academics",
            "graduate",
            "undergraduate",
            "requirement",
            "pyfa",
        )
        if not any(term in probe for term in good_terms):
            return False
    if table == "facts":
        subject = clean_scalar(row.get("subject"))
        obj = clean_scalar(row.get("object_value")) or clean_scalar(row.get("object_json"))
        if len(subject) < 2 or len(obj) < 2:
            return False
        if len(clean_scalar(row.get("evidence"))) > 2500 and row_confidence(row) < 0.90:
            return False
    if table == "leadership_roles":
        name = clean_scalar(row.get("name"))
        role = clean_scalar(row.get("role"))
        bad_names = {"vice dean", "dean", "berkeley", "previous academic and professional roles", "sist committees"}
        if not name or name.lower() in bad_names or not role:
            return False
    return True


def structured_title(table: str, row: Mapping[str, Any]) -> str:
    for key in ("course_code", "code", "course_name", "name_cn", "name", "professor", "course_name_en", "title", "subject", "program_name"):
        value = clean_scalar(row.get(key))
        if value and not is_placeholder_value(value):
            return value[:180]
    for key in ("name_en", "source_url", "profile_url", "homepage"):
        value = clean_scalar(row.get(key))
        if value and not is_placeholder_value(value):
            if key in {"source_url", "profile_url", "homepage"}:
                host = host_from_url_or_path(value, "")
                return host or table
            return value[:180]
    return table


def source_url_from_row(row: Mapping[str, Any]) -> str:
    for key in ("source_url", "profile_url", "homepage", "detail_url", "url", "list_page_url"):
        value = clean_scalar(row.get(key))
        if value:
            return value
    return ""


def markdown_for_structured_row(table: str, row: Mapping[str, Any]) -> str:
    title = structured_title(table, row)
    lines = [f"# {title}", "", f"来源表: {table}"]
    preferred_order = [
        "school",
        "program_name",
        "cohort",
        "degree",
        "course_code",
        "code",
        "course_name",
        "name_cn",
        "course_name_en",
        "credits",
        "hours",
        "term",
        "semester",
        "instructor",
        "teacher",
        "teachers",
        "name",
        "name_en",
        "title",
        "role",
        "faculty_category",
        "research_area",
        "email",
        "phone",
        "office",
        "homepage",
        "profile_url",
        "event_type",
        "event_date",
        "published_at",
        "requirement_type",
        "requirement_text",
        "min_credits",
        "source_url",
    ]
    emitted = set()
    for key in preferred_order:
        value = clean_scalar(row.get(key))
        value = clean_structured_markdown_value(value, key)
        if key in TEACHER_FIELD_KEYS:
            value = clean_teacher_value(value)
        if value:
            if key in {"name", "name_en", "title"} and is_placeholder_value(value):
                continue
            if key in {"name_en", "course_name_en"} and re.fullmatch(r"\d+(?:\.\d+)?", value):
                continue
            label = FIELD_LABELS.get(key, key)
            lines.append(f"- {label}: {value}")
            emitted.add(key)
    for key, value_obj in row.items():
        if key in emitted or key in {"id", "source_document_id", "object_json", "raw_json"}:
            continue
        if table == "courses" and key == "evidence":
            continue
        value = clean_scalar(value_obj)
        value = clean_structured_markdown_value(value, key)
        if key in TEACHER_FIELD_KEYS:
            value = clean_teacher_value(value)
        if not value:
            continue
        if key in {"name", "name_en", "title"} and is_placeholder_value(value):
            continue
        if key == "teaching_section" and value.count("](") >= 3:
            continue
        if value.count("](") >= 5 and not DATE_RE.search(value):
            continue
        if key in {"name_en", "course_name_en"} and re.fullmatch(r"\d+(?:\.\d+)?", value):
            continue
        if key == "evidence" and len(value) > 700:
            value = value[:700].rstrip() + "..."
        elif len(value) > 1400:
            value = value[:1400].rstrip() + "..."
        label = FIELD_LABELS.get(key, key)
        lines.append(f"- {label}: {value}")
    return "\n".join(lines).strip()


def load_sist_documents_map(data_root: Path) -> Dict[int, Dict[str, Any]]:
    docs: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(data_root / "sist/jsonl/documents.jsonl"):
        try:
            docs[int(row["id"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return docs


def build_sist_structured(
    conn: sqlite3.Connection,
    data_root: Path,
    stats: BuildStats,
    seen_chunk_hashes: set[str],
) -> None:
    docs_by_id = load_sist_documents_map(data_root)
    for table, base_quality in STRUCTURED_TABLES.items():
        path = data_root / f"sist/jsonl/{table}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if not meaningful_structured_row(table, row):
                stats.docs_skipped[f"structured_{table}:filtered"] += 1
                continue
            doc_row = docs_by_id.get(int(row.get("source_document_id") or 0), {})
            source_url = source_url_from_row(row) or clean_scalar(doc_row.get("url"))
            host = host_from_url_or_path(source_url, path.as_posix())
            date = extract_date(
                "",
                clean_scalar(row.get("valid_from")),
                clean_scalar(row.get("published_at")),
                clean_scalar(row.get("observed_at")),
                clean_scalar(doc_row.get("source_published_at")),
            )
            title = structured_title(table, row)
            text = markdown_for_structured_row(table, row)
            if len(text) < 40:
                continue
            if looks_garbled(text):
                stats.docs_skipped[f"structured_{table}:garbled"] += 1
                continue
            doc = CandidateDoc(
                source_dataset="course_sist",
                source_path=f"data/sist/jsonl/{table}.jsonl#{row.get('id', '')}",
                source_type="structured_table",
                category=table,
                title=title,
                url=source_url,
                host=host,
                date=date,
                text=text,
                quality_score=max(0.05, min(base_quality * row_confidence(row), 1.0)),
                metadata={"table": table, "record_id": row.get("id"), "source_document_id": row.get("source_document_id")},
            )
            cleaned = CleanResult(text=text, raw_lines=text.count("\n") + 1, kept_lines=text.count("\n") + 1, removed_lines=0, removed_blacklist=0)
            try:
                doc_id = insert_document(conn, doc, cleaned)
            except sqlite3.IntegrityError:
                continue
            chunk_hash = stable_hash(normalize_for_hash(text), length=40)
            if chunk_hash in seen_chunk_hashes:
                stats.duplicate_chunks += 1
                continue
            seen_chunk_hashes.add(chunk_hash)
            chunk_id = insert_chunk(conn, doc_id=doc_id, chunk_index=0, text=text, doc=doc, metadata={"table": table})
            conn.execute(
                """
                INSERT INTO structured_records(
                    chunk_id, source_dataset, source_path, table_name, record_id,
                    title, source_url, quality_score, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    doc.source_dataset,
                    doc.source_path,
                    table,
                    clean_scalar(row.get("id")),
                    title,
                    source_url,
                    doc.quality_score,
                    json.dumps(row, ensure_ascii=False, sort_keys=True),
                ),
            )
            stats.structured_records[table] += 1
            stats.chunks_inserted["structured_table"] += 1
            stats.docs_kept["structured_table"] += 1


def flatten_self_json_records(name: str, data: Any) -> Iterator[Dict[str, Any]]:
    if name == "course_teacher_map.json" and isinstance(data, dict):
        for code, teachers in data.items():
            yield {"code": code, "teachers": teachers, "source": name}
        return
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield dict(item)
        return
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        record = dict(item)
                        record.setdefault("group", key)
                        yield record
            elif isinstance(value, dict):
                record = dict(value)
                record.setdefault("group", key)
                yield record


def parse_float(value: Any) -> Optional[float]:
    text = clean_scalar(value)
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def meaningful_self_json_row(table: str, row: Mapping[str, Any]) -> bool:
    course_like_tables = {"courses_unified", "courses_clean", "sist_courses", "course_teacher_map"}
    if table in course_like_tables:
        code = clean_scalar(row.get("course_code")) or clean_scalar(row.get("code"))
        name = clean_scalar(row.get("course_name")) or clean_scalar(row.get("name_cn")) or clean_scalar(row.get("name_en"))
        teachers = clean_scalar(row.get("instructor")) or clean_scalar(row.get("teacher")) or clean_scalar(row.get("teachers"))
        teachers = clean_teacher_value(teachers)
        if not any((code, name, teachers)):
            return False
        credits = parse_float(row.get("credits"))
        if credits is not None and credits > 20:
            return False
    return True


def build_self_structured(
    conn: sqlite3.Connection,
    data_root: Path,
    stats: BuildStats,
    seen_chunk_hashes: set[str],
) -> None:
    root = data_root / "shanghaitech_data/data"
    if not root.exists():
        return
    for filename, base_quality in SELECTED_SELF_JSON.items():
        path = root / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        table = path.stem
        for index, row in enumerate(flatten_self_json_records(filename, data)):
            if not isinstance(row, dict):
                continue
            if not meaningful_self_json_row(table, row):
                stats.docs_skipped[f"structured_json_{table}:filtered"] += 1
                continue
            source_url = source_url_from_row(row)
            host = host_from_url_or_path(source_url, path.as_posix())
            date = extract_date("", clean_scalar(row.get("semester")), clean_scalar(row.get("source")), clean_scalar(row.get("date")))
            title = structured_title(table, row)
            text = markdown_for_structured_row(table, row)
            if len(text) < 35:
                continue
            if looks_garbled(text):
                stats.docs_skipped[f"structured_json_{table}:garbled"] += 1
                continue
            doc = CandidateDoc(
                source_dataset="self_crawl_structured",
                source_path=f"{path.relative_to(PROJECT_ROOT).as_posix()}#{index}",
                source_type="structured_json",
                category=table,
                title=title,
                url=source_url,
                host=host,
                date=date,
                text=text,
                quality_score=base_quality,
                metadata={"json_file": filename, "record_index": index},
            )
            cleaned = CleanResult(text=text, raw_lines=text.count("\n") + 1, kept_lines=text.count("\n") + 1, removed_lines=0, removed_blacklist=0)
            try:
                doc_id = insert_document(conn, doc, cleaned)
            except sqlite3.IntegrityError:
                continue
            chunk_hash = stable_hash(normalize_for_hash(text), length=40)
            if chunk_hash in seen_chunk_hashes:
                stats.duplicate_chunks += 1
                continue
            seen_chunk_hashes.add(chunk_hash)
            chunk_id = insert_chunk(conn, doc_id=doc_id, chunk_index=0, text=text, doc=doc, metadata={"json_file": filename})
            conn.execute(
                """
                INSERT INTO structured_records(
                    chunk_id, source_dataset, source_path, table_name, record_id,
                    title, source_url, quality_score, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    doc.source_dataset,
                    doc.source_path,
                    table,
                    str(index),
                    title,
                    source_url,
                    doc.quality_score,
                    json.dumps(row, ensure_ascii=False, sort_keys=True),
                ),
            )
            stats.structured_records[table] += 1
            stats.chunks_inserted["structured_json"] += 1
            stats.docs_kept["structured_json"] += 1


def write_metadata(conn: sqlite3.Connection, stats: BuildStats, output_path: Path, args: argparse.Namespace) -> None:
    summary = {
        "built_at": utc_now(),
        "output_path": str(output_path),
        "args": vars(args),
        "docs_seen": dict(stats.docs_seen),
        "docs_kept": dict(stats.docs_kept),
        "docs_skipped": dict(stats.docs_skipped),
        "chunks_inserted": dict(stats.chunks_inserted),
        "structured_records": dict(stats.structured_records),
        "duplicate_chunks": stats.duplicate_chunks,
        "short_docs": stats.short_docs,
        "low_quality_docs": stats.low_quality_docs,
        "dynamic_blacklist_count": stats.dynamic_blacklist_count,
        "top_blacklist_lines": stats.top_blacklist_lines,
    }
    conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("build_summary", json.dumps(summary, ensure_ascii=False, sort_keys=True)))
    conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("schema_version", "rag_unified_v1"))


def scalar_query(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] if row else 0)


def query_rows(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> List[Tuple[Any, ...]]:
    return list(conn.execute(sql, params).fetchall())


def write_report(conn: sqlite3.Connection, stats: BuildStats, report_path: Path, output_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Unified RAG Database Build Report")
    lines.append("")
    lines.append(f"- Built at UTC: `{utc_now()}`")
    lines.append(f"- Database: `{output_path}`")
    lines.append(f"- Documents: **{scalar_query(conn, 'select count(*) from documents')}**")
    lines.append(f"- Chunks: **{scalar_query(conn, 'select count(*) from chunks')}**")
    lines.append(f"- Structured records: **{scalar_query(conn, 'select count(*) from structured_records')}**")
    lines.append(f"- Duplicate chunks skipped: **{stats.duplicate_chunks}**")
    lines.append(f"- Dynamic blacklist lines: **{stats.dynamic_blacklist_count}**")
    lines.append("")

    def add_table(title: str, rows: Sequence[Tuple[Any, ...]], headers: Sequence[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(item).replace("\n", " ") for item in row) + " |")
        lines.append("")

    add_table(
        "Chunks by Source Type",
        query_rows(conn, "select source_type, count(*), round(avg(quality_score), 3) from chunks group by source_type order by count(*) desc"),
        ("source_type", "chunks", "avg_quality"),
    )
    add_table(
        "Top Hosts",
        query_rows(conn, "select coalesce(nullif(host,''), '(none)') as host, count(*) from chunks group by host order by count(*) desc limit 30"),
        ("host", "chunks"),
    )
    add_table(
        "Top Categories",
        query_rows(conn, "select coalesce(nullif(category,''), '(none)') as category, count(*) from chunks group by category order by count(*) desc limit 30"),
        ("category", "chunks"),
    )
    add_table(
        "Structured Record Tables",
        query_rows(conn, "select table_name, count(*) from structured_records group by table_name order by count(*) desc"),
        ("table", "records"),
    )

    lines.append("## Top Dynamic Blacklist Lines")
    lines.append("")
    lines.append("| line | document_frequency |")
    lines.append("| --- | ---: |")
    for line, count in stats.top_blacklist_lines[:30]:
        lines.append(f"| `{line.replace('`', '')}` | {count} |")
    lines.append("")

    lines.append("## Build Counters")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "docs_seen": dict(stats.docs_seen),
                "docs_kept": dict(stats.docs_kept),
                "docs_skipped": dict(stats.docs_skipped),
                "chunks_inserted": dict(stats.chunks_inserted),
                "structured_records": dict(stats.structured_records),
                "short_docs": stats.short_docs,
                "low_quality_docs": stats.low_quality_docs,
                "duplicate_chunks": stats.duplicate_chunks,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    lines.append("```")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_sample_file(conn: sqlite3.Connection, sample_path: Path, per_source_type: int = 4) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# RAG Database Review Samples")
    lines.append("")
    lines.append("These are deterministic samples for manual quality review.")
    lines.append("")
    source_types = [row[0] for row in conn.execute("select distinct source_type from chunks order by source_type")]
    for source_type in source_types:
        rows = conn.execute(
            """
            select id, title, source_path, url, category, quality_score, text
            from chunks
            where source_type = ?
            order by ((id * 1103515245 + 12345) & 2147483647)
            limit ?
            """,
            (source_type, per_source_type),
        ).fetchall()
        lines.append(f"## {source_type}")
        lines.append("")
        for row in rows:
            chunk_id, title, source_path, url, category, quality, text = row
            snippet = str(text).replace("\n", " ")
            if len(snippet) > 900:
                snippet = snippet[:900].rstrip() + "..."
            lines.append(f"### chunk {chunk_id} | {title or '(no title)'}")
            lines.append("")
            lines.append(f"- source_path: `{source_path}`")
            lines.append(f"- url: `{url or ''}`")
            lines.append(f"- category: `{category or ''}`")
            lines.append(f"- quality_score: `{quality:.3f}`")
            lines.append("")
            lines.append("> " + snippet)
            lines.append("")
    sample_path.write_text("\n".join(lines), encoding="utf-8")


def build_database(args: argparse.Namespace) -> None:
    data_root = (PROJECT_ROOT / args.data_root).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    report_path = (PROJECT_ROOT / args.report).resolve()
    sample_path = (PROJECT_ROOT / args.samples).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(output_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()

    print("Collecting text candidates...", flush=True)
    text_candidates = list(iter_text_candidates(data_root))
    print(f"Text candidates: {len(text_candidates)}", flush=True)

    print("Building dynamic line blacklist...", flush=True)
    dynamic_blacklist, top_blacklist_lines = build_dynamic_line_blacklist(text_candidates)
    stats = BuildStats()
    stats.dynamic_blacklist_count = len(dynamic_blacklist)
    stats.top_blacklist_lines = top_blacklist_lines

    conn = sqlite3.connect(str(output_path))
    try:
        create_schema(conn)
        print("Building text documents/chunks...", flush=True)
        seen_chunk_hashes = build_text_documents(
            conn,
            text_candidates,
            dynamic_blacklist,
            stats,
            min_doc_chars=args.min_doc_chars,
        )
        conn.commit()

        print("Building course-provided structured records...", flush=True)
        build_sist_structured(conn, data_root, stats, seen_chunk_hashes)
        conn.commit()

        print("Building self-crawled structured records...", flush=True)
        build_self_structured(conn, data_root, stats, seen_chunk_hashes)
        conn.commit()

        write_metadata(conn, stats, output_path, args)
        conn.commit()
        write_report(conn, stats, report_path, output_path)
        write_sample_file(conn, sample_path, per_source_type=args.samples_per_source_type)
        conn.commit()
    finally:
        conn.close()

    print(f"Database written: {output_path}")
    print(f"Report written: {report_path}")
    print(f"Samples written: {sample_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cleaned unified SQLite RAG database from project data.")
    parser.add_argument("--data-root", default="data", help="Data root relative to project root.")
    parser.add_argument("--output", default="data/rag/knowledge.sqlite", help="Output SQLite database.")
    parser.add_argument("--report", default="data/rag/build_report.md", help="Build report markdown.")
    parser.add_argument("--samples", default="data/rag/review_samples.md", help="Deterministic sample markdown for review.")
    parser.add_argument("--min-doc-chars", type=int, default=120)
    parser.add_argument("--samples-per-source-type", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    build_database(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
