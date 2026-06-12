from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from retrieval.keyword_search import make_snippet, tokenize

from .text_index import make_match_query


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = Path("data/rag/knowledge.sqlite")
COURSE_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,}\d{2,4}[A-Z]?)(?![A-Za-z0-9])", flags=re.IGNORECASE)
PROGRAM_CODE_RE = re.compile(r"(?<![A-Za-z0-9])(CS|EE|IE|CSEE|SI)(?![A-Za-z0-9])", flags=re.IGNORECASE)
YEAR_RE = re.compile(r"(20\d{2})")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
CJK_PHRASE_RE = re.compile(r"[\u4e00-\u9fff]{2,8}")
ASCII_ENTITY_RE = re.compile(r"(?<![A-Za-z0-9_+-])([A-Za-z][A-Za-z0-9_+-]{3,})(?![A-Za-z0-9_+-])")

ASCII_ENTITY_STOPWORDS = {
    "about",
    "assistant",
    "campus",
    "course",
    "courses",
    "faculty",
    "homepage",
    "information",
    "instructor",
    "overview",
    "profile",
    "research",
    "school",
    "shanghaitech",
    "sist",
    "student",
    "students",
    "teacher",
    "university",
}


def resolve_db_path(db_path: Path = DEFAULT_DB_PATH) -> Path:
    path = Path(db_path)
    if path.is_absolute():
        return path

    candidates = [path, PROJECT_ROOT / path]
    if path == DEFAULT_DB_PATH:
        candidates.extend(
            [
                PROJECT_ROOT.parent / "rag" / "knowledge.sqlite",
                Path.cwd() / "rag" / "knowledge.sqlite",
                Path.cwd().parent / "rag" / "knowledge.sqlite",
            ]
        )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return path


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
    "course_schedule",
    "course_teacher_map",
    "prof_courses_full",
    "faculty_members",
    "sist_faculty",
    "faculty",
    "faculty_merged",
    "all_faculty",
    "professors",
    "professors_clean",
    "professors_enriched",
    "contacts",
    "program_requirements",
    "program_sources",
    "program",
    "events",
    "news",
    "leadership",
    "leadership_roles",
    "university_overview",
    "university_contact",
    "sist_overview",
    "sist_degree_programs",
    "sist_news_events",
    "sist_research",
}

COURSE_ROUTE_CATEGORIES = {
    "courses",
    "sist_courses",
    "courses_clean",
    "courses_unified",
    "course_schedule",
    "course_teacher_map",
    "prof_courses_full",
}

PROGRAM_ROUTE_CATEGORIES = {
    "program",
    "program_requirements",
    "program_sources",
    "sist_degree_programs",
}

FACULTY_ROUTE_CATEGORIES = {
    "faculty",
    "faculty_members",
    "sist_faculty",
    "faculty_merged",
    "all_faculty",
    "professors",
    "professors_clean",
    "professors_enriched",
    "contacts",
    "leadership",
    "leadership_roles",
}

POLICY_ROUTE_CATEGORIES = {
    "events",
    "news",
    "纯文本资料",
    "教授课程数据",
    "结构化数据",
}

STRUCTURED_SOURCE_TYPES = {
    "structured_table",
    "structured_json",
    "rag_json",
    "training",
    "sist_text",
}

INTRO_QUERY_TERMS = ("介绍", "简介", "是什么", "了解", "概况")
INTRO_QUERY_TERMS_LOWER = ("overview", "about", "introduction", "profile")
INTRO_EVIDENCE_MARKERS = (
    "创立于",
    "成立于",
    "创办",
    "专业技术型社团",
    "明星社团",
    "科技社团",
    "科创社团",
    "学生组织",
    "核心团队",
    "开源共享",
    "协作与开发",
    "创意与创造",
    "综合性科创开发组织",
    "致力于",
    "research interests",
)
FORMAL_INTRO_MARKERS = (
    "创立于",
    "成立于",
    "创办",
    "专业技术型社团",
    "明星社团",
    "科创社团",
    "核心团队",
    "开源共享",
    "协作与开发",
    "综合性科创开发组织",
)
LIST_PAGE_TITLES = ("学生社团", "大道社团", "本科生招生", "研究生招生介绍一览")
EVENT_TITLE_MARKERS = (" day", "pi day", "派对", "游园会")
SCHOLARSHIP_QUERY_TERMS = ("奖学金", "奖助", "助学金", "奖助学金")
SCHOLARSHIP_QUERY_TERMS_LOWER = ("scholarship", "fellowship", "financial aid")
SCHOLARSHIP_POLICY_MARKERS = (
    "研究生奖助体系",
    "研究生国家奖学金",
    "研究生学业奖学金",
    "研究生国家助学金",
    "校研究生等级奖学金",
    "助研",
    "助教",
    "助管",
    "三助",
    "收费和奖助学金管理办法",
    "学费和奖助",
    "奖助标准",
    "奖助生均",
    "国家奖学金管理办法",
    "本科生国家奖学金管理办法",
    "上海市奖学金",
)
SCHOLARSHIP_PERSON_MARKERS = ("获得者", "风采", "人物专访", "青春榜样", "毕业生故事", "荣获")
ADVISOR_QUERY_TERMS = (
    "导师",
    "博导",
    "硕导",
    "指导老师",
    "教授",
    "教师",
    "老师",
    "课题组",
    "实验室",
    "advisor",
    "supervisor",
    "faculty",
    "mentor",
    "pi",
)
ADVISOR_RECOMMENDATION_TERMS = (
    "推荐",
    "有哪些",
    "哪些",
    "适合",
    "可以联系",
    "想做",
    "想找",
    "方向",
    "领域",
    "研究方向",
    "研究兴趣",
    "research area",
    "research interests",
)
FACULTY_PROFILE_MARKERS = (
    "博导",
    "博士生导师",
    "硕士生导师",
    "助理教授",
    "副教授",
    "教授、研究员",
    "研究员、博导",
    "个人主页:",
    "个人主页：",
    "博士毕业院校",
    "办公室:",
    "办公室：",
    "邮箱:",
    "邮箱：",
    "research interests",
    "associate professor",
    "assistant professor",
    "professor",
)
NON_FACULTY_MEMBER_MARKERS = (
    "身份:硕士",
    "身份: 硕士",
    "身份:博士生",
    "身份: 博士生",
    "身份:研究助理",
    "年级:研",
    "年级:博",
    "master student",
    "doctoral student",
    "ph.d. student",
    "current students",
)
RESEARCH_TOPIC_STOPWORDS = {
    "我想",
    "想做",
    "做机",
    "方向",
    "哪些",
    "导师",
    "可以",
    "推荐",
    "有哪",
    "老师",
    "教授",
    "教师",
    "研究",
    "领域",
    "课题",
    "个人",
    "主页",
    "邮箱",
    "办公室",
    "faculty",
    "advisor",
    "supervisor",
    "mentor",
    "research",
    "interests",
    "area",
    "areas",
}
RESEARCH_TOPIC_EXPANSIONS = {
    "机器人": [
        "robotics",
        "robotic",
        "robot",
        "自动化与机器人",
        "STAR",
        "具身智能",
        "移动机器人",
        "机器人导航",
        "机器人操作",
        "SLAM",
        "无人机",
        "遥操作",
        "触觉交互",
    ],
    "具身智能": ["embodied intelligence", "embodied ai", "robotics", "机器人"],
    "自动化与机器人": ["STAR", "robotics", "机器人", "具身智能"],
}

QUERY_EXPANSIONS = {
    "上科大": ["上海科技大学", "ShanghaiTech University"],
    "徐旭明": ["何旭明", "Xuming He", "hexm", "PLUS Lab"],
    "何旭明": ["Xuming He", "hexm", "PLUS Lab"],
    "类型的高校": ["全日制普通高等学校", "研究型", "创新型", "小规模", "高水平", "国际化"],
    "什么类型的高校": ["全日制普通高等学校", "研究型", "创新型", "小规模", "高水平", "国际化"],
    "日常管理": ["上海市人民政府负责日常管理", "负责日常管理", "上海市人民政府"],
    "第四单元": ["考试科目", "业务课二", "专业课"],
    "先修": ["先修课程", "Prerequisites"],
    "任课": ["任课教师", "Instructor"],
    "老师": ["任课教师", "Instructor"],
    "导师": ["advisor", "supervisor", "professor", "faculty", "研究方向", "博士生导师", "博导"],
    "学分": ["Credit", "credits"],
    "研究中心": ["Research Center", "研究中心"],
    "研究方向": ["research interests", "研究兴趣", "研究方向"],
    "机器人": RESEARCH_TOPIC_EXPANSIONS["机器人"],
    "具身智能": RESEARCH_TOPIC_EXPANSIONS["具身智能"],
    "自动化与机器人": RESEARCH_TOPIC_EXPANSIONS["自动化与机器人"],
    "招生方式": ["招生方式", "申请考核", "硕博连读", "直接攻博"],
    "学术型博士": ["学术型博士项目", "上海科技大学常任教授或特聘教授作为指导老师", "全日制非定向"],
    "专业型博士": ["专业型博士项目", "全日制非定向、非全日制定向", "工程实践经验", "行业专家", "研究生导师"],
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


def _query_ascii_entities(query: str) -> List[str]:
    entities: List[str] = []
    seen = set()
    for match in ASCII_ENTITY_RE.finditer(query):
        value = match.group(0)
        key = value.lower().strip("_+-")
        if key in ASCII_ENTITY_STOPWORDS:
            continue
        if COURSE_CODE_RE.fullmatch(value) or PROGRAM_CODE_RE.fullmatch(value):
            continue
        if key in seen:
            continue
        seen.add(key)
        entities.append(value)
    return entities[:5]


def _compact_ascii(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _has_intro_intent(query: str) -> bool:
    lowered = query.lower()
    return any(term in query for term in INTRO_QUERY_TERMS) or any(
        term in lowered for term in INTRO_QUERY_TERMS_LOWER
    )


def _has_scholarship_intent(query: str) -> bool:
    lowered = query.lower()
    return any(term in query for term in SCHOLARSHIP_QUERY_TERMS) or any(
        term in lowered for term in SCHOLARSHIP_QUERY_TERMS_LOWER
    )


def _has_scholarship_overview_intent(query: str) -> bool:
    if not _has_scholarship_intent(query):
        return False
    lowered = query.lower()
    if any(marker in query for marker in ("获得者", "获奖者", "谁", "名单", "风采", "人物")):
        return False
    return _has_intro_intent(query) or any(
        term in query or term in lowered
        for term in (
            "上科大",
            "上海科技大学",
            "体系",
            "标准",
            "政策",
            "办法",
            "管理办法",
            "申请",
            "评选",
            "包括",
            "有哪些",
            "多少",
            "学费",
            "financial aid",
        )
    )


def _marker_count(text: str, markers: Sequence[str]) -> int:
    text_lower = text.lower()
    return sum(1 for marker in markers if marker in text or marker.lower() in text_lower)


def _has_faculty_profile_marker(text: str) -> bool:
    text_lower = text.lower()
    return any(marker in text or marker.lower() in text_lower for marker in FACULTY_PROFILE_MARKERS)


def _has_non_faculty_member_noise(text: str) -> bool:
    text_lower = text.lower()
    return any(marker in text or marker.lower() in text_lower for marker in NON_FACULTY_MEMBER_MARKERS)


def _has_advisor_recommendation_intent(query: str) -> bool:
    lowered = query.lower()
    has_advisor = any(term in query or term in lowered for term in ADVISOR_QUERY_TERMS)
    if not has_advisor:
        return False
    return any(term in query or term in lowered for term in ADVISOR_RECOMMENDATION_TERMS)


def _clean_topic_fragment(value: str) -> str:
    value = value.strip()
    for splitter in ("导师", "老师", "教授", "研究方向", "研究兴趣"):
        if splitter in value:
            value = value.split(splitter, 1)[0]
    value = re.sub(r"^(我)?(想|要)?(做|找|研究|关注|有关|关于|适合|联系|推荐)+", "", value)
    value = re.sub(r"的?(?:SIST|信息学院|上海科技大学|上科大)?$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[，。？?、；;：:\s]+$", "", value)
    return value


def _research_topic_terms(query: str) -> List[str]:
    terms: List[str] = []
    seen = set()

    def add(value: str) -> None:
        value = _clean_topic_fragment(value)
        if not value:
            return
        key = value.lower()
        if key in seen or key in RESEARCH_TOPIC_STOPWORDS:
            return
        if value in {"如果", "是否", "匹配", "合适", "适合", "可以", "老师", "教师", "导师", "教授"}:
            return
        if any(char in value for char in "我想做究方向导师推荐可以哪些") and value not in RESEARCH_TOPIC_EXPANSIONS:
            return
        if any(existing and value in existing for existing in terms):
            return
        for existing in list(terms):
            if existing in value:
                terms.remove(existing)
                seen.discard(existing.lower())
        if len(value) < 2 or len(value) > 24:
            return
        seen.add(key)
        terms.append(value)

    for match in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9+#.\- ]{2,24}?)(?:方向|领域|课题)", query):
        add(match.group(1))

    for match in re.finditer(
        r"(?:想|要)?(?:找|做|研究|关注|从事)(?:研究)?"
        r"([\u4e00-\u9fffA-Za-z0-9+#.\-、与及和 ]{2,32}?)"
        r"(?:的?(?:SIST|信息学院|上海科技大学|上科大)?(?:老师|教师|导师|教授)|[，。？?])",
        query,
        flags=re.IGNORECASE,
    ):
        add(match.group(1))

    if terms:
        return terms[:8]

    for token in tokenize(query):
        token_lower = token.lower()
        if token_lower in RESEARCH_TOPIC_STOPWORDS:
            continue
        if len(token) < 2:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) <= 4 or token in RESEARCH_TOPIC_EXPANSIONS:
                add(token)
        elif token_lower not in ASCII_ENTITY_STOPWORDS:
            add(token)

    return terms[:8]


def _expanded_research_topic_terms(query: str) -> List[str]:
    terms = _research_topic_terms(query)
    out: List[str] = []
    seen = set()
    for term in terms:
        for value in [term, *RESEARCH_TOPIC_EXPANSIONS.get(term, [])]:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
    return out[:18]


def _contains_course_code(text: str, code: str) -> bool:
    return bool(re.search(rf"(?<![A-Z0-9]){re.escape(code.upper())}(?![A-Z0-9])", text.upper()))


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
    if _query_course_codes(query):
        return True
    lowered = query.lower()
    return any(
        term in query or term in lowered
        for term in (
            "课程",
            "课表",
            "任课",
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


def _has_schedule_intent(query: str) -> bool:
    lowered = query.lower()
    return any(
        term in query or term in lowered
        for term in (
            "课表",
            "开课",
            "上课",
            "时间",
            "地点",
            "教室",
            "第几周",
            "周次",
            "schedule",
            "semester",
            "spring",
            "fall",
            "class schedule",
        )
    )


def _has_policy_intent(query: str) -> bool:
    lowered = query.lower()
    return any(
        term in query or term in lowered
        for term in (
            "发布时间",
            "发布者",
            "发布",
            "细则",
            "办法",
            "规定",
            "制度",
            "通知",
            "公告",
            "policy",
            "published",
            "published_at",
            "rule",
            "regulation",
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


def _structured_route_boost(query: str, row: sqlite3.Row) -> float:
    text = str(row["text"] or "")
    title = str(row["title"] or "")
    title_lower = title.lower()
    text_head_lower = text[:1200].lower()
    path_lower = str(row["source_path"] or "").lower()
    url_lower = str(row["url"] or "").lower()
    source_type = str(row["source_type"] or "")
    category = str(row["category"] or "")
    score = 0.0

    if _has_course_intent(query):
        if category in COURSE_ROUTE_CATEGORIES:
            score += 34.0
        elif source_type in {"structured_table", "structured_json"} and any(
            marker in path_lower or marker in title_lower
            for marker in ("course", "schedule", "课程", "课表")
        ):
            score += 24.0
        elif category in PROGRAM_ROUTE_CATEGORIES:
            score -= 10.0
        if _has_schedule_intent(query):
            if category == "course_schedule" or "schedule" in path_lower or "class sched" in title_lower:
                score += 24.0
            if any(term in text for term in ("教学中心", "信息学院", "周", "春季", "秋季")):
                score += 8.0
        for code in _query_course_codes(query):
            if _contains_course_code(f"{title}\n{text}", code):
                score += 30.0
            else:
                score -= 18.0

    if _has_degree_intent(query):
        if category in PROGRAM_ROUTE_CATEGORIES:
            score += 34.0
        elif category in COURSE_ROUTE_CATEGORIES:
            score -= 10.0
        if any(term in title_lower for term in ("培养方案", "program", "degree")):
            score += 12.0
        if any(marker in path_lower or marker in url_lower for marker in ("program", "degree", "pyfa", "undergraduate")):
            score += 10.0
        if "培养目标" in query:
            if "培养目标" in text[:1600]:
                score += 30.0
            if category == "program_requirements" and any(term in text[:1600] for term in ("要求类型: general", "要求类型: objective")):
                score += 14.0
            if any(term in text[:1200] for term in ("要求类型: credits", "课程代码", "学分")):
                score -= 14.0
        query_lower = query.lower()
        if any(term in query for term in ("本科", "本科生")) or "bachelor" in query_lower:
            if any(term in text[:1200] for term in ("学位层次: bachelor", "本科生", "undergraduate")):
                score += 18.0
            if any(term in text[:1200] for term in ("学位层次: master", "学位层次: doctor", "硕士研究生", "博士研究生", "普博")):
                score -= 34.0
        for code in _query_program_codes(query):
            if code == "EE":
                if any(term in haystack_part for haystack_part in (title, text[:1800], url_lower) for term in ("EE", "电子信息工程", "electrical", "electronic", "in%20ee")):
                    score += 30.0
                if any(term in text[:1800] or term in title for term in ("CS专业", "计算机科学与技术", "computer science")):
                    score -= 38.0
            elif code == "CS":
                if any(term in text[:1800] or term in title_lower or term in url_lower for term in ("CS专业", "计算机科学与技术", "computer science", "in%20cs")):
                    score += 30.0
                if any(term in text[:1800] or term in title for term in ("EE专业", "电子信息工程", "electrical", "electronic")):
                    score -= 38.0

    if _profileish_query(query):
        if category in FACULTY_ROUTE_CATEGORIES:
            score += 34.0
        elif category in {"sist_news_events", "news"}:
            score -= 12.0
        if any(term in text for term in ("邮箱", "办公室", "研究方向", "个人主页", "博士毕业院校")):
            score += 10.0

    if _has_scholarship_intent(query):
        scholarship_overview = _has_scholarship_overview_intent(query)
        scholarship_text = f"{title}\n{text[:2500]}"
        scholarship_lower = scholarship_text.lower()
        has_scholarship_term = any(term in scholarship_text for term in SCHOLARSHIP_QUERY_TERMS) or any(
            term in scholarship_lower for term in SCHOLARSHIP_QUERY_TERMS_LOWER
        )
        policy_count = _marker_count(scholarship_text, SCHOLARSHIP_POLICY_MARKERS)
        if has_scholarship_term:
            score += 20.0
        else:
            score -= 120.0
        if scholarship_overview and policy_count:
            score += 80.0 + min(90.0, policy_count * 14.0)
            if source_type in STRUCTURED_SOURCE_TYPES or source_type in {"web", "text_pages"}:
                score += 12.0
            if any(marker in url_lower for marker in ("openinfo", "yanzhao", "2827", "yjszs", "bkzn")):
                score += 16.0
        elif policy_count:
            score += 8.0
        if scholarship_overview:
            if policy_count:
                score += 70.0
            if any(marker in title or marker in text[:500] for marker in SCHOLARSHIP_PERSON_MARKERS) and not policy_count:
                score -= 60.0
            if "开放课题" in title or "开放课题" in text[:300]:
                score -= 90.0
            if "申请指南" in title and "奖学金" not in title and "奖助" not in title:
                score -= 55.0

    if _has_policy_intent(query):
        if category in POLICY_ROUTE_CATEGORIES:
            score += 18.0
        if source_type == "rag_json":
            score += 8.0
        if any(term in text[:800] for term in ("发布时间:", "发布者:", "published_at", "valid_from", "source_url")):
            score += 16.0
        for phrase in _query_exact_cjk_phrases(query):
            if phrase in title:
                score += 20.0
            elif phrase in text[:900]:
                score += 6.0

    if _has_sist_overview_intent(query):
        if category in {"sist_overview", "university_overview"}:
            score += 30.0
        elif category in COURSE_ROUTE_CATEGORIES | FACULTY_ROUTE_CATEGORIES:
            score -= 12.0

    return score


def _profileish_query(query: str) -> bool:
    lowered = query.lower()
    if _has_advisor_recommendation_intent(query):
        return True
    return any(
        term in query or term in lowered
        for term in (
            "主页",
            "个人主页",
            "教师",
            "教授",
            "导师",
            "博导",
            "硕导",
            "研究方向",
            "研究兴趣",
            "邮箱",
            "办公室",
            "profile",
            "homepage",
            "faculty",
            "advisor",
            "supervisor",
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
        "方向",
        "个人主页",
        "任课教师",
        "上海市人民政府",
        "中国科学院",
        "是什么",
        "什么",
        "有哪些",
        "哪些",
        "可以",
        "推荐",
        "老师",
        "教师",
        "导师",
        "教授",
    }
    out: List[str] = []
    seen = set()

    profile_name_patterns = (
        r"([\u4e00-\u9fff]{2,4})的(?:研究方向|研究兴趣|邮箱|办公室|个人主页|主页|电话)",
        r"([\u4e00-\u9fff]{2,4})(?:主页|个人主页)",
        r"([\u4e00-\u9fff]{2,4})(?:教授|老师|导师|副教授|研究员)",
        r"([\u4e00-\u9fff]{2,4})(?:是否|是不是|能否|可否)(?:匹配|合适|适合|可以|列出|获得|是)",
        r"([\u4e00-\u9fff]{2,4})(?:是哪一年|哪一年|何时)(?:加入|晋升|获得|毕业|任)",
    )
    for pattern in profile_name_patterns:
        for match in re.finditer(pattern, query):
            phrase = match.group(1)
            if phrase in stop or phrase in seen:
                continue
            seen.add(phrase)
            out.append(phrase)
            if len(out) >= 5:
                return out

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


def _contains_profile_name_phrase(text: str, phrase: str) -> bool:
    allowed_after = ("博士", "教授", "老师", "导师", "副教授", "研究员", "的", "，", "。", "、", "；", "：", ":", " ", "\n", "\t", "|", ")", "）")
    start = 0
    while True:
        index = text.find(phrase, start)
        if index < 0:
            return False
        before = text[index - 1] if index > 0 else ""
        after_index = index + len(phrase)
        after = text[after_index] if after_index < len(text) else ""
        suffix = text[after_index : after_index + 3]
        before_ok = not before or not _is_cjk_char(before)
        after_ok = not after or not _is_cjk_char(after) or any(suffix.startswith(item) for item in allowed_after)
        if before_ok and after_ok:
            return True
        start = index + 1


def _expand_query(query: str) -> str:
    additions: List[str] = []
    lowered = query.lower()
    for trigger, values in QUERY_EXPANSIONS.items():
        if trigger.lower() in lowered:
            additions.extend(values)
    if _has_advisor_recommendation_intent(query):
        additions.extend(
            [
                "导师",
                "教授",
                "博士生导师",
                "博导",
                "研究方向",
                "个人主页",
                "faculty",
                "advisor",
                "supervisor",
                "research interests",
            ]
        )
        additions.extend(_expanded_research_topic_terms(query))
    if _has_scholarship_overview_intent(query):
        additions.extend(
            [
                "奖助学金",
                "研究生奖助体系",
                "研究生国家奖学金",
                "研究生学业奖学金",
                "研究生国家助学金",
                "上海科技大学研究生收费和奖助学金管理办法",
                "financial aid",
            ]
        )
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
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = resolve_db_path(db_path)
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

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        candidate_limit: int = 180,
        include_structured: bool = True,
    ) -> List[UnifiedSearchHit]:
        assert self.conn is not None
        candidates: Dict[int, Tuple[sqlite3.Row, float]] = {}
        for row, raw_score in self._exact_candidates(query, limit=max(top_k * 8, 40)):
            candidates[int(row["id"])] = (row, raw_score)
        if include_structured:
            for row, raw_score in self._structured_candidates(query, top_k=top_k):
                current = candidates.get(int(row["id"]))
                if current is None or raw_score > current[1]:
                    candidates[int(row["id"])] = (row, raw_score)
        for row, raw_score in self._fts_candidates(query, limit=candidate_limit):
            current = candidates.get(int(row["id"]))
            if current is None or raw_score > current[1]:
                candidates[int(row["id"])] = (row, raw_score)

        hits = [
            _row_to_hit(row, query, self._rerank(query, row, raw_score))
            for row, raw_score in candidates.values()
        ]
        hits.sort(key=lambda hit: hit.rank, reverse=True)
        return self._dedupe(hits, top_k=top_k, query=query)

    def structured_search(self, query: str, *, top_k: int = 64) -> List[UnifiedSearchHit]:
        assert self.conn is not None
        candidates: Dict[int, Tuple[sqlite3.Row, float]] = {}
        for row, raw_score in self._structured_candidates(query, top_k=top_k):
            current = candidates.get(int(row["id"]))
            if current is None or raw_score > current[1]:
                candidates[int(row["id"])] = (row, raw_score)
        hits = [
            _row_to_hit(row, query, self._rerank(query, row, raw_score))
            for row, raw_score in candidates.values()
        ]
        hits.sort(key=lambda hit: hit.rank, reverse=True)
        return self._dedupe(hits, top_k=top_k, query=query)

    def _structured_candidates(self, query: str, *, top_k: int) -> Iterable[Tuple[sqlite3.Row, float]]:
        if self.schema_variant == "clean_rag_data":
            return self._clean_supplemental_candidates(query, limit=max(top_k * 10, 60))
        return [
            *self._structured_route_candidates(query, limit=max(top_k * 12, 80)),
            *self._source_route_candidates(query, limit=max(top_k * 20, 200)),
        ]

    def neighbor_hits(
        self,
        hits: Sequence[UnifiedSearchHit],
        *,
        query: str,
        window: int = 1,
        limit: int = 32,
    ) -> List[UnifiedSearchHit]:
        assert self.conn is not None
        if not hits or window <= 0 or limit <= 0:
            return []

        pairs: List[Tuple[int, int]] = []
        seen_pairs = set()
        existing_ids = {hit.chunk_id for hit in hits}
        base_rank_by_pair: Dict[Tuple[int, int], float] = {}
        for hit in hits:
            row = self.conn.execute("select doc_id, chunk_index from chunks where id = ?", (hit.chunk_id,)).fetchone()
            if row is None:
                continue
            doc_id = int(row["doc_id"])
            chunk_index = int(row["chunk_index"])
            for delta in range(-window, window + 1):
                if delta == 0:
                    continue
                pair = (doc_id, chunk_index + delta)
                if pair in seen_pairs or pair[1] < 0:
                    continue
                seen_pairs.add(pair)
                pairs.append(pair)
                base_rank_by_pair[pair] = max(base_rank_by_pair.get(pair, hit.rank - 0.05), hit.rank - 0.05)

        out: List[UnifiedSearchHit] = []
        for doc_id, chunk_index in pairs:
            row = self.conn.execute(
                "select * from chunks where doc_id = ? and chunk_index = ?",
                (doc_id, chunk_index),
            ).fetchone()
            if row is None or int(row["id"]) in existing_ids:
                continue
            out.append(_row_to_hit(row, query, base_rank_by_pair.get((doc_id, chunk_index), 0.0)))
            if len(out) >= limit:
                break
        return out

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

    def _source_route_candidates(self, query: str, *, limit: int) -> Iterable[Tuple[sqlite3.Row, float]]:
        assert self.conn is not None
        query_lower = query.lower()
        clauses: List[str] = []

        if (
            "sist" in query_lower
            and (
                "英文介绍" in query
                or "about" in query_lower
                or "mission" in query_lower
                or "使命" in query
                or "五个学院" in query
                or "创办顺序" in query
                or "主要研究领域" in query
                or "research areas" in query_lower
                or "computer engineering" in query_lower
                or "computer science" in query_lower
                or "computing & mathematical" in query_lower
                or "interdisciplinary information technology" in query_lower
            )
        ):
            clauses.append("(lower(url) like '%sist_en/2724/list%' or lower(source_path) like '%sist_en/2724/list%')")

        if (
            "上海科技大学" in query
            and (
                "本科专业" in query
                or "专业覆盖" in query
                or "学生人数" in query
                or "在籍学生" in query
                or "截至2025年12月" in query
                or "本科生" in query
                or "硕士研究生" in query
                or "博士研究生" in query
                or "学校概况" in query
            )
        ):
            clauses.append(
                "("
                "lower(url) like '%shanghaitech.edu.cn/1054/main%' "
                "or lower(url) like '%shanghaitech.edu.cn/main.htm%' "
                "or lower(source_path) like '%1054_main%' "
                "or lower(source_path) like '%course_1054_main%'"
                ")"
            )

        if (
            ("sist" in query_lower or "信息学院" in query or "信息科学与技术学院" in query)
            and (
                "研究生招生" in query
                or "招生页" in query
                or "博士项目" in query
                or "学术型博士" in query
                or "专业型博士" in query
                or "招生方式" in query
                or "导师要求" in query
                or "前沿学科领域" in query
            )
        ):
            clauses.append("(lower(url) like '%sist.shanghaitech.edu.cn/yjszs%' or lower(source_path) like '%yjszs%')")

        if not clauses:
            return []

        rows = self.conn.execute(
            f"""
            select *
            from chunks
            where {' or '.join(clauses)}
            order by quality_score desc, length(text) desc, id asc
            limit ?
            """,
            (limit,),
        ).fetchall()
        return [(row, self._source_route_score(query, row)) for row in rows]

    def _source_route_score(self, query: str, row: sqlite3.Row) -> float:
        text = str(row["text"] or "")
        title = str(row["title"] or "")
        haystack = f"{title}\n{text}".lower()
        haystack_raw = f"{title}\n{text}"
        score = 70.0
        if ("主要研究领域" in query or "research areas" in query.lower()) and "main research areas" in haystack:
            score += 80.0
        if ("使命" in query or "mission" in query.lower()) and (
            "technology innovators" in haystack or "sist's mission" in haystack
        ):
            score += 80.0
        if ("创办顺序" in query or "第一个" in query or "five schools" in query.lower()) and "first of five schools" in haystack:
            score += 80.0
        if ("computer engineering" in query.lower() or "计算机工程" in query) and "computer engineering" in haystack:
            score += 70.0
        if ("computer science" in query.lower() or "机器学习" in query or "机器人" in query) and "computer science" in haystack:
            score += 50.0
        if "computing & mathematical" in query.lower() and "computing & mathematical science" in haystack:
            score += 70.0
        if "interdisciplinary information technology" in query.lower() and "interdisciplinary information technology" in haystack:
            score += 70.0
        if ("本科专业" in query or "专业覆盖" in query) and (
            "本科专业12个" in haystack or "涵盖物理、化学、材料、生物、信息、管理、创意与艺术、数学、历史" in haystack_raw
        ):
            score += 130.0
        if ("截至2025年12月" in query or "学生人数" in query or "本科生" in query) and "截至2025年12月" in haystack:
            score += 130.0
            if all(term in haystack_raw for term in ("本科生2082", "硕士研究生", "2857", "博士研究生", "2029")):
                score += 120.0
        if "前沿学科领域" in query and "覆盖计算机科学与技术、电子科学与技术、信息与通信工程" in haystack:
            score += 80.0
        if ("学术型博士" in query or "专业型博士" in query or "导师要求" in query) and "专业型博士项目" in haystack:
            score += 60.0
        if "学术型博士" in query and "专业型博士" in query and "学术型博士项目" in haystack_raw and "专业型博士项目" in haystack_raw:
            score += 160.0
            if "全日制非定向" in haystack_raw:
                score += 70.0
            if "常任教授或特聘教授" in haystack_raw:
                score += 70.0
            if "非全日制定向" in haystack_raw:
                score += 60.0
            if "工程实践经验" in haystack_raw and "行业专家" in haystack_raw:
                score += 70.0
        if ("普通招考" in query or "硕士学位" in query or "博士项目" in query) and "普通招考" in haystack:
            score += 50.0
        return score

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

        for entity in _query_ascii_entities(query):
            entity_lower = entity.lower()
            variants = [entity_lower]
            if entity_lower.endswith("pie") and len(entity_lower) > 3:
                variants.append(entity_lower[:-3] + " pie")
            entity_clauses = []
            for variant in variants:
                entity_clauses.append("(lower(title) like ? or lower(text) like ? or lower(url) like ? or lower(source_path) like ?)")
                like = f"%{variant}%"
                params.extend([like, like, like, like])
            clauses.append("(" + " or ".join(entity_clauses) + ")")

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

    def _structured_route_candidates(self, query: str, *, limit: int) -> Iterable[Tuple[sqlite3.Row, float]]:
        assert self.conn is not None
        clauses: List[str] = []
        params: List[Any] = []
        years = _query_years(query)
        course_codes = _query_course_codes(query)
        program_codes = _query_program_codes(query)

        def category_clause(categories: Sequence[str]) -> str:
            placeholders = ",".join("?" for _ in categories)
            params.extend(categories)
            return f"category in ({placeholders})"

        if _has_course_intent(query):
            base = [
                "("
                + category_clause(sorted(COURSE_ROUTE_CATEGORIES))
                + " or (source_type in ('structured_table', 'structured_json') and "
                + "(lower(source_path) like '%course%' or lower(source_path) like '%schedule%' "
                + "or lower(title) like '%course%' or title like '%课程%' or title like '%课表%'))"
                + ")"
            ]
            if course_codes:
                code_clauses = []
                for code in course_codes:
                    code_clauses.append("(upper(title) like ? or upper(text) like ? or upper(source_path) like ?)")
                    code_like = f"%{code}%"
                    params.extend([code_like, code_like, code_like])
                base.append("(" + " or ".join(code_clauses) + ")")
            if years:
                year_clauses = []
                for year in years:
                    year_clauses.append("(title like ? or text like ? or source_path like ? or url like ?)")
                    params.extend([f"%{year}%", f"%{year}%", f"%{year}%", f"%{year}%"])
                base.append("(" + " or ".join(year_clauses) + ")")
            if _has_schedule_intent(query):
                base.append(
                    "(lower(title) like '%schedule%' or lower(source_path) like '%schedule%' "
                    "or title like '%课表%' or text like '%教学中心%' or text like '%周%')"
                )
            clauses.append("(" + " and ".join(base) + ")")

        if _has_degree_intent(query):
            base = [
                "("
                + category_clause(sorted(PROGRAM_ROUTE_CATEGORIES))
                + " or (source_type in ('structured_table', 'structured_json', 'training', 'sist_text') and "
                + "(lower(source_path) like '%program%' or lower(title) like '%program%' "
                + "or title like '%培养方案%' or text like '%培养目标%'))"
                + ")"
            ]
            if years:
                year_clauses = []
                for year in years:
                    year_clauses.append("(title like ? or text like ? or source_path like ? or url like ?)")
                    params.extend([f"%{year}%", f"%{year}%", f"%{year}%", f"%{year}%"])
                base.append("(" + " or ".join(year_clauses) + ")")
            if program_codes:
                code_clauses = []
                for code in program_codes:
                    if code == "CS":
                        code_clauses.append("(text like '%计算机科学与技术%' or lower(text) like '%computer science%' or lower(url) like '%in%20cs%')")
                    elif code == "EE":
                        code_clauses.append("(text like '%电子信息工程%' or lower(text) like '%electrical%' or lower(text) like '%electronic%' or lower(url) like '%in%20ee%')")
                    else:
                        code_clauses.append("(upper(title) like ? or upper(text) like ? or upper(url) like ?)")
                        code_like = f"%{code}%"
                        params.extend([code_like, code_like, code_like])
                base.append("(" + " or ".join(code_clauses) + ")")
            clauses.append("(" + " and ".join(base) + ")")

        if _profileish_query(query):
            names = [
                phrase
                for phrase in _query_exact_cjk_phrases(query)
                if phrase not in {"教授", "教师", "导师", "邮箱", "办公室", "研究方向"}
            ]
            if names:
                base = [
                    "("
                    + category_clause(sorted(FACULTY_ROUTE_CATEGORIES))
                    + " or (source_type in ('structured_table', 'structured_json', 'sist_text', 'web') and "
                    + "(lower(source_path) like '%faculty%' or title like '%师资%' or text like '%研究方向%'))"
                    + ")"
                ]
                name_clauses = []
                for name in names:
                    name_clauses.append("(title like ? or text like ? or url like ? or source_path like ?)")
                    params.extend([f"%{name}%", f"%{name}%", f"%{name}%", f"%{name}%"])
                base.append("(" + " or ".join(name_clauses) + ")")
                clauses.append("(" + " and ".join(base) + ")")

        if _has_advisor_recommendation_intent(query):
            topic_terms = _expanded_research_topic_terms(query)
            if topic_terms:
                base = [
                    "("
                    + category_clause(sorted(FACULTY_ROUTE_CATEGORIES))
                    + " or (source_type in ('rag_json', 'structured_table', 'structured_json', 'sist_text', 'web', 'raw_html') and "
                    + "(lower(source_path) like '%faculty%' or lower(source_path) like '%prof%' "
                    + "or lower(url) like '%main.htm%' or text like '%研究方向%' "
                    + "or text like '%博导%' or text like '%博士生导师%' "
                    + "or lower(text) like '%research interests%'))"
                    + ")",
                    "(text like '%研究方向%' or text like '%博导%' or text like '%博士生导师%' "
                    "or text like '%助理教授%' or text like '%副教授%' or text like '%教授、研究员%' "
                    "or text like '%个人主页:%' or lower(text) like '%research interests%' "
                    "or lower(url) like '%main.htm%')",
                ]
                topic_clauses = []
                for term in topic_terms:
                    like = f"%{term}%"
                    topic_clauses.append("(title like ? or text like ? or url like ? or source_path like ?)")
                    params.extend([like, like, like, like])
                base.append("(" + " or ".join(topic_clauses) + ")")
                clauses.append("(" + " and ".join(base) + ")")

        if _has_policy_intent(query):
            phrases = _query_exact_cjk_phrases(query)
            base = [
                "("
                + category_clause(sorted(POLICY_ROUTE_CATEGORIES))
                + " or source_type in ('rag_json', 'structured_table')"
                + ")",
                "(text like '%发布时间:%' or text like '%发布者:%' or text like '%published_at%' or text like '%valid_from%' or title like '%细则%' or title like '%办法%' or title like '%制度%')",
            ]
            if phrases:
                phrase_clauses = []
                for phrase in phrases:
                    phrase_clauses.append("(title like ? or text like ?)")
                    params.extend([f"%{phrase}%", f"%{phrase}%"])
                base.append("(" + " or ".join(phrase_clauses) + ")")
            clauses.append("(" + " and ".join(base) + ")")

        if _has_scholarship_overview_intent(query):
            scholarship_clauses = []
            for marker in SCHOLARSHIP_POLICY_MARKERS:
                scholarship_clauses.append("(title like ? or text like ? or url like ? or source_path like ?)")
                like = f"%{marker}%"
                params.extend([like, like, like, like])
            clauses.append("(" + " or ".join(scholarship_clauses) + ")")

        if _has_sist_overview_intent(query):
            clauses.append(
                "("
                "category in ('sist_overview', 'university_overview') "
                "or url in ('https://sist.shanghaitech.edu.cn/', 'https://sist.shanghaitech.edu.cn/sist_en/', 'https://faculty.sist.shanghaitech.edu.cn/') "
                "or lower(text) like '%sist at a glance%' "
                "or lower(text) like '%about sist%' "
                "or lower(text) like '%vision and mission%'"
                ")"
            )

        if not clauses:
            return []

        rows = self.conn.execute(
            f"""
            select *
            from chunks
            where {' or '.join(clauses)}
            order by
                source_type in ('structured_table', 'structured_json') desc,
                source_type = 'rag_json' desc,
                quality_score desc,
                id asc
            limit ?
            """,
            (*params, limit),
        ).fetchall()
        return [(row, 32.0) for row in rows]

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

        if _has_advisor_recommendation_intent(query):
            topic_terms = _expanded_research_topic_terms(query)
            if topic_terms:
                base = [
                    "category = 'sist_faculty'",
                    "(text like '%研究方向%' or text like '%博导%' or text like '%博士生导师%' "
                    "or text like '%助理教授%' or text like '%副教授%' or text like '%个人主页:%' "
                    "or lower(text) like '%research interests%' or lower(source_url) like '%main.htm%')",
                ]
                topic_clauses = []
                for term in topic_terms:
                    like = f"%{term}%"
                    topic_clauses.append("(title like ? or text like ? or source_url like ?)")
                    params.extend([like, like, like])
                base.append("(" + " or ".join(topic_clauses) + ")")
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

        compact_title = _compact_ascii(title)
        compact_head = _compact_ascii(text[:1800])
        compact_haystack = _compact_ascii(haystack)
        query_entities = _query_ascii_entities(query)
        entity_in_title = False
        entity_in_head = False
        entity_in_haystack = False
        for entity in query_entities:
            compact_entity = _compact_ascii(entity)
            if not compact_entity:
                continue
            if compact_entity in compact_title:
                entity_in_title = True
                entity_in_haystack = True
                score += 38.0
            if compact_entity in compact_head:
                entity_in_head = True
                entity_in_haystack = True
                score += 28.0
            elif compact_entity in compact_haystack:
                entity_in_haystack = True
                score += 12.0

        if _has_intro_intent(query):
            text_head = text[:2200]
            intro_count = _marker_count(text_head, INTRO_EVIDENCE_MARKERS)
            formal_count = _marker_count(text_head, FORMAL_INTRO_MARKERS)
            if query_entities:
                compact_early_text = _compact_ascii(text_head[:420])
                early_entity = any(_compact_ascii(entity) in compact_early_text for entity in query_entities)
                early_formal_count = _marker_count(text_head[:420], FORMAL_INTRO_MARKERS)
                if intro_count and entity_in_head:
                    score += 24.0 + min(44.0, intro_count * 8.0)
                elif intro_count and entity_in_haystack:
                    score += 10.0 + min(24.0, intro_count * 4.0)
                if formal_count and entity_in_head:
                    score += 42.0 + min(36.0, formal_count * 9.0)
                if formal_count and entity_in_head and source_type in STRUCTURED_SOURCE_TYPES:
                    score += 12.0
                if early_entity and early_formal_count:
                    score += 54.0
                    if source_type == "sist_text":
                        score += 18.0
                    if category == "program":
                        score += 18.0
                    elif category == "admission":
                        score += 6.0
                elif formal_count and source_type in {"rag_json", "raw_html", "text_pages", "web"}:
                    score -= 18.0

                title_lower = title.lower()
                url_lower = str(row["url"] or "").lower()
                path_lower = str(row["source_path"] or "").lower()
                listish = (
                    any(marker in title for marker in LIST_PAGE_TITLES)
                    or "list.htm" in url_lower
                    or "list.htm" in path_lower
                )
                eventish = any(marker in title_lower for marker in (" day", "pi day")) or any(
                    marker in title for marker in ("派对", "游园会")
                )
                if listish and formal_count == 0:
                    score -= 42.0
                if eventish and formal_count == 0:
                    score -= 28.0
                if source_type in {"raw_html", "web"} and formal_count == 0:
                    score -= 14.0
                if source_type == "raw_html" and len(title) > 160 and any(
                    marker in title for marker in ("导航", "学院概况", "招生工作")
                ):
                    score -= 18.0
            elif intro_count:
                score += 26.0

        for code in _query_course_codes(query):
            if code == title.upper():
                score += 35.0
            if f"课程代码: {code}" in text or f"# {code}" in text:
                score += 45.0
            elif _contains_course_code(haystack, code):
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
            if category in {"contacts", "faculty_members", "sist_faculty", "professors_enriched", "leadership", "leadership_roles"}:
                score += 8.0
            if "邮箱:" in text or "email" in haystack_lower or "办公室" in text or "office" in haystack_lower:
                score += 4.0

        if _profileish_query(query):
            phrases = _query_exact_cjk_phrases(query)
            matched_profile_name = False
            for phrase in phrases:
                if title == phrase:
                    score += 90.0
                    matched_profile_name = True
                elif _contains_exact_cjk_phrase(title, phrase) or _contains_profile_name_phrase(title, phrase):
                    score += 18.0
                    matched_profile_name = True
                phrase_in_profile = _contains_profile_name_phrase(text[:1800], phrase)
                if (
                    phrase_in_profile
                    and (
                        text.startswith(phrase + " ")
                        or text.startswith(phrase + "\n")
                        or f"姓名:{phrase}" in text
                        or f"姓名: {phrase}" in text
                    )
                ):
                    score += 42.0
                    matched_profile_name = True
                elif phrase_in_profile or _contains_exact_cjk_phrase(haystack, phrase):
                    score += 8.0
                    matched_profile_name = True
                if phrase_in_profile and _has_faculty_profile_marker(text[:2500]):
                    score += 60.0
                if phrase_in_profile and any(
                    marker.lower() in text[:2800].lower()
                    for marker in ("研究方向", "研究兴趣", "当前研究领域", "主要研究方向", "research interests")
                ):
                    score += 42.0
            if phrases and category in FACULTY_ROUTE_CATEGORIES and not matched_profile_name:
                score -= 120.0
            if phrases and source_type in {"sist_text", "web", "rag_json"} and _has_faculty_profile_marker(text[:2500]) and not matched_profile_name:
                score -= 70.0
            if any(marker in text for marker in ("个人主页:", "研究方向:", "邮箱:", "办公室:", "博士毕业院校:")):
                score += 12.0
            if category in {"faculty", "faculty_members", "sist_faculty", "professors_enriched", "leadership", "leadership_roles"}:
                score += 8.0

        if _has_advisor_recommendation_intent(query):
            topic_terms = _expanded_research_topic_terms(query)
            has_topic = any(term.lower() in haystack_lower for term in topic_terms)
            has_profile_marker = _has_faculty_profile_marker(haystack[:3500])
            has_faculty_title = any(
                marker in text[:1800]
                for marker in ("博士生导师", "博导", "助理教授", "副教授", "教授、研究员", "研究员、博导")
            )
            url_lower = str(row["url"] or "").lower()
            path_lower = str(row["source_path"] or "").lower()

            if has_topic:
                score += 18.0
            if has_profile_marker:
                score += 38.0
            if has_topic and has_profile_marker:
                score += 105.0
            if has_topic and "研究方向" in text[:2500]:
                score += 36.0
            if has_faculty_title:
                score += 48.0
            if "main.htm" in url_lower or "/prof_" in path_lower or "_main.htm" in path_lower:
                score += 36.0
            if category in FACULTY_ROUTE_CATEGORIES:
                score += 20.0
            if source_type in {"web", "sist_text", "structured_json", "rag_json"} and has_topic and has_profile_marker:
                score += 18.0

            if _has_non_faculty_member_noise(text[:3000]) and not has_faculty_title:
                score -= 110.0
            if category in COURSE_ROUTE_CATEGORIES and not has_profile_marker:
                score -= 70.0
            if category in PROGRAM_ROUTE_CATEGORIES | {"admission", "sist_overview", "university_overview"} and not has_profile_marker:
                score -= 65.0
            if category in {"events", "news", "sist_news_events", "PDF通告"} and not has_profile_marker:
                score -= 70.0
            if source_type == "rag_json" and category in {"教授课程数据", "纯文本资料", "PDF通告"} and not has_faculty_title:
                score -= 55.0

        if "日常管理" in query and "上海科技大学" in query:
            if "上海市人民政府负责日常管理" in haystack or "负责日常管理的全日制普通高等学校" in haystack:
                score += 45.0
            if any(term in title + text[:800] for term in ("仪器设备", "科研经费", "基建项目", "采购")):
                score -= 28.0

        if any(term in query for term in ("开放时间", "图书馆")) and "library" in str(row["host"] or ""):
            score += 10.0

        score += _structured_route_boost(query, row)
        if self.schema_variant == "clean_rag_data":
            score += _clean_intent_boost(query, row)

        return score

    def _dedupe(self, hits: Sequence[UnifiedSearchHit], *, top_k: int, query: str = "") -> List[UnifiedSearchHit]:
        out: List[UnifiedSearchHit] = []
        seen_text = set()
        seen_path_title = set()
        seen_title_text = set()
        source_counts: Dict[Tuple[str, str], int] = {}
        title_counts: Dict[str, int] = {}
        max_per_source = 2 if top_k >= 6 else 1
        query_codes = _query_course_codes(query)
        query_phrases = _query_exact_cjk_phrases(query)
        for hit in hits:
            normalized_text = re.sub(r"\s+", "", hit.text)
            text_key = normalized_text[:700]
            path_title_key = (hit.path, hit.title, hit.chunk_index)
            title_key = re.sub(r"\s+", "", hit.title.lower())
            title_text_key = (title_key, normalized_text[:240])
            if hit.url:
                source_key = ("url", re.sub(r"[#?].*$", "", hit.url).rstrip("/").lower())
            else:
                source_key = ("path", re.sub(r"#chunk=\d+.*$", "", hit.path).rstrip("/").lower())
            if text_key in seen_text or path_title_key in seen_path_title or title_text_key in seen_title_text:
                continue
            if source_counts.get(source_key, 0) >= max_per_source:
                continue
            if title_key:
                title_lower = hit.title.lower()
                title_matches_query = any(code in hit.title.upper() for code in query_codes) or any(
                    phrase and phrase in hit.title for phrase in query_phrases
                )
                title_is_multi_fact = any(
                    marker in title_lower or marker in hit.title
                    for marker in ("schedule", "class sched", "课程表", "课表", "培养方案", "program")
                )
                title_limit = 2 if title_matches_query or title_is_multi_fact else 1
                if title_counts.get(title_key, 0) >= title_limit:
                    continue
            seen_text.add(text_key)
            seen_path_title.add(path_title_key)
            seen_title_text.add(title_text_key)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            if title_key:
                title_counts[title_key] = title_counts.get(title_key, 0) + 1
            out.append(hit)
            if len(out) >= top_k:
                break
        return out
