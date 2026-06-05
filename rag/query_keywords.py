from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional

from qwen_api import QwenClient


GENERIC_QUERY_TERMS = (
    "介绍一下",
    "简单介绍",
    "介绍",
    "是什么",
    "什么是",
    "请问",
    "告诉我",
    "帮我查",
    "查询",
    "一下",
    "about",
    "tell me",
    "introduce",
    "introduction",
)


@dataclass(frozen=True)
class QueryKeywordPlan:
    enabled: bool
    search_query: str
    keywords: List[str] = field(default_factory=list)
    raw: str = ""
    error: str = ""


SYSTEM_PROMPT = """你是一个RAG检索查询规划器。
你的任务不是回答问题,而是把用户问题改写成适合检索上海科技大学和SIST知识库的关键词。

只输出一个JSON对象,不要输出解释。JSON格式:
{"keywords":["关键词1","关键词2"]}

规则:
1. 保留问题里的课程号、年份、教师姓名、学院名、专业名、研究中心名、英文术语。
2. 可以补充中英文别名和同义词,例如 任课教师/Instructor, 研究方向/research interests。
3. 如果用户问“介绍/简介/是什么/概况/overview”某个学院或机构,保留实体名,并补充概况类检索词,例如 About SIST、SIST overview、Vision and Mission、SIST AT A GLANCE。
4. 删除“介绍一下”“是什么”“请问”“告诉我”“帮我查”等普通问句词,只保留可用于检索的实体、属性和同义词。
5. 关键词要短,不要输出完整句子。
6. 不要猜答案,不要编造事实。
7. 最多输出12个关键词。"""


def build_keyword_prompt(query: str) -> List[Mapping[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"用户问题:\n{query}\n\n请输出检索关键词JSON。"},
    ]


def normalize_keyword(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    lower = text.lower()
    for term in GENERIC_QUERY_TERMS:
        if re.fullmatch(re.escape(term), lower, flags=re.IGNORECASE):
            return ""
        text = re.sub(re.escape(term), "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" ,，;；、。.!！?？:：\"'`[]{}()（）<>《》")
    return text


def _json_candidates(text: str) -> Iterable[str]:
    stripped = text.strip()
    if not stripped:
        return
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        yield fence.group(1).strip()
    yield stripped
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = stripped.find(start_char)
        end = stripped.rfind(end_char)
        if start >= 0 and end > start:
            yield stripped[start : end + 1]


def _flatten_json_keywords(obj: Any) -> List[str]:
    if isinstance(obj, list):
        return [normalize_keyword(item) for item in obj]
    if not isinstance(obj, dict):
        return []

    values: List[str] = []
    for key in ("keywords", "must_have", "aliases", "search_terms", "terms"):
        item = obj.get(key)
        if isinstance(item, list):
            values.extend(normalize_keyword(value) for value in item)
        elif isinstance(item, str):
            values.extend(normalize_keyword(part) for part in re.split(r"[,，;；、\n]+", item))
    return values


def _parse_keyword_array_fragment(text: str) -> List[str]:
    match = re.search(r'"keywords"\s*:\s*\[(.*)', text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    return [normalize_keyword(item) for item in re.findall(r'"([^"]+)"', match.group(1))]


def parse_keywords(text: str, *, max_keywords: int = 12) -> List[str]:
    values: List[str] = []
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        values.extend(_flatten_json_keywords(parsed))
        if values:
            break

    if not values:
        values.extend(_parse_keyword_array_fragment(text))

    if not values:
        thinking_markers = ("thinking process", "analyze user input", "apply rules", "self-correction")
        lower = text.lower()
        if len(text) <= 500 and not any(marker in lower for marker in thinking_markers):
            values.extend(normalize_keyword(part) for part in re.split(r"[,，;；、\n]+", text))

    out: List[str] = []
    seen = set()
    for value in values:
        if not value or len(value) > 80:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max_keywords:
            break
    return out


def make_search_query(query: str, keywords: Iterable[str]) -> str:
    return " ".join(keyword for keyword in keywords if keyword)


def generate_query_keywords(
    client: QwenClient,
    query: str,
    *,
    max_keywords: int = 12,
    max_tokens: Optional[int] = 256,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> QueryKeywordPlan:
    result = client.chat(
        messages=build_keyword_prompt(query),
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    raw = result.answer.strip() or result.raw.strip()
    keywords = parse_keywords(raw, max_keywords=max_keywords)
    return QueryKeywordPlan(
        enabled=True,
        search_query=make_search_query(query, keywords),
        keywords=keywords,
        raw=raw,
    )
