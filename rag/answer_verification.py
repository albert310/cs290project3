from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional

from qwen_api import QwenClient

from .query_keywords import normalize_keyword, parse_keywords


@dataclass(frozen=True)
class AnswerVerification:
    enabled: bool
    keywords: List[str] = field(default_factory=list)
    search_query: str = ""
    search_queries: List[str] = field(default_factory=list)
    raw: str = ""
    error: str = ""
    hit_count: int = 0
    new_hit_count: int = 0
    draft_answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "keywords": self.keywords,
            "search_query": self.search_query,
            "search_queries": self.search_queries,
            "raw": self.raw,
            "error": self.error,
            "hit_count": self.hit_count,
            "new_hit_count": self.new_hit_count,
            "draft_answer": self.draft_answer,
        }


SYSTEM_PROMPT = """你是一个RAG答案二次核验检索规划器。
你会看到用户问题和一版待核验答案。你的任务不是回答问题,而是从待核验答案中抽取最需要二次检索核验的关键词。

只输出JSON,不要输出解释或思考过程。格式:
{"keywords":["关键词1","关键词2"],"query":"短检索式"}

规则:
1. 必须覆盖待核验答案中的关键实体、课程号、教师姓名、英文名、年份、邮箱、学校名、会议名、数值或结论。
2. 优先保留英文原文和官方表述,例如 Lehigh University、large language models、CS282、Hao Wang。
3. 如果答案是拒答,关键词应来自用户问题中的关键实体和缺失字段,用于确认是否真的没有资料。
4. 关键词要短,不要输出完整句子。
5. 最多输出10个关键词。
6. 不要编造答案中没有、问题中也没有的事实。"""


def _json_candidates(text: str) -> Iterable[str]:
    stripped = text.strip()
    if not stripped:
        return
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        yield fence.group(1).strip()
    yield stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        yield stripped[start : end + 1]


def _parse_query(raw: str) -> str:
    for candidate in _json_candidates(raw):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            query = normalize_keyword(obj.get("query"))
            if query:
                return query
    return ""


def build_verification_keyword_messages(query: str, answer: str) -> List[Mapping[str, str]]:
    user = f"""用户问题:
{query}

待核验答案:
{answer}

请输出二次核验检索关键词JSON。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def make_verification_search_query(query: str, keywords: Iterable[str], planned_query: str = "") -> str:
    parts: List[str] = []
    if planned_query:
        parts.append(normalize_keyword(planned_query))
    for keyword in keywords:
        value = normalize_keyword(keyword)
        if value and value not in parts:
            parts.append(value)
    if not parts:
        parts.append(normalize_keyword(query))
    return " ".join(parts).strip()


def generate_answer_verification(
    client: QwenClient,
    *,
    query: str,
    answer: str,
    max_keywords: int = 10,
    max_tokens: Optional[int] = 256,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> AnswerVerification:
    result = client.chat(
        messages=build_verification_keyword_messages(query, answer),
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    raw = result.answer.strip() or result.raw.strip()
    keywords = parse_keywords(raw, max_keywords=max_keywords)
    planned_query = _parse_query(raw)
    return AnswerVerification(
        enabled=True,
        keywords=keywords,
        search_query=make_verification_search_query(query, keywords, planned_query),
        raw=raw,
        draft_answer=answer,
    )
