from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional

from qwen_api import QwenClient

from .query_keywords import normalize_keyword, parse_keywords


@dataclass(frozen=True)
class SearchRolloutDecision:
    action: str
    keywords: List[str] = field(default_factory=list)
    query: str = ""
    note: str = ""
    raw: str = ""
    error: str = ""

    @property
    def wants_search(self) -> bool:
        return self.action == "search" and bool(self.keywords or self.query)


@dataclass(frozen=True)
class SearchRolloutStep:
    step: int
    decision: SearchRolloutDecision
    search_query: str = ""
    hit_count: int = 0
    new_hit_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.decision.action,
            "keywords": self.decision.keywords,
            "query": self.decision.query,
            "note": self.decision.note,
            "search_query": self.search_query,
            "hit_count": self.hit_count,
            "new_hit_count": self.new_hit_count,
            "raw": self.decision.raw,
            "error": self.decision.error,
        }


SYSTEM_PROMPT = """你是一个RAG检索rollout控制器。
你会看到用户问题和当前已经检索到的证据。你必须决定:
1. 当前证据是否已经足够回答用户问题;
2. 如果不够, 下一步应该搜索哪些关键词。

只输出JSON,不要输出解释或思考过程。格式二选一:
{"action":"answer","note":"证据已足够/或已经无法继续有效搜索"}
{"action":"search","keywords":["关键词1","关键词2"],"query":"用于下一轮检索的短查询","note":"缺少什么证据"}

规则:
1. 只能根据证据判断,不要使用资料外知识。
2. 如果证据已直接支持问题中的所有关键实体、数值、年份、课程号和教师姓名,选择answer。
3. 如果证据缺少某个实体的主页、课程记录、招生页、新闻页或对比对象,选择search。
4. 搜索关键词要具体,最多10个。优先包含姓名、英文名、课程号、页面主题、年份、研究方向、邮箱、办公室等。
5. 如果要从课程跳到教师主页,下一轮不要带原课程号,应搜索教师姓名/英文名 + profile/homepage/研究方向。
6. 如果要比较两个对象,缺哪个对象就搜索哪个对象的课程号或名称。
7. 如果连续证据明显无关,换一组更具体的关键词。
8. 不要编造答案,不要在JSON里写最终答案。"""


def normalize_query(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" ,，;；、。.!！?？:：\"'`[]{}()（）<>《》")
    return text[:180]


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


def parse_rollout_decision(raw: str) -> SearchRolloutDecision:
    for candidate in _json_candidates(raw):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        action = normalize_query(obj.get("action")).lower()
        if action not in {"answer", "search"}:
            action = "answer"
        raw_keywords = obj.get("keywords", [])
        if isinstance(raw_keywords, list):
            keywords = [normalize_keyword(item) for item in raw_keywords if normalize_keyword(item)]
        elif isinstance(raw_keywords, str):
            keywords = parse_keywords(raw_keywords, max_keywords=10)
        else:
            keywords = []
        query = normalize_query(obj.get("query"))
        note = normalize_query(obj.get("note") or obj.get("reason"))
        return SearchRolloutDecision(action=action, keywords=keywords, query=query, note=note, raw=raw)

    keywords = parse_keywords(raw, max_keywords=10)
    if keywords:
        return SearchRolloutDecision(action="search", keywords=keywords, query=" ".join(keywords), raw=raw)
    return SearchRolloutDecision(action="answer", raw=raw, error="invalid_json")


def make_rollout_search_query(decision: SearchRolloutDecision) -> str:
    parts: List[str] = []
    if decision.query:
        parts.append(decision.query)
    for keyword in decision.keywords:
        keyword = normalize_keyword(keyword)
        if keyword and keyword not in parts:
            parts.append(keyword)
    return " ".join(parts).strip()


def build_rollout_messages(
    *,
    query: str,
    evidence: str,
    previous_searches: List[str],
    step: int,
    max_steps: int,
) -> List[Mapping[str, str]]:
    previous = "\n".join(f"- {item}" for item in previous_searches) or "无"
    user = f"""用户问题:
{query}

当前证据:
{evidence}

已经搜索过:
{previous}

当前可继续搜索步数: {max(max_steps - step + 1, 0)}

请判断是 answer 还是 search。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def decide_next_search(
    client: QwenClient,
    *,
    query: str,
    evidence: str,
    previous_searches: List[str],
    step: int,
    max_steps: int,
    max_tokens: Optional[int] = 512,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> SearchRolloutDecision:
    result = client.chat(
        messages=build_rollout_messages(
            query=query,
            evidence=evidence,
            previous_searches=previous_searches,
            step=step,
            max_steps=max_steps,
        ),
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    raw = result.answer.strip() or result.raw.strip()
    return parse_rollout_decision(raw)
