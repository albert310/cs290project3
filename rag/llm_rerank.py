from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Sequence, TYPE_CHECKING

from qwen_api import QwenClient

if TYPE_CHECKING:
    from .unified_index import UnifiedSearchHit


SYSTEM_PROMPT = """你是一个RAG证据重排器。
你的任务不是回答问题,而是在候选资料中选出最能直接支持答案的证据并排序。

只输出一个JSON对象,不要输出解释或思考过程。格式:
{"ranked_indices":[1,3,2]}

规则:
1. ranked_indices 只能包含候选资料编号,按相关性从高到低排序。
2. 优先选择直接包含用户问题关键实体、课程号、教师姓名、年份、数值、邮箱、办公室、政策名称的资料。
3. 官方页面、结构化表格、教师主页、课程记录优先于导航页、列表页、泛泛新闻和只含相似词的资料。
4. 对时间敏感的问题,优先选择日期更新、仍然有效或更接近用户问题年份的资料。
5. 降低重复页面、重复片段、同一来源的近似拷贝、信息量很少的列表页或导航页优先级。
6. 含有更多可验证字段的资料更优先,例如教师个人主页、课程记录、联系方式、研究方向、办公室、邮箱、URL和发布日期。
7. 如果问题需要多跳证据,例如课程到任课教师主页,应同时保留课程记录和教师主页资料。
8. 如果候选资料都弱相关,仍按相对有用程度排序,不要编造候选编号。
9. 最多返回需要放入上下文的前N个编号。"""


@dataclass(frozen=True)
class LLMRerankResult:
    enabled: bool
    stage: str = ""
    query: str = ""
    candidate_count: int = 0
    selected_count: int = 0
    ranked_indices: List[int] = field(default_factory=list)
    input_chunk_ids: List[int] = field(default_factory=list)
    selected_chunk_ids: List[int] = field(default_factory=list)
    raw: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "stage": self.stage,
            "query": self.query,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "ranked_indices": self.ranked_indices,
            "input_chunk_ids": self.input_chunk_ids,
            "selected_chunk_ids": self.selected_chunk_ids,
            "raw": self.raw,
            "error": self.error,
        }


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


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
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start >= 0 and end > start:
        yield stripped[start : end + 1]


def _flatten_indices(value: Any) -> List[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(item) for item in re.findall(r"\d+", value)]
    if isinstance(value, list):
        out: List[int] = []
        for item in value:
            if isinstance(item, dict):
                for key in ("index", "idx", "id", "candidate"):
                    if key in item:
                        out.extend(_flatten_indices(item[key]))
                        break
            else:
                out.extend(_flatten_indices(item))
        return out
    return []


def _parse_ranked_indices(raw: str, candidate_count: int) -> List[int]:
    values: List[int] = []
    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            for key in ("ranked_indices", "ranking", "indices", "ranked", "sources"):
                if key in parsed:
                    values = _flatten_indices(parsed[key])
                    break
        elif isinstance(parsed, list):
            values = _flatten_indices(parsed)
        if values:
            break

    if not values:
        match = re.search(
            r'"(?:ranked_indices|ranking|indices|ranked|sources)"\s*:\s*\[([^\]]+)\]',
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            values = [int(item) for item in re.findall(r"\d+", match.group(1))]

    out: List[int] = []
    seen = set()
    for value in values:
        if value < 1 or value > candidate_count or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _candidate_block(index: int, hit: "UnifiedSearchHit", *, max_chars: int) -> str:
    meta = [
        f"编号: {index}",
        f"当前规则分: {hit.rank:.3f}",
        f"标题: {hit.title or hit.category or hit.source_type}",
        f"来源类型: {hit.source_type}",
        f"类别: {hit.category}",
    ]
    if hit.date:
        meta.append(f"日期: {hit.date}")
    if hit.url:
        meta.append(f"URL: {hit.url}")
    else:
        meta.append(f"路径: {hit.path}")
    body = trim_text(re.sub(r"\s+", " ", hit.text).strip(), max_chars)
    return "\n".join(meta + [f"正文: {body}"])


def build_rerank_messages(
    query: str,
    hits: Sequence["UnifiedSearchHit"],
    *,
    top_k: int,
    max_chars_per_hit: int,
) -> List[Mapping[str, str]]:
    candidates = "\n\n".join(
        _candidate_block(index, hit, max_chars=max_chars_per_hit)
        for index, hit in enumerate(hits, start=1)
    )
    user = f"""用户问题:
{query}

需要返回的数量N: {top_k}

候选资料:
{candidates}

请只输出JSON。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def rerank_hits_with_llm(
    client: QwenClient,
    query: str,
    hits: Sequence["UnifiedSearchHit"],
    *,
    top_k: int,
    stage: str = "",
    max_candidates: int = 24,
    max_chars_per_hit: int = 700,
    max_tokens: Optional[int] = 768,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> tuple[List["UnifiedSearchHit"], LLMRerankResult]:
    candidates = list(hits[: max(top_k, max_candidates)])
    if not candidates:
        result = LLMRerankResult(enabled=True, stage=stage, query=query)
        return [], result

    input_chunk_ids = [hit.chunk_id for hit in candidates]
    raw = ""
    try:
        response = client.chat(
            messages=build_rerank_messages(
                query,
                candidates,
                top_k=top_k,
                max_chars_per_hit=max_chars_per_hit,
            ),
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        raw = response.answer.strip() or response.raw.strip()
        ranked_indices = _parse_ranked_indices(raw, len(candidates))
        if not ranked_indices:
            raise ValueError("LLM reranker returned no valid candidate indices")
    except Exception as exc:
        selected = candidates[:top_k]
        result = LLMRerankResult(
            enabled=True,
            stage=stage,
            query=query,
            candidate_count=len(candidates),
            selected_count=len(selected),
            input_chunk_ids=input_chunk_ids,
            selected_chunk_ids=[hit.chunk_id for hit in selected],
            raw=raw,
            error=f"{type(exc).__name__}: {exc}",
        )
        return selected, result

    ordered: List["UnifiedSearchHit"] = []
    seen = set()
    for index in ranked_indices:
        hit = candidates[index - 1]
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        ordered.append(hit)
        if len(ordered) >= top_k:
            break
    for hit in candidates:
        if len(ordered) >= top_k:
            break
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        ordered.append(hit)

    result = LLMRerankResult(
        enabled=True,
        stage=stage,
        query=query,
        candidate_count=len(candidates),
        selected_count=len(ordered),
        ranked_indices=ranked_indices,
        input_chunk_ids=input_chunk_ids,
        selected_chunk_ids=[hit.chunk_id for hit in ordered],
        raw=raw,
    )
    return ordered, result
