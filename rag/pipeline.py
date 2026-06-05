from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from qwen_api import QwenClient

from .text_index import TextFTSIndex, TextSearchHit


@dataclass
class RAGConfig:
    texts_dir: Path = Path("data/sist/texts")
    cache_path: Path = Path(".cache/rag_baseline_texts.sqlite")
    top_k: int = 6
    chunk_chars: int = 900
    overlap_chars: int = 120
    min_chars: int = 80
    max_context_chars: int = 5600
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    enable_thinking: bool = True


@dataclass
class RAGAnswer:
    query: str
    answer: str
    raw: str
    think: str = ""
    hits: List[TextSearchHit] = field(default_factory=list)
    prompt: str = ""
    usage: Optional[Dict[str, Any]] = None
    search_query: str = ""
    query_keywords: List[str] = field(default_factory=list)
    query_keyword_raw: str = ""
    query_keyword_error: str = ""
    search_rollout: List[Dict[str, Any]] = field(default_factory=list)
    answer_verification: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "raw": self.raw,
            "think": self.think,
            "hits": [hit.to_dict() for hit in self.hits],
            "usage": self.usage,
            "search_query": self.search_query,
            "query_keywords": self.query_keywords,
            "query_keyword_raw": self.query_keyword_raw,
            "query_keyword_error": self.query_keyword_error,
            "search_rollout": self.search_rollout,
            "answer_verification": self.answer_verification,
        }


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_context(hits: List[TextSearchHit], *, max_context_chars: int) -> str:
    parts: List[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        header = f"[{index}] source={hit.source_id}\n"
        budget = max_context_chars - used - len(header)
        if budget <= 0:
            break
        body = trim_text(hit.text.strip(), max(200, budget))
        part = header + body
        parts.append(part)
        used += len(part) + 2
        if used >= max_context_chars:
            break
    return "\n\n".join(parts)


def build_prompt(query: str, hits: List[TextSearchHit], *, max_context_chars: int) -> str:
    context = build_context(hits, max_context_chars=max_context_chars)
    if not context:
        context = "（没有检索到可用资料）"
    return f"""你是上海科技大学和信息科学与技术学院（SIST）的问答助手。

请严格遵守：
1. 只能根据【检索资料】回答，不要使用资料外知识，不要编造。
2. 如果资料没有直接支持答案，第一句必须回答“根据当前资料无法确认。”，然后简短说明缺少什么证据。
3. 如果用户问题中的学院、课程、专业、研究中心、教师或奖项在资料中没有直接出现，应拒绝编造。
4. 回答要简洁，优先使用中文。不要输出思考过程。
5. 如能回答，请在末尾列出使用的来源编号，例如“来源：[1][3]”。

【检索资料】
{context}

【用户问题】
{query}

【回答】
"""


class BaselineRAG:
    def __init__(self, config: Optional[RAGConfig] = None, client: Optional[QwenClient] = None) -> None:
        self.config = config or RAGConfig()
        self.client = client or QwenClient()
        self.index = TextFTSIndex(
            texts_dir=self.config.texts_dir,
            cache_path=self.config.cache_path,
            chunk_chars=self.config.chunk_chars,
            overlap_chars=self.config.overlap_chars,
            min_chars=self.config.min_chars,
        )

    def open(self, *, rebuild_index: bool = False) -> "BaselineRAG":
        self.index.open(rebuild=rebuild_index)
        return self

    def close(self) -> None:
        self.index.close()

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> List[TextSearchHit]:
        return self.index.search(query, top_k=top_k or self.config.top_k)

    def build_prompt(self, query: str, hits: List[TextSearchHit]) -> str:
        return build_prompt(query, hits, max_context_chars=self.config.max_context_chars)

    def answer(self, query: str, *, top_k: Optional[int] = None) -> RAGAnswer:
        hits = self.retrieve(query, top_k=top_k)
        prompt = self.build_prompt(query, hits)
        if not hits:
            return RAGAnswer(
                query=query,
                answer="根据当前资料无法确认。没有检索到可用资料，不能编造答案。",
                raw="",
                think="",
                hits=[],
                prompt=prompt,
            )

        result = self.client.chat(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.config.enable_thinking}},
        )
        answer = result.answer.strip() or result.raw.strip()
        return RAGAnswer(
            query=query,
            answer=answer,
            raw=result.raw,
            think=result.think,
            hits=hits,
            prompt=prompt,
            usage=result.usage,
        )

    def stream(self, query: str, *, top_k: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        hits = self.retrieve(query, top_k=top_k)
        prompt = self.build_prompt(query, hits)
        yield {"event": "sources", "hits": [hit.to_dict() for hit in hits]}

        if not hits:
            yield {
                "event": "answer_delta",
                "delta": "根据当前资料无法确认。没有检索到可用资料，不能编造答案。",
            }
            yield {"event": "done", "usage": None}
            return

        for event in self.client.stream_chat(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.config.enable_thinking}},
        ):
            if event.kind == "think" and event.delta:
                yield {"event": "think_delta", "delta": event.delta}
            elif event.kind == "answer" and event.delta:
                yield {"event": "answer_delta", "delta": event.delta}
            elif event.kind == "done":
                yield {"event": "done", "usage": event.usage, "finish_reason": event.finish_reason}
