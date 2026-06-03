from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from qwen_api import QwenClient

from .pipeline import RAGAnswer, trim_text
from .query_keywords import QueryKeywordPlan, generate_query_keywords
from .search_rollout import (
    SearchRolloutDecision,
    SearchRolloutStep,
    decide_next_search,
    make_rollout_search_query,
)
from .unified_index import UnifiedRAGIndex, UnifiedSearchHit


@dataclass
class UnifiedRAGConfig:
    db_path: Path = Path("data/rag/knowledge.sqlite")
    top_k: int = 8
    max_context_chars: int = 7200
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    enable_thinking: bool = True
    enable_llm_query_keywords: bool = False
    query_keyword_max_tokens: Optional[int] = 256
    query_keyword_max_terms: int = 12
    query_keyword_enable_thinking: bool = False
    query_keyword_original_boost: float = 12.0
    enable_iterative_search: bool = False
    max_search_steps: int = 5
    rollout_decision_max_tokens: Optional[int] = 512
    rollout_decision_enable_thinking: bool = False
    rollout_hits_per_step: int = 5


def build_unified_context(hits: List[UnifiedSearchHit], *, max_context_chars: int) -> str:
    parts: List[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        meta = []
        if hit.title:
            meta.append(f"title={hit.title}")
        meta.append(f"type={hit.source_type}")
        if hit.category:
            meta.append(f"category={hit.category}")
        if hit.date:
            meta.append(f"date={hit.date}")
        if hit.url:
            meta.append(f"url={hit.url}")
        meta.append(f"path={hit.path}")
        header = f"[{index}] " + " | ".join(meta) + "\n"
        budget = max_context_chars - used - len(header)
        if budget <= 0:
            break
        body = trim_text(hit.text.strip(), max(260, budget))
        part = header + body
        parts.append(part)
        used += len(part) + 2
        if used >= max_context_chars:
            break
    return "\n\n".join(parts)


def build_unified_prompt(query: str, hits: List[UnifiedSearchHit], *, max_context_chars: int) -> str:
    context = build_unified_context(hits, max_context_chars=max_context_chars)
    if not context:
        context = "（没有检索到可用资料）"
    return f"""你是上海科技大学和信息科学与技术学院（SIST）的问答助手。

请严格遵守：
1. 只能根据【检索资料】回答，不要使用资料外知识，不要编造。
2. 如果资料没有直接支持答案，第一句必须回答“根据当前资料无法确认。”。
3. 拒答时必须复述问题中的关键实体或课程代码，并说明“未找到/不存在/无官方信息”，最后说明“不能编造”。
4. 如果检索资料只出现相似词、泛词、旧年份或无关上下文，不能当作证据。
5. 课程代码、教师姓名、机构名称、年份和数值必须以检索资料为准。
6. 回答要简洁，优先使用中文。不要输出思考过程。
7. 如能回答，请在末尾列出使用的来源编号，例如“来源：[1][3]”。

【检索资料】
{context}

【用户问题】
{query}

【回答】
    """


def build_rollout_evidence(hits: List[UnifiedSearchHit], *, max_chars: int = 5200) -> str:
    parts: List[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        header_items = [f"[{index}]", f"title={hit.title or hit.category}", f"type={hit.source_type}"]
        if hit.date:
            header_items.append(f"date={hit.date}")
        if hit.url:
            header_items.append(f"url={hit.url}")
        header = " | ".join(header_items) + "\n"
        body = trim_text(hit.text.strip(), 520)
        part = header + body
        if used + len(part) > max_chars:
            break
        parts.append(part)
        used += len(part) + 2
    return "\n\n".join(parts) or "（没有检索到可用资料）"


class UnifiedRAG:
    def __init__(self, config: Optional[UnifiedRAGConfig] = None, client: Optional[QwenClient] = None) -> None:
        self.config = config or UnifiedRAGConfig()
        self.client = client or QwenClient()
        self.index = UnifiedRAGIndex(self.config.db_path)

    def open(self) -> "UnifiedRAG":
        self.index.open()
        return self

    def close(self) -> None:
        self.index.close()

    def plan_query(self, query: str) -> QueryKeywordPlan:
        if not self.config.enable_llm_query_keywords:
            return QueryKeywordPlan(enabled=False, search_query=query)
        try:
            return generate_query_keywords(
                self.client,
                query,
                max_keywords=self.config.query_keyword_max_terms,
                max_tokens=self.config.query_keyword_max_tokens,
                temperature=0.0,
                enable_thinking=self.config.query_keyword_enable_thinking,
            )
        except Exception as exc:
            return QueryKeywordPlan(
                enabled=True,
                search_query=query,
                error=f"{type(exc).__name__}: {exc}",
            )

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        search_query: Optional[str] = None,
    ) -> List[UnifiedSearchHit]:
        target_k = top_k or self.config.top_k
        if search_query is None:
            search_query = self.plan_query(query).search_query
        if search_query == query:
            return self.index.search(query, top_k=target_k)

        original_hits = self.index.search(query, top_k=max(target_k * 2, target_k))
        keyword_hits = self.index.search(search_query, top_k=max(target_k * 2, target_k))
        return self._merge_retrieval_hits(original_hits, keyword_hits, top_k=target_k)

    def _merge_retrieval_hits(
        self,
        original_hits: List[UnifiedSearchHit],
        keyword_hits: List[UnifiedSearchHit],
        *,
        top_k: int,
    ) -> List[UnifiedSearchHit]:
        by_chunk: Dict[int, UnifiedSearchHit] = {}
        for hit in keyword_hits:
            by_chunk[hit.chunk_id] = hit
        for hit in original_hits:
            boosted = replace(hit, rank=hit.rank + self.config.query_keyword_original_boost)
            current = by_chunk.get(hit.chunk_id)
            if current is None or boosted.rank > current.rank:
                by_chunk[hit.chunk_id] = boosted
        hits = sorted(by_chunk.values(), key=lambda item: item.rank, reverse=True)
        return hits[:top_k]

    def _append_new_hits(
        self,
        current: List[UnifiedSearchHit],
        new_hits: List[UnifiedSearchHit],
        *,
        limit: Optional[int] = None,
    ) -> tuple[List[UnifiedSearchHit], int]:
        seen = {hit.chunk_id for hit in current}
        out = list(current)
        added = 0
        for hit in new_hits:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            out.append(hit)
            added += 1
            if limit is not None and added >= limit:
                break
        return out, added

    def run_search_rollout(
        self,
        query: str,
        hits: List[UnifiedSearchHit],
        *,
        top_k: Optional[int] = None,
    ) -> tuple[List[UnifiedSearchHit], List[SearchRolloutStep]]:
        if not self.config.enable_iterative_search:
            return hits, []

        target_k = top_k or self.config.top_k
        max_steps = max(0, min(self.config.max_search_steps, 5))
        rollout_steps: List[SearchRolloutStep] = []
        previous_searches: List[str] = []
        accumulated = list(hits)

        for step in range(1, max_steps + 1):
            evidence = build_rollout_evidence(accumulated)
            try:
                decision = decide_next_search(
                    self.client,
                    query=query,
                    evidence=evidence,
                    previous_searches=previous_searches,
                    step=step,
                    max_steps=max_steps,
                    max_tokens=self.config.rollout_decision_max_tokens,
                    temperature=0.0,
                    enable_thinking=self.config.rollout_decision_enable_thinking,
                )
            except Exception as exc:
                rollout_steps.append(
                    SearchRolloutStep(
                        step=step,
                        decision=SearchRolloutDecision(
                            action="answer",
                            error=f"{type(exc).__name__}: {exc}",
                        ),
                    )
                )
                break

            if not decision.wants_search:
                rollout_steps.append(SearchRolloutStep(step=step, decision=decision))
                break

            search_query = make_rollout_search_query(decision)
            if not search_query or search_query.lower() in {item.lower() for item in previous_searches}:
                rollout_steps.append(SearchRolloutStep(step=step, decision=decision, search_query=search_query))
                break

            previous_searches.append(search_query)
            new_hits = self.index.search(search_query, top_k=max(target_k, self.config.rollout_hits_per_step))
            accumulated, added = self._append_new_hits(
                accumulated,
                new_hits,
                limit=self.config.rollout_hits_per_step,
            )
            rollout_steps.append(
                SearchRolloutStep(
                    step=step,
                    decision=decision,
                    search_query=search_query,
                    hit_count=len(new_hits),
                    new_hit_count=added,
                )
            )
            if added == 0:
                break

        return accumulated, rollout_steps

    def build_prompt(self, query: str, hits: List[UnifiedSearchHit]) -> str:
        return build_unified_prompt(query, hits, max_context_chars=self.config.max_context_chars)

    def answer(self, query: str, *, top_k: Optional[int] = None) -> RAGAnswer:
        query_plan = self.plan_query(query)
        hits = self.retrieve(query, top_k=top_k, search_query=query_plan.search_query)
        hits, rollout_steps = self.run_search_rollout(query, hits, top_k=top_k)
        prompt = self.build_prompt(query, hits)
        if not hits:
            return RAGAnswer(
                query=query,
                answer=f"根据当前资料无法确认。未找到与“{query}”直接相关的官方信息，不能编造答案。",
                raw="",
                think="",
                hits=[],
                prompt=prompt,
                search_query=query_plan.search_query,
                query_keywords=query_plan.keywords,
                query_keyword_raw=query_plan.raw,
                query_keyword_error=query_plan.error,
                search_rollout=[step.to_dict() for step in rollout_steps],
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
            search_query=query_plan.search_query,
            query_keywords=query_plan.keywords,
            query_keyword_raw=query_plan.raw,
            query_keyword_error=query_plan.error,
            search_rollout=[step.to_dict() for step in rollout_steps],
        )

    def stream(self, query: str, *, top_k: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        query_plan = self.plan_query(query)
        if query_plan.enabled:
            yield {
                "event": "query_keywords",
                "keywords": query_plan.keywords,
                "search_query": query_plan.search_query,
                "raw": query_plan.raw,
                "error": query_plan.error,
            }
        hits = self.retrieve(query, top_k=top_k, search_query=query_plan.search_query)
        yield {"event": "sources", "hits": [hit.to_dict() for hit in hits]}
        hits, rollout_steps = self.run_search_rollout(query, hits, top_k=top_k)
        for step in rollout_steps:
            yield {"event": "search_rollout_step", **step.to_dict()}
        if rollout_steps:
            yield {"event": "sources", "hits": [hit.to_dict() for hit in hits]}
        prompt = self.build_prompt(query, hits)

        if not hits:
            yield {
                "event": "answer_delta",
                "delta": f"根据当前资料无法确认。未找到与“{query}”直接相关的官方信息，不能编造答案。",
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
