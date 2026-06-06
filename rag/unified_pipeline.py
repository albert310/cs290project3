from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from qwen_api import QwenClient

from .answer_verification import AnswerVerification, generate_answer_verification
from .pipeline import RAGAnswer, trim_text
from .query_keywords import QueryKeywordPlan, generate_query_keywords
from .search_rollout import (
    SearchRolloutDecision,
    SearchRolloutStep,
    decide_next_search,
    make_rollout_search_query,
)
from .unified_index import UnifiedRAGIndex, UnifiedSearchHit


DEFAULT_TANTIVY_INDEX_DIR = Path(".cache/tantivy_rag")
COURSE_CODE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,}\d{2,4}[A-Z]?(?![A-Za-z0-9])", flags=re.IGNORECASE)
ENGLISH_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")
CJK_NAME_RE = re.compile(r"[\u4e00-\u9fff]{2,4}")

PERSON_ATTR_TERMS = (
    "博士",
    "phd",
    "毕业院校",
    "毕业年份",
    "教育背景",
    "邮箱",
    "email",
    "办公室",
    "office",
    "电话",
    "研究方向",
    "研究兴趣",
    "research",
    "profile",
    "homepage",
    "个人主页",
)

NON_PERSON_CJK_TERMS = {
    "博士学位",
    "毕业院校",
    "毕业年份",
    "教育背景",
    "任课教师",
    "当前资料",
    "官方信息",
    "课程代码",
    "上海科技",
    "上海科技大学",
    "信息科学",
    "技术学院",
    "信息科学与技术学院",
}

NON_PERSON_CJK_SUBSTRINGS = (
    "博士",
    "毕业",
    "课程",
    "资料",
    "信息",
    "学院",
    "大学",
    "任课",
    "教师",
    "老师",
    "春季",
    "秋季",
    "学位",
    "年份",
    "院校",
    "官方",
    "无法",
    "确认",
    "哪一",
    "哪一年",
    "关于",
    "来自",
    "相关",
    "找到",
)

NON_PERSON_ENGLISH_TERMS = {
    "Machine Learning",
    "Lehigh University",
    "ShanghaiTech University",
    "School of Information",
    "Information Science",
}

NON_NAME_ENGLISH_TOKENS = {
    "Dr",
    "Ph",
    "PhD",
    "Professor",
    "Prof",
    "University",
    "School",
    "Department",
    "Faculty",
}

CONTEXT_ALIASES = {
    "美国理海大学": "美国理海大学 (Lehigh University)",
    "自然语言处理": "自然语言处理 (natural language processing)",
    "机器学习": "机器学习 (machine learning)",
    "人工智能": "人工智能 (artificial intelligence)",
    "大语言模型": "大语言模型 (large language models, LLM, 大型语言模型)",
}


@dataclass
class UnifiedRAGConfig:
    db_path: Path = Path("data/rag/knowledge.sqlite")
    retrieval_backend: str = "sqlite"
    tantivy_index_dir: Path = DEFAULT_TANTIVY_INDEX_DIR
    tantivy_candidates: int = 240
    top_k: int = 8
    max_context_chars: Optional[int] = 7200
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    enable_thinking: bool = True
    enable_llm_query_keywords: bool = True
    query_keyword_max_tokens: Optional[int] = 256
    query_keyword_max_terms: int = 12
    query_keyword_enable_thinking: bool = False
    enable_iterative_search: bool = False
    max_search_steps: int = 5
    rollout_decision_max_tokens: Optional[int] = 512
    rollout_decision_enable_thinking: bool = False
    rollout_hits_per_step: int = 5
    enable_answer_verification: bool = False
    verification_keyword_max_tokens: Optional[int] = 256
    verification_keyword_max_terms: int = 10
    verification_keyword_enable_thinking: bool = False
    verification_hits: int = 6


def build_unified_context(hits: List[UnifiedSearchHit], *, max_context_chars: Optional[int]) -> str:
    parts: List[str] = []
    used = 0
    has_limit = max_context_chars is not None and max_context_chars > 0
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
        if has_limit:
            budget = max_context_chars - used - len(header)
            if budget <= 0:
                break
            body = trim_text(enrich_context_aliases(hit.text.strip()), max(260, budget))
        else:
            body = enrich_context_aliases(hit.text.strip())
        part = header + body
        parts.append(part)
        used += len(part) + 2
        if has_limit and used >= max_context_chars:
            break
    return "\n\n".join(parts)


def enrich_context_aliases(text: str) -> str:
    for source, replacement in CONTEXT_ALIASES.items():
        text = text.replace(source, replacement)
    return text


def build_unified_prompt(query: str, hits: List[UnifiedSearchHit], *, max_context_chars: Optional[int]) -> str:
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
6. 最终答案必须复述用户问题中的关键限定条件，例如课程号、学期/年份、教师姓名/英文名，再给出答案。
7. 回答要简洁，优先使用中文。不要输出思考过程。
8. 如能回答，请在末尾列出使用的来源编号，例如“来源：[1][3]”。

【检索资料】
{context}

【用户问题】
{query}

【回答】
    """


def build_verified_prompt(
    query: str,
    draft_answer: str,
    original_hits: List[UnifiedSearchHit],
    verified_hits: List[UnifiedSearchHit],
    *,
    max_context_chars: Optional[int],
) -> str:
    all_hits = verified_hits + original_hits
    context = build_unified_context(all_hits, max_context_chars=max_context_chars)
    if not context:
        context = "（没有检索到可用资料）"
    return f"""你是上海科技大学和信息科学与技术学院（SIST）的问答助手。现在需要对一版草稿答案做二次核验后再给最终答案。

请严格遵守：
1. 只能根据【原始检索资料】和【二次核验资料】回答，不要使用资料外知识，不要编造。
2. 必须交叉比对草稿答案中的关键实体、课程号、教师姓名、年份、邮箱、学校名、数值和结论。
3. 如果二次核验资料与草稿答案冲突，优先采用更直接、更官方、更具体的资料；如果无法确认，第一句必须回答“根据当前资料无法确认。”。
4. 如果资料没有直接支持答案，必须拒答，并说明缺少什么证据；拒答时最后必须说明“不能编造”。
5. 对英文专名、课程号、邮箱、会议名、学校名和关键术语，必须保留资料中的英文原文；如需要，可以同时补充中文说明，不能只写中文译名。
6. 最终答案必须复述用户问题中的关键限定条件，例如课程号、学期/年份、教师姓名/英文名，再给出核验后的答案。
7. 回答要简洁，优先使用中文。不要输出思考过程。
8. 如能回答，请在末尾列出使用的来源编号，例如“来源：[1][3]”。

【待核验草稿答案】
{draft_answer}

【原始检索资料 + 二次核验资料】
{context}

【用户问题】
{query}

【二次核验后的最终回答】
"""


def strip_course_codes(text: str) -> str:
    return re.sub(r"\s+", " ", COURSE_CODE_TOKEN_RE.sub(" ", text)).strip()


def verification_needs_profile_search(query: str, draft_answer: str, verification: AnswerVerification) -> bool:
    haystack = " ".join([query, draft_answer, verification.search_query, *verification.keywords]).lower()
    return any(term.lower() in haystack for term in PERSON_ATTR_TERMS)


def _add_unique(values: List[str], value: str) -> None:
    value = re.sub(r"\s+", " ", value).strip(" ,，;；、。.!！?？:：\"'`[]{}()（）<>《》")
    if value and value.lower() not in {item.lower() for item in values}:
        values.append(value)


def normalize_english_profile_name(value: str) -> str:
    tokens = [token for token in value.split() if token not in NON_NAME_ENGLISH_TOKENS]
    if len(tokens) < 2:
        return ""
    return " ".join(tokens[:3])


def extract_profile_names(query: str, draft_answer: str, verification: AnswerVerification) -> List[str]:
    texts = list(verification.keywords)
    names: List[str] = []
    for text in texts + [verification.search_query, query, draft_answer]:
        for match in CJK_NAME_RE.finditer(text):
            value = match.group(0)
            if value in NON_PERSON_CJK_TERMS:
                continue
            if any(term in value for term in NON_PERSON_CJK_SUBSTRINGS):
                continue
            if len(value) > 3:
                continue
            _add_unique(names, value)
        for match in ENGLISH_NAME_RE.finditer(text):
            value = normalize_english_profile_name(match.group(0))
            if value in NON_PERSON_ENGLISH_TERMS or "University" in value or "School" in value:
                continue
            _add_unique(names, value)
        if names and text in verification.keywords:
            continue
        if names and text not in verification.keywords:
            break
    return names[:4]


def make_profile_verification_queries(query: str, draft_answer: str, verification: AnswerVerification) -> List[str]:
    if not verification_needs_profile_search(query, draft_answer, verification):
        return []
    names = extract_profile_names(query, draft_answer, verification)
    if not names:
        return []

    haystack = " ".join([query, draft_answer, verification.search_query, *verification.keywords]).lower()
    attrs = ["profile", "homepage", "faculty", "个人主页"]
    if any(term in haystack for term in ("博士", "phd", "毕业院校", "毕业年份", "教育背景")):
        attrs.extend(["教育背景", "博士学位", "PhD", "university", "year"])
    if any(term in haystack for term in ("邮箱", "email", "办公室", "office", "电话")):
        attrs.extend(["邮箱", "email", "contact", "办公室", "office"])
    if any(term in haystack for term in ("研究方向", "研究兴趣", "research")):
        attrs.extend(["研究方向", "research interests"])
    return [" ".join(names + attrs)]


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


def chat_visible_answer(result: Any, query: str) -> str:
    answer = result.answer.strip()
    if answer:
        return answer
    if result.think.strip():
        return f"根据当前资料无法确认。模型输出在思考阶段被截断，未形成对“{query}”的可核验答案，不能编造。"
    return result.raw.strip() or f"根据当前资料无法确认。未生成对“{query}”的可用答案，不能编造。"


class UnifiedRAG:
    def __init__(self, config: Optional[UnifiedRAGConfig] = None, client: Optional[QwenClient] = None) -> None:
        self.config = config or UnifiedRAGConfig()
        self.client = client or QwenClient()
        self.index = UnifiedRAGIndex(self.config.db_path)
        self.tantivy_index: Any | None = None

    def open(self) -> "UnifiedRAG":
        if self.config.retrieval_backend == "tantivy":
            from .tantivy_index import TantivyRAGIndex

            self.tantivy_index = TantivyRAGIndex(
                self.config.db_path,
                self.config.tantivy_index_dir,
                candidate_limit=self.config.tantivy_candidates,
            ).open()
        else:
            self.index.open()
        return self

    def close(self) -> None:
        self.index.close()
        if self.tantivy_index is not None:
            self.tantivy_index.close()
            self.tantivy_index = None

    def _search_backend(self, query: str, *, top_k: int) -> List[UnifiedSearchHit]:
        if self.config.retrieval_backend == "tantivy":
            if self.tantivy_index is None:
                raise RuntimeError("Tantivy retriever is not open.")
            return self.tantivy_index.search(query, top_k=top_k)
        return self.index.search(query, top_k=top_k)

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
                search_query="",
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
        if not search_query.strip():
            return []
        return self._search_backend(search_query, top_k=target_k)

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

    def run_answer_verification(
        self,
        query: str,
        draft_answer: str,
        hits: List[UnifiedSearchHit],
        *,
        top_k: Optional[int] = None,
    ) -> tuple[AnswerVerification, List[UnifiedSearchHit]]:
        if not self.config.enable_answer_verification:
            return AnswerVerification(enabled=False), []

        try:
            verification = generate_answer_verification(
                self.client,
                query=query,
                answer=draft_answer,
                max_keywords=self.config.verification_keyword_max_terms,
                max_tokens=self.config.verification_keyword_max_tokens,
                temperature=0.0,
                enable_thinking=self.config.verification_keyword_enable_thinking,
            )
        except Exception as exc:
            return (
                AnswerVerification(
                    enabled=True,
                    error=f"{type(exc).__name__}: {exc}",
                    draft_answer=draft_answer,
                ),
                [],
            )

        target_k = max(top_k or self.config.top_k, self.config.verification_hits)
        candidates: List[UnifiedSearchHit] = []
        search_queries = make_profile_verification_queries(query, draft_answer, verification)
        search_queries.append(verification.search_query or query)
        for search_query in list(search_queries):
            no_course_query = strip_course_codes(search_query)
            if no_course_query and no_course_query.lower() not in {item.lower() for item in search_queries}:
                search_queries.append(no_course_query)

        seen_queries = set()
        actual_search_queries: List[str] = []
        for search_query in search_queries:
            key = search_query.lower()
            if not search_query or key in seen_queries:
                continue
            seen_queries.add(key)
            actual_search_queries.append(search_query)
            candidates.extend(self._search_backend(search_query, top_k=target_k))

        _, added = self._append_new_hits(hits, candidates, limit=self.config.verification_hits)
        verification = AnswerVerification(
            enabled=True,
            keywords=verification.keywords,
            search_query=verification.search_query,
            search_queries=actual_search_queries,
            raw=verification.raw,
            error=verification.error,
            hit_count=len(candidates),
            new_hit_count=added,
            draft_answer=draft_answer,
        )

        seen = {hit.chunk_id for hit in hits}
        verification_hits: List[UnifiedSearchHit] = []
        for hit in candidates:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            verification_hits.append(hit)
            if len(verification_hits) >= self.config.verification_hits:
                break
        return verification, verification_hits

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
            new_hits = self._search_backend(search_query, top_k=max(target_k, self.config.rollout_hits_per_step))
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
            draft_answer = f"根据当前资料无法确认。未找到与“{query}”直接相关的官方信息，不能编造答案。"
            result_raw = ""
            result_think = ""
            usage = None
        else:
            result = self.client.chat(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                # Long retrieval contexts make Qwen's internal thinking spend the
                # whole generation budget before it reaches the final answer.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            draft_answer = chat_visible_answer(result, query)
            result_raw = result.raw
            result_think = result.think
            usage = result.usage

        answer = draft_answer
        answer_verification, verification_hits = self.run_answer_verification(
            query,
            draft_answer,
            hits,
            top_k=top_k,
        )
        final_hits = verification_hits + hits if verification_hits else hits
        if answer_verification.enabled and not answer_verification.error:
            verified_prompt = build_verified_prompt(
                query,
                draft_answer,
                hits,
                verification_hits,
                max_context_chars=self.config.max_context_chars,
            )
            verified_result = self.client.chat(
                verified_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                # The verified prompt already contains the draft and cross-check evidence;
                # Qwen's internal thinking can loop on this long, repetitive context.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            answer = chat_visible_answer(verified_result, query)
            result_raw = verified_result.raw
            result_think = verified_result.think
            usage = verified_result.usage
            prompt = verified_prompt

        return RAGAnswer(
            query=query,
            answer=answer,
            raw=result_raw,
            think=result_think,
            hits=final_hits,
            prompt=prompt,
            usage=usage,
            search_query=query_plan.search_query,
            query_keywords=query_plan.keywords,
            query_keyword_raw=query_plan.raw,
            query_keyword_error=query_plan.error,
            search_rollout=[step.to_dict() for step in rollout_steps],
            answer_verification=answer_verification.to_dict(),
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
            draft_answer = f"根据当前资料无法确认。未找到与“{query}”直接相关的官方信息，不能编造答案。"
        elif self.config.enable_answer_verification:
            draft_result = self.client.chat(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            draft_answer = chat_visible_answer(draft_result, query)
        else:
            draft_answer = ""

        if self.config.enable_answer_verification:
            answer_verification, verification_hits = self.run_answer_verification(
                query,
                draft_answer,
                hits,
                top_k=top_k,
            )
            yield {"event": "answer_verification", **answer_verification.to_dict()}
            if answer_verification.error:
                yield {"event": "answer_delta", "delta": draft_answer}
                yield {"event": "done", "usage": None}
                return
            else:
                original_hits = hits
                if verification_hits:
                    hits = verification_hits + hits
                    yield {"event": "sources", "hits": [hit.to_dict() for hit in hits]}
                prompt = build_verified_prompt(
                    query,
                    draft_answer,
                    original_hits,
                    verification_hits,
                    max_context_chars=self.config.max_context_chars,
                )

        if not hits and not self.config.enable_answer_verification:
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
            # Long retrieval contexts make Qwen's internal thinking spend the
            # whole generation budget before it reaches the final answer.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ):
            if event.kind == "think" and event.delta:
                yield {"event": "think_delta", "delta": event.delta}
            elif event.kind == "answer" and event.delta:
                yield {"event": "answer_delta", "delta": event.delta}
            elif event.kind == "done":
                yield {"event": "done", "usage": event.usage, "finish_reason": event.finish_reason}
