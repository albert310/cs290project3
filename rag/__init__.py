"""Baseline RAG pipeline for Project 3."""

from .answer_verification import AnswerVerification, generate_answer_verification
from .dense_index import DenseVectorRAGIndex, EmbeddingAPIError, OpenAIEmbeddingClient, build_dense_index
from .llm_rerank import LLMRerankResult, rerank_hits_with_llm
from .pipeline import BaselineRAG, RAGAnswer, RAGConfig
from .query_keywords import QueryKeywordPlan, generate_query_keywords, parse_keywords
from .search_rollout import SearchRolloutDecision, SearchRolloutStep
from .text_index import TextChunk, TextFTSIndex, TextSearchHit
from .unified_index import UnifiedRAGIndex, UnifiedSearchHit
from .unified_pipeline import UnifiedRAG, UnifiedRAGConfig


def __getattr__(name: str):
    if name == "TantivyRAGIndex":
        from .tantivy_index import TantivyRAGIndex

        return TantivyRAGIndex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaselineRAG",
    "RAGAnswer",
    "RAGConfig",
    "AnswerVerification",
    "generate_answer_verification",
    "DenseVectorRAGIndex",
    "EmbeddingAPIError",
    "OpenAIEmbeddingClient",
    "build_dense_index",
    "LLMRerankResult",
    "rerank_hits_with_llm",
    "QueryKeywordPlan",
    "generate_query_keywords",
    "parse_keywords",
    "SearchRolloutDecision",
    "SearchRolloutStep",
    "TextChunk",
    "TantivyRAGIndex",
    "TextFTSIndex",
    "TextSearchHit",
    "UnifiedRAG",
    "UnifiedRAGConfig",
    "UnifiedRAGIndex",
    "UnifiedSearchHit",
]
