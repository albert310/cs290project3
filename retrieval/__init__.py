"""Retrieval components for the Project 3 RAG system."""

__all__ = ["BM25FIndex", "SearchHit", "expand_query", "tokenize"]


def __getattr__(name):
    if name in __all__:
        from . import keyword_search

        return getattr(keyword_search, name)
    raise AttributeError(name)
