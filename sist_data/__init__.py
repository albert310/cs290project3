"""Dataset helpers for the Project 3 SIST RAG data."""

__all__ = ["SISTDataset"]


def __getattr__(name):
    if name == "SISTDataset":
        from .loader import SISTDataset

        return SISTDataset
    raise AttributeError(name)
