from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .unified_index import UnifiedRAGIndex, UnifiedSearchHit, _row_to_hit


DEFAULT_DENSE_INDEX_DIR = Path(".cache/dense_rag")
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding-4b"


class EmbeddingAPIError(RuntimeError):
    pass


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("QWEN_EMBEDDING_BASE_URL")
            or os.environ.get("EMBEDDING_BASE_URL")
            or DEFAULT_EMBEDDING_BASE_URL
        ).rstrip("/")
        self.model = model or os.environ.get("QWEN_EMBEDDING_MODEL") or os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        self.api_key = api_key or os.environ.get("QWEN_EMBEDDING_API_KEY") or os.environ.get("EMBEDDING_API_KEY")
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
        }
        body = self._post_json("/v1/embeddings", payload)
        data = body.get("data") or []
        if len(data) != len(texts):
            raise EmbeddingAPIError(f"embedding server returned {len(data)} vectors for {len(texts)} inputs")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in ordered]
        if any(not isinstance(vec, list) for vec in vectors):
            raise EmbeddingAPIError("embedding server response did not contain embedding arrays")
        return np.asarray(vectors, dtype=np.float32)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingAPIError(f"embedding server returned HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise EmbeddingAPIError(f"could not reach embedding server: {exc}") from exc


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32, copy=False)
    arr = vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def build_embedding_text(row: sqlite3.Row, *, max_chars: int) -> str:
    title = str(row["title"] or "").strip()
    url = str(row["url"] or "").strip()
    category = str(row["category"] or "").strip()
    text = str(row["text"] or "").strip()
    prefix_parts = [part for part in (title, category, url) if part]
    prefix = "\n".join(prefix_parts)
    combined = f"{prefix}\n{text}" if prefix else text
    return combined[:max_chars]


@dataclass(frozen=True)
class DenseIndexMetadata:
    db_path: str
    model: str
    embedding_base_url: str
    total: int
    dim: int
    dtype: str
    text_max_chars: int
    built_count: int
    created_at: float
    updated_at: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DenseIndexMetadata":
        return cls(
            db_path=str(data.get("db_path") or ""),
            model=str(data.get("model") or ""),
            embedding_base_url=str(data.get("embedding_base_url") or ""),
            total=int(data.get("total") or 0),
            dim=int(data.get("dim") or 0),
            dtype=str(data.get("dtype") or "float16"),
            text_max_chars=int(data.get("text_max_chars") or 1800),
            built_count=int(data.get("built_count") or 0),
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_path": self.db_path,
            "model": self.model,
            "embedding_base_url": self.embedding_base_url,
            "total": self.total,
            "dim": self.dim,
            "dtype": self.dtype,
            "text_max_chars": self.text_max_chars,
            "built_count": self.built_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class DenseVectorRAGIndex:
    def __init__(
        self,
        unified: UnifiedRAGIndex,
        *,
        index_dir: Path = DEFAULT_DENSE_INDEX_DIR,
        embedding_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        block_size: int = 8192,
        timeout: float = 120.0,
    ) -> None:
        self.unified = unified
        self.index_dir = Path(index_dir)
        self.client = OpenAIEmbeddingClient(base_url=embedding_base_url, model=embedding_model, timeout=timeout)
        self.block_size = max(512, int(block_size))
        self.metadata: Optional[DenseIndexMetadata] = None
        self.chunk_ids: Optional[np.ndarray] = None
        self.embeddings: Optional[np.ndarray] = None

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "metadata.json"

    @property
    def ids_path(self) -> Path:
        return self.index_dir / "chunk_ids.npy"

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / "embeddings.npy"

    def exists(self) -> bool:
        return self.metadata_path.exists() and self.ids_path.exists() and self.embeddings_path.exists()

    def open(self) -> "DenseVectorRAGIndex":
        if not self.exists():
            raise FileNotFoundError(f"dense index is missing under {self.index_dir}")
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = DenseIndexMetadata.from_dict(json.load(handle))
        if self.metadata.built_count < self.metadata.total:
            raise RuntimeError(
                f"dense index is incomplete: {self.metadata.built_count}/{self.metadata.total} vectors built"
            )
        self.chunk_ids = np.load(self.ids_path, mmap_mode="r")
        self.embeddings = np.load(self.embeddings_path, mmap_mode="r")
        return self

    def close(self) -> None:
        self.metadata = None
        self.chunk_ids = None
        self.embeddings = None

    def search(self, query: str, *, top_k: int = 64) -> List[UnifiedSearchHit]:
        if not query.strip() or self.embeddings is None or self.chunk_ids is None:
            return []
        query_vec = normalize_vectors(self.client.embed([query]))[0]
        top_k = max(1, min(int(top_k), int(self.chunk_ids.shape[0])))
        best_positions: List[np.ndarray] = []
        best_scores: List[np.ndarray] = []
        keep_per_block = min(max(top_k * 2, 32), self.block_size)

        for start in range(0, int(self.chunk_ids.shape[0]), self.block_size):
            end = min(start + self.block_size, int(self.chunk_ids.shape[0]))
            block = np.asarray(self.embeddings[start:end], dtype=np.float32)
            scores = block.dot(query_vec)
            if scores.size > keep_per_block:
                local = np.argpartition(scores, -keep_per_block)[-keep_per_block:]
            else:
                local = np.arange(scores.size)
            best_positions.append(local + start)
            best_scores.append(scores[local])

        positions = np.concatenate(best_positions)
        scores = np.concatenate(best_scores)
        if scores.size > top_k:
            selected = np.argpartition(scores, -top_k)[-top_k:]
        else:
            selected = np.arange(scores.size)
        selected_positions = positions[selected]
        selected_scores = scores[selected]
        order = np.argsort(-selected_scores)
        chunk_ids = [int(self.chunk_ids[int(selected_positions[index])]) for index in order]
        score_by_id = {chunk_id: float(selected_scores[index]) for chunk_id, index in zip(chunk_ids, order)}
        rows = self._fetch_rows(chunk_ids)
        hits: List[UnifiedSearchHit] = []
        for chunk_id in chunk_ids:
            row = rows.get(chunk_id)
            if row is None:
                continue
            # Keep dense scores in the hit for diagnostics. Cross-channel
            # fusion uses list order rather than absolute score magnitude.
            hits.append(_row_to_hit(row, query, score_by_id.get(chunk_id, 0.0)))
        return hits

    def _fetch_rows(self, chunk_ids: Sequence[int]) -> Dict[int, sqlite3.Row]:
        assert self.unified.conn is not None
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.unified.conn.execute(f"select * from chunks where id in ({placeholders})", tuple(chunk_ids)).fetchall()
        return {int(row["id"]): row for row in rows}


def build_dense_index(
    *,
    db_path: Path,
    index_dir: Path = DEFAULT_DENSE_INDEX_DIR,
    embedding_base_url: Optional[str] = None,
    embedding_model: Optional[str] = None,
    batch_size: int = 16,
    text_max_chars: int = 1800,
    resume: bool = True,
    timeout: float = 120.0,
) -> DenseIndexMetadata:
    db_path = Path(db_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAIEmbeddingClient(base_url=embedding_base_url, model=embedding_model, timeout=timeout)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    total = int(conn.execute("select count(*) from chunks").fetchone()[0])
    metadata_path = index_dir / "metadata.json"
    ids_path = index_dir / "chunk_ids.npy"
    embeddings_path = index_dir / "embeddings.npy"

    metadata: Optional[DenseIndexMetadata] = None
    if resume and metadata_path.exists() and ids_path.exists() and embeddings_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = DenseIndexMetadata.from_dict(json.load(handle))
        if metadata.total != total or metadata.text_max_chars != text_max_chars:
            metadata = None

    offset = metadata.built_count if metadata else 0
    if metadata is None:
        first_rows = conn.execute("select * from chunks order by id limit ?", (max(1, batch_size),)).fetchall()
        if not first_rows:
            raise RuntimeError("knowledge database contains no chunks")
        first_vectors = normalize_vectors(client.embed([build_embedding_text(row, max_chars=text_max_chars) for row in first_rows]))
        dim = int(first_vectors.shape[1])
        chunk_ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=np.int64, shape=(total,))
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="w+", dtype=np.float16, shape=(total, dim))
        now = time.time()
        metadata = DenseIndexMetadata(
            db_path=str(db_path),
            model=client.model,
            embedding_base_url=client.base_url,
            total=total,
            dim=dim,
            dtype="float16",
            text_max_chars=text_max_chars,
            built_count=0,
            created_at=now,
            updated_at=now,
        )
        _write_metadata(metadata_path, metadata)
        _write_batch(chunk_ids, embeddings, first_rows, first_vectors, start=0)
        offset = len(first_rows)
        metadata = _replace_metadata(metadata, built_count=offset)
        _write_metadata(metadata_path, metadata)
    else:
        chunk_ids = np.lib.format.open_memmap(ids_path, mode="r+", dtype=np.int64, shape=(metadata.total,))
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="r+",
            dtype=np.float16,
            shape=(metadata.total, metadata.dim),
        )

    while offset < total:
        rows = conn.execute("select * from chunks order by id limit ? offset ?", (batch_size, offset)).fetchall()
        if not rows:
            break
        vectors = normalize_vectors(client.embed([build_embedding_text(row, max_chars=text_max_chars) for row in rows]))
        _write_batch(chunk_ids, embeddings, rows, vectors, start=offset)
        offset += len(rows)
        metadata = _replace_metadata(metadata, built_count=offset)
        _write_metadata(metadata_path, metadata)
        print(f"dense index progress: {offset}/{total}", flush=True)

    metadata = _replace_metadata(metadata, built_count=offset)
    _write_metadata(metadata_path, metadata)
    return metadata


def _write_batch(
    chunk_ids: np.ndarray,
    embeddings: np.ndarray,
    rows: Sequence[sqlite3.Row],
    vectors: np.ndarray,
    *,
    start: int,
) -> None:
    end = start + len(rows)
    chunk_ids[start:end] = [int(row["id"]) for row in rows]
    embeddings[start:end] = vectors.astype(np.float16)
    chunk_ids.flush()
    embeddings.flush()


def _replace_metadata(metadata: DenseIndexMetadata, *, built_count: int) -> DenseIndexMetadata:
    return DenseIndexMetadata(
        db_path=metadata.db_path,
        model=metadata.model,
        embedding_base_url=metadata.embedding_base_url,
        total=metadata.total,
        dim=metadata.dim,
        dtype=metadata.dtype,
        text_max_chars=metadata.text_max_chars,
        built_count=built_count,
        created_at=metadata.created_at,
        updated_at=time.time(),
    )


def _write_metadata(path: Path, metadata: DenseIndexMetadata) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
