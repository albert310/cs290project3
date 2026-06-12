from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import mimetypes
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Sequence
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_api import QwenAPIError
from rag import BaselineRAG, RAGConfig, UnifiedRAG, UnifiedRAGConfig


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web" / "static"
RAG_INSTANCE: Any | None = None
RAG_LOCK = threading.Lock()
THINKING_DEFAULT_MAX_TOKENS = 8192


def find_port(start: int, host: str) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port found from {start}")


def json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def optional_bool(payload: Dict[str, Any], key: str) -> bool | None:
    if key not in payload:
        return None
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{key} must be boolean")


def optional_int(
    payload: Dict[str, Any],
    key: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    if key not in payload:
        return None
    value = payload.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be integer") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{key} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{key} must be <= {max_value}")
    return parsed


def request_max_tokens(rag: Any, enable_thinking: bool | None, max_tokens: int | None) -> int | None:
    if max_tokens is not None:
        return max_tokens
    config = getattr(rag, "config", None)
    thinking = enable_thinking if enable_thinking is not None else bool(getattr(config, "enable_thinking", False))
    if thinking:
        return THINKING_DEFAULT_MAX_TOKENS
    return None


@contextmanager
def temporary_config_overrides(rag: Any, **overrides: Any | None):
    config = getattr(rag, "config", None)
    original: Dict[str, Any] = {}
    if config is not None:
        for key, value in overrides.items():
            if value is None or not hasattr(config, key):
                continue
            original[key] = getattr(config, key)
            setattr(config, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(config, key, value)


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "SISTRAG/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"ok": True})
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            self.handle_chat()
            return
        if parsed.path == "/api/chat/stream":
            self.handle_chat_stream()
            return
        if parsed.path != "/api/chat":
            self.send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return

    def handle_chat(self) -> None:
        global RAG_INSTANCE
        if RAG_INSTANCE is None:
            self.send_json({"error": "RAG is not initialized"}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8")) if body else {}
            query = str(payload.get("query") or "").strip()
            top_k = int(payload.get("top_k") or RAG_INSTANCE.config.top_k)
            enable_thinking = optional_bool(payload, "enable_thinking")
            verify_answer = optional_bool(payload, "verify_answer")
            llm_rerank = optional_bool(payload, "llm_rerank")
            iterative_search = optional_bool(payload, "iterative_search")
            max_tokens = optional_int(payload, "max_tokens", min_value=128, max_value=8192)
        except Exception as exc:
            self.send_json({"error": f"invalid request: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        if not query:
            self.send_json({"error": "empty query"}, status=HTTPStatus.BAD_REQUEST)
            return

        started = time.perf_counter()
        try:
            effective_max_tokens = request_max_tokens(RAG_INSTANCE, enable_thinking, max_tokens)
            with RAG_LOCK:
                with temporary_config_overrides(
                    RAG_INSTANCE,
                    enable_thinking=enable_thinking,
                    enable_answer_verification=verify_answer,
                    enable_llm_rerank=llm_rerank,
                    enable_iterative_search=iterative_search,
                    max_tokens=effective_max_tokens,
                ):
                    result = RAG_INSTANCE.answer(query, top_k=top_k)
            latency = time.perf_counter() - started
            self.send_json(
                {
                    "answer": result.answer,
                    "think": result.think,
                    "query_keywords": result.query_keywords,
                    "search_query": result.search_query or result.query,
                    "query_keyword_error": result.query_keyword_error,
                    "llm_rerank": result.llm_rerank,
                    "search_rollout": result.search_rollout,
                    "answer_verification": result.answer_verification,
                    "hits": [hit.to_dict() for hit in result.hits],
                    "latency_sec": latency,
                    "usage": result.usage,
                    "enable_thinking": enable_thinking,
                    "verify_answer": verify_answer,
                    "enable_llm_rerank": (
                        llm_rerank if llm_rerank is not None else RAG_INSTANCE.config.enable_llm_rerank
                    ),
                    "enable_iterative_search": (
                        iterative_search
                        if iterative_search is not None
                        else RAG_INSTANCE.config.enable_iterative_search
                    ),
                    "max_tokens": (
                        effective_max_tokens
                        if effective_max_tokens is not None
                        else RAG_INSTANCE.config.max_tokens
                    ),
                }
            )
        except QwenAPIError as exc:
            status = HTTPStatus.BAD_GATEWAY if exc.status is not None else HTTPStatus.SERVICE_UNAVAILABLE
            self.send_json({"error": f"Qwen service error: {exc}"}, status=status)
        except Exception as exc:
            self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_chat_stream(self) -> None:
        global RAG_INSTANCE
        if RAG_INSTANCE is None:
            self.send_json({"error": "RAG is not initialized"}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8")) if body else {}
            query = str(payload.get("query") or "").strip()
            top_k = int(payload.get("top_k") or RAG_INSTANCE.config.top_k)
            enable_thinking = optional_bool(payload, "enable_thinking")
            verify_answer = optional_bool(payload, "verify_answer")
            llm_rerank = optional_bool(payload, "llm_rerank")
            iterative_search = optional_bool(payload, "iterative_search")
            max_tokens = optional_int(payload, "max_tokens", min_value=128, max_value=8192)
        except Exception as exc:
            self.send_json({"error": f"invalid request: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        if not query:
            self.send_json({"error": "empty query"}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        started = time.perf_counter()
        try:
            effective_max_tokens = request_max_tokens(RAG_INSTANCE, enable_thinking, max_tokens)
            with RAG_LOCK:
                with temporary_config_overrides(
                    RAG_INSTANCE,
                    enable_thinking=enable_thinking,
                    enable_answer_verification=verify_answer,
                    enable_llm_rerank=llm_rerank,
                    enable_iterative_search=iterative_search,
                    max_tokens=effective_max_tokens,
                ):
                    for event in RAG_INSTANCE.stream(query, top_k=top_k):
                        if event.get("event") == "done":
                            event["latency_sec"] = time.perf_counter() - started
                            event["max_tokens"] = (
                                effective_max_tokens
                                if effective_max_tokens is not None
                                else RAG_INSTANCE.config.max_tokens
                            )
                        self.send_sse(str(event.get("event") or "message"), event)
        except BrokenPipeError:
            return
        except QwenAPIError as exc:
            self.send_sse("error", {"event": "error", "error": f"Qwen service error: {exc}"})
        except Exception as exc:
            self.send_sse("error", {"event": "error", "error": f"{type(exc).__name__}: {exc}"})

    def serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            file_path = STATIC_DIR / "index.html"
        else:
            relative = path.lstrip("/")
            if relative.startswith("static/"):
                relative = relative[len("static/") :]
            file_path = STATIC_DIR / relative

        try:
            resolved = file_path.resolve()
            if not str(resolved).startswith(str(STATIC_DIR.resolve())) or not resolved.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = resolved.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_sse(self, event: str, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        message = f"event: {event}\ndata: {data}\n\n".encode("utf-8")
        self.wfile.write(message)
        self.wfile.flush()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the TypeScript web chat frontend and RAG API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--rag-mode", choices=("unified", "baseline"), default="unified")
    parser.add_argument("--db-path", default="data/rag/knowledge.sqlite")
    parser.add_argument("--retrieval-backend", choices=("sqlite", "tantivy"), default="sqlite")
    parser.add_argument("--tantivy-index-dir", default=".cache/tantivy_rag")
    parser.add_argument("--tantivy-candidates", type=int, default=240)
    parser.add_argument("--lexical-candidates", type=int, default=240)
    parser.add_argument("--structured-candidates", type=int, default=160)
    parser.add_argument("--dense-retrieval", dest="dense_retrieval", action="store_true")
    parser.add_argument("--no-dense-retrieval", dest="dense_retrieval", action="store_false")
    parser.set_defaults(dense_retrieval=False)
    parser.add_argument("--dense-index-dir", default=".cache/dense_rag")
    parser.add_argument("--dense-candidates", type=int, default=160)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--embedding-model", default="qwen3-embedding-4b")
    parser.add_argument("--lexical-rrf-weight", type=float, default=1.0)
    parser.add_argument("--structured-rrf-weight", type=float, default=1.15)
    parser.add_argument("--dense-rrf-weight", type=float, default=1.0)
    parser.add_argument("--hybrid-rrf-k", type=int, default=60)
    parser.add_argument("--neighbor-expansion-window", type=int, default=1)
    parser.add_argument("--neighbor-expansion-limit", type=int, default=48)
    parser.add_argument("--texts-dir", default="data/sist/texts")
    parser.add_argument("--cache-path", default=".cache/rag_baseline_texts.sqlite")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--llm-query-keywords", dest="llm_query_keywords", action="store_true", default=True)
    parser.add_argument("--no-llm-query-keywords", dest="llm_query_keywords", action="store_false")
    parser.add_argument("--query-keyword-max-tokens", type=int, default=256)
    parser.add_argument("--query-keyword-max-terms", type=int, default=12)
    parser.add_argument("--query-keyword-thinking", action="store_true")
    parser.add_argument("--llm-rerank", dest="llm_rerank", action="store_true", default=True)
    parser.add_argument("--no-llm-rerank", dest="llm_rerank", action="store_false")
    parser.add_argument("--llm-rerank-candidates", type=int, default=64)
    parser.add_argument("--llm-rerank-max-tokens", type=int, default=768)
    parser.add_argument("--llm-rerank-thinking", action="store_true")
    parser.add_argument("--llm-rerank-chars-per-hit", type=int, default=500)
    parser.add_argument("--iterative-search", dest="iterative_search", action="store_true", default=True)
    parser.add_argument("--no-iterative-search", dest="iterative_search", action="store_false")
    parser.add_argument("--max-search-steps", type=int, default=5)
    parser.add_argument("--rollout-decision-max-tokens", type=int, default=512)
    parser.add_argument("--rollout-decision-thinking", action="store_true")
    parser.add_argument("--rollout-hits-per-step", type=int, default=5)
    parser.add_argument("--verify-answer", action="store_true")
    parser.add_argument("--verification-keyword-max-tokens", type=int, default=256)
    parser.add_argument("--verification-keyword-max-terms", type=int, default=10)
    parser.add_argument("--verification-keyword-thinking", action="store_true")
    parser.add_argument("--verification-hits", type=int, default=6)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global RAG_INSTANCE
    args = parse_args(argv)
    port = find_port(args.port, args.host)
    if args.rag_mode == "baseline":
        config = RAGConfig(
            texts_dir=Path(args.texts_dir),
            cache_path=Path(args.cache_path),
            top_k=args.top_k or 6,
            max_context_chars=args.max_context_chars if args.max_context_chars is not None else 5600,
            max_tokens=args.max_tokens,
        )
        RAG_INSTANCE = BaselineRAG(config).open(rebuild_index=args.rebuild_index)
    else:
        config = UnifiedRAGConfig(
            db_path=Path(args.db_path),
            retrieval_backend=args.retrieval_backend,
            tantivy_index_dir=Path(args.tantivy_index_dir),
            tantivy_candidates=args.tantivy_candidates,
            lexical_candidates=args.lexical_candidates,
            structured_candidates=args.structured_candidates,
            enable_dense_retrieval=args.dense_retrieval,
            dense_index_dir=Path(args.dense_index_dir),
            dense_candidates=args.dense_candidates,
            dense_embedding_base_url=args.embedding_base_url,
            dense_embedding_model=args.embedding_model,
            lexical_rrf_weight=args.lexical_rrf_weight,
            structured_rrf_weight=args.structured_rrf_weight,
            dense_rrf_weight=args.dense_rrf_weight,
            hybrid_rrf_k=args.hybrid_rrf_k,
            neighbor_expansion_window=args.neighbor_expansion_window,
            neighbor_expansion_limit=args.neighbor_expansion_limit,
            top_k=args.top_k or 8,
            max_context_chars=args.max_context_chars if args.max_context_chars is not None else 7200,
            max_tokens=args.max_tokens,
            enable_llm_query_keywords=args.llm_query_keywords,
            query_keyword_max_tokens=args.query_keyword_max_tokens,
            query_keyword_max_terms=args.query_keyword_max_terms,
            query_keyword_enable_thinking=args.query_keyword_thinking,
            enable_llm_rerank=args.llm_rerank,
            llm_rerank_candidates=args.llm_rerank_candidates,
            llm_rerank_max_tokens=args.llm_rerank_max_tokens,
            llm_rerank_enable_thinking=args.llm_rerank_thinking,
            llm_rerank_chars_per_hit=args.llm_rerank_chars_per_hit,
            enable_iterative_search=args.iterative_search,
            max_search_steps=args.max_search_steps,
            rollout_decision_max_tokens=args.rollout_decision_max_tokens,
            rollout_decision_enable_thinking=args.rollout_decision_thinking,
            rollout_hits_per_step=args.rollout_hits_per_step,
            enable_answer_verification=args.verify_answer,
            verification_keyword_max_tokens=args.verification_keyword_max_tokens,
            verification_keyword_max_terms=args.verification_keyword_max_terms,
            verification_keyword_enable_thinking=args.verification_keyword_thinking,
            verification_hits=args.verification_hits,
        )
        RAG_INSTANCE = UnifiedRAG(config).open()
    server = ThreadingHTTPServer((args.host, port), ChatHandler)
    print(f"Serving {args.rag_mode} web chat at http://{args.host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web chat server.")
    finally:
        server.server_close()
        RAG_INSTANCE.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
