from __future__ import annotations

import argparse
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

from rag import BaselineRAG, RAGConfig, UnifiedRAG, UnifiedRAGConfig


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web" / "static"
RAG_INSTANCE: Any | None = None
RAG_LOCK = threading.Lock()


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
        except Exception as exc:
            self.send_json({"error": f"invalid request: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        if not query:
            self.send_json({"error": "empty query"}, status=HTTPStatus.BAD_REQUEST)
            return

        started = time.perf_counter()
        try:
            with RAG_LOCK:
                result = RAG_INSTANCE.answer(query, top_k=top_k)
            latency = time.perf_counter() - started
            self.send_json(
                {
                    "answer": result.answer,
                    "think": result.think,
                    "query_keywords": result.query_keywords,
                    "search_query": result.search_query or result.query,
                    "query_keyword_error": result.query_keyword_error,
                    "search_rollout": result.search_rollout,
                    "hits": [hit.to_dict() for hit in result.hits],
                    "latency_sec": latency,
                    "usage": result.usage,
                }
            )
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
            with RAG_LOCK:
                for event in RAG_INSTANCE.stream(query, top_k=top_k):
                    if event.get("event") == "done":
                        event["latency_sec"] = time.perf_counter() - started
                    self.send_sse(str(event.get("event") or "message"), event)
        except BrokenPipeError:
            return
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
    parser.add_argument("--texts-dir", default="data/sist/texts")
    parser.add_argument("--cache-path", default=".cache/rag_baseline_texts.sqlite")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--llm-query-keywords", action="store_true")
    parser.add_argument("--query-keyword-max-tokens", type=int, default=256)
    parser.add_argument("--query-keyword-max-terms", type=int, default=12)
    parser.add_argument("--query-keyword-thinking", action="store_true")
    parser.add_argument("--iterative-search", action="store_true")
    parser.add_argument("--max-search-steps", type=int, default=5)
    parser.add_argument("--rollout-decision-max-tokens", type=int, default=512)
    parser.add_argument("--rollout-decision-thinking", action="store_true")
    parser.add_argument("--rollout-hits-per-step", type=int, default=5)
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
            max_context_chars=args.max_context_chars or 5600,
            max_tokens=args.max_tokens,
        )
        RAG_INSTANCE = BaselineRAG(config).open(rebuild_index=args.rebuild_index)
    else:
        config = UnifiedRAGConfig(
            db_path=Path(args.db_path),
            top_k=args.top_k or 8,
            max_context_chars=args.max_context_chars or 7200,
            max_tokens=args.max_tokens,
            enable_llm_query_keywords=args.llm_query_keywords,
            query_keyword_max_tokens=args.query_keyword_max_tokens,
            query_keyword_max_terms=args.query_keyword_max_terms,
            query_keyword_enable_thinking=args.query_keyword_thinking,
            enable_iterative_search=args.iterative_search,
            max_search_steps=args.max_search_steps,
            rollout_decision_max_tokens=args.rollout_decision_max_tokens,
            rollout_decision_enable_thinking=args.rollout_decision_thinking,
            rollout_hits_per_step=args.rollout_hits_per_step,
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
