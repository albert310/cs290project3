from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


Message = Mapping[str, str]
THINK_CLOSE = "</think>"
THINK_OPEN = "<think>"
THINK_PREFIXES = (
    THINK_OPEN,
    "Here's a thinking process:",
    "Here is a thinking process:",
    "Thinking process:",
)


class QwenAPIError(RuntimeError):
    """Raised when the OpenAI-compatible server returns an error."""

    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(frozen=True)
class ChatResult:
    raw: str
    answer: str
    think: str
    finish_reason: Optional[str]
    usage: Optional[Dict[str, Any]]
    response: Dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class StreamEvent:
    kind: str
    delta: str = ""
    raw_delta: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    response: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def is_answer(self) -> bool:
        return self.kind == "answer" and bool(self.delta)

    @property
    def is_think(self) -> bool:
        return self.kind == "think" and bool(self.delta)


def _strip_open_tag(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith(THINK_OPEN):
        return text
    leading_len = len(text) - len(stripped)
    return text[:leading_len] + stripped[len(THINK_OPEN) :]


def _strip_think_prefix(text: str) -> str:
    stripped = text.lstrip()
    leading_len = len(text) - len(stripped)
    for prefix in THINK_PREFIXES:
        if stripped.startswith(prefix):
            return text[:leading_len] + stripped[len(prefix) :].lstrip()
    return text


def is_thinking_only(content: str) -> bool:
    stripped = content.lstrip()
    return any(stripped.startswith(prefix) for prefix in THINK_PREFIXES)


def split_think(content: str) -> Tuple[str, str]:
    """Split a Qwen response into (think, answer).

    The local server currently emits reasoning text before a closing
    ``</think>`` marker. Standard ``<think>...</think>`` output is supported
    too. If no thinking marker is present, the full content is returned as the
    answer.
    """

    if THINK_CLOSE in content:
        think, answer = content.split(THINK_CLOSE, 1)
        return _strip_think_prefix(_strip_open_tag(think)).strip(), answer.strip()
    stripped = content.lstrip()
    if is_thinking_only(content):
        return _strip_think_prefix(_strip_open_tag(content)).strip(), ""
    return "", content.strip()


class _StreamingThinkSplitter:
    def __init__(self) -> None:
        self._mode: Optional[str] = None
        self._pending = ""

    def feed(self, delta: str) -> List[Tuple[str, str]]:
        self._pending += delta
        if self._mode is None:
            decided = self._decide_initial_mode()
            if decided is None:
                return []
            self._mode = decided
            if self._mode == "think":
                self._pending = _strip_open_tag(self._pending)
        return self._drain_pending()

    def finish(self) -> List[Tuple[str, str]]:
        if not self._pending:
            return []
        kind = self._mode or self._decide_initial_mode(force=True) or "answer"
        text = self._pending
        self._pending = ""
        if kind == "think":
            text = _strip_open_tag(text)
        return [(kind, text)] if text else []

    def _decide_initial_mode(self, force: bool = False) -> Optional[str]:
        if THINK_CLOSE in self._pending:
            return "think"

        stripped = self._pending.lstrip()
        if not stripped:
            return None

        for prefix in THINK_PREFIXES:
            if stripped.startswith(prefix):
                return "think"
            if prefix.startswith(stripped):
                return None if not force else "answer"

        max_prefix_len = max(len(prefix) for prefix in THINK_PREFIXES)
        if force or len(stripped) >= max_prefix_len:
            return "answer"
        return "answer"

    def _drain_pending(self) -> List[Tuple[str, str]]:
        if self._mode == "answer":
            text = self._pending
            self._pending = ""
            return [("answer", text)] if text else []

        if THINK_CLOSE in self._pending:
            before, after = self._pending.split(THINK_CLOSE, 1)
            self._pending = ""
            self._mode = "answer"
            parts: List[Tuple[str, str]] = []
            if before:
                parts.append(("think", before))
            if after:
                parts.append(("answer", after))
            return parts

        keep = len(THINK_CLOSE) - 1
        if len(self._pending) <= keep:
            return []
        emit = self._pending[:-keep]
        self._pending = self._pending[-keep:]
        return [("think", emit)] if emit else []


class QwenClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("QWEN_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
        self.model = model or os.environ.get("QWEN_MODEL") or "qwen3.6-27b"
        self.api_key = api_key or os.environ.get("QWEN_API_KEY")
        self.timeout = timeout

    def chat(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[Sequence[Message]] = None,
        max_tokens: Optional[int] = 700,
        temperature: Optional[float] = 0,
        model: Optional[str] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> ChatResult:
        payload = self._build_payload(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            extra_body=extra_body,
        )
        body = self._post_json("/v1/chat/completions", payload)
        choice = body.get("choices", [{}])[0]
        raw = ((choice.get("message") or {}).get("content")) or ""
        think, answer = split_think(raw)
        return ChatResult(
            raw=raw,
            answer=answer,
            think=think,
            finish_reason=choice.get("finish_reason"),
            usage=body.get("usage"),
            response=body,
        )

    def ask(self, prompt: str, **kwargs: Any) -> str:
        return self.chat(prompt, **kwargs).answer

    def stream_chat(
        self,
        prompt: Optional[str] = None,
        *,
        messages: Optional[Sequence[Message]] = None,
        max_tokens: Optional[int] = 700,
        temperature: Optional[float] = 0,
        model: Optional[str] = None,
        include_usage: bool = True,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[StreamEvent]:
        payload = self._build_payload(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            extra_body=extra_body,
        )
        payload["stream"] = True
        if include_usage:
            payload["stream_options"] = {"include_usage": True}

        splitter = _StreamingThinkSplitter()
        finish_reason: Optional[str] = None
        usage: Optional[Dict[str, Any]] = None

        for obj in self._stream_json("/v1/chat/completions", payload):
            if obj.get("usage"):
                usage = obj["usage"]

            choices = obj.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice.get("finish_reason")

            raw_delta = ((choice.get("delta") or {}).get("content")) or ""
            if not raw_delta:
                continue

            for kind, delta in splitter.feed(raw_delta):
                yield StreamEvent(kind=kind, delta=delta, raw_delta=raw_delta, response=obj)

        for kind, delta in splitter.finish():
            yield StreamEvent(kind=kind, delta=delta)

        yield StreamEvent(kind="done", finish_reason=finish_reason, usage=usage)

    def stream_answer(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        for event in self.stream_chat(prompt, **kwargs):
            if event.kind == "answer" and event.delta:
                yield event.delta

    def stream_think(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        for event in self.stream_chat(prompt, **kwargs):
            if event.kind == "think" and event.delta:
                yield event.delta

    def models(self) -> Dict[str, Any]:
        return self._get_json("/v1/models")

    def _build_payload(
        self,
        *,
        prompt: Optional[str],
        messages: Optional[Sequence[Message]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        model: Optional[str],
        extra_body: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if prompt is None and messages is None:
            raise ValueError("Pass either prompt or messages.")
        if prompt is not None and messages is not None:
            raise ValueError("Pass prompt or messages, not both.")

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": list(messages) if messages is not None else [{"role": "user", "content": prompt or ""}],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if extra_body:
            payload.update(extra_body)
        return payload

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, path: str, payload: Optional[Mapping[str, Any]] = None) -> urllib.request.addinfourl:
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        method = "GET" if payload is None else "POST"
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise QwenAPIError(f"Qwen server returned HTTP {exc.code}", status=exc.code, body=body) from exc
        except urllib.error.URLError as exc:
            raise QwenAPIError(f"Could not reach Qwen server: {exc}") from exc

    def _get_json(self, path: str) -> Dict[str, Any]:
        with self._request(path) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        with self._request(path, payload) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _stream_json(self, path: str, payload: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
        with self._request(path, payload) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                yield json.loads(data)
