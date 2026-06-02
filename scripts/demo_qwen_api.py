#!/usr/bin/env python3
from __future__ import annotations

import argparse

from qwen_api import QwenClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo the local qwen_api wrapper.")
    parser.add_argument("prompt", nargs="?", default="2+3 equals what? Answer with one number.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--show-think", action="store_true")
    parser.add_argument("--show-raw", action="store_true")
    args = parser.parse_args()

    client = QwenClient(base_url=args.base_url, model=args.model)

    if args.stream:
        for event in client.stream_chat(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ):
            if event.kind == "think" and args.show_think:
                print(event.delta, end="", flush=True)
            elif event.kind == "answer":
                print(event.delta, end="", flush=True)
            elif event.kind == "done":
                print()
                print(f"finish_reason={event.finish_reason} usage={event.usage}")
        return 0

    result = client.chat(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if args.show_raw:
        print("=== raw ===")
        print(result.raw)
    if args.show_think:
        print("=== think ===")
        print(result.think)
    print("=== answer ===")
    print(result.answer)
    print(f"finish_reason={result.finish_reason} usage={result.usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
