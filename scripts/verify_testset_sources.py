from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from evaluate_testset import normalize_text, read_jsonl, score_answer, target_groups, validate_records


TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", flags=re.IGNORECASE | re.DOTALL)


def decode_page(raw: bytes) -> str:
    for encoding in ("utf-8", "gb18030", "gbk", "latin1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def page_to_text(raw: bytes) -> str:
    text = decode_page(raw)
    text = SCRIPT_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    return html.unescape(" ".join(text.split()))


def cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.html"


def fetch_url(url: str, cache_dir: Path, retries: int = 3) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, url)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urlopen(req, timeout=25).read()
            path.write_bytes(raw)
            return page_to_text(raw)
        except Exception as exc:  # network can be flaky on faculty pages
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    if path.exists():
        return page_to_text(path.read_bytes())
    raise URLError(f"failed to fetch {url}: {last_error}")


def group_in_text(text_norm: str, group: Sequence[str]) -> bool:
    return any(normalize_text(value) in text_norm for value in group)


def verify_record_sources(item: Mapping[str, Any], fetched: Mapping[str, str]) -> tuple[int, int]:
    urls = list(item.get("source_urls", [])) + list(item.get("cross_check_urls", []))
    combined_norm = normalize_text(" ".join(fetched[url] for url in urls if url in fetched))
    checked = 0
    failed = 0

    for group in target_groups(item.get("source_evidence", [])):
        checked += 1
        if not group_in_text(combined_norm, group):
            failed += 1
            print(f"source evidence missing for {item['id']}: {group}", file=sys.stderr)

    for value in item.get("negative_targets", []):
        checked += 1
        if normalize_text(value) in combined_norm:
            failed += 1
            print(f"negative target appears in official sources for {item['id']}: {value}", file=sys.stderr)

    return checked, failed


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Second-pass verification for the web-verified test set.")
    parser.add_argument("--testset", default="eval/testset_web_verified.jsonl")
    parser.add_argument("--cache-dir", default=".cache/testset_sources")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    records = read_jsonl(Path(args.testset))
    validate_records(records)

    bad_gt = [item["id"] for item in records if not score_answer(item["gt_answer"], item["eval"])]
    if bad_gt:
        print("ground-truth answers failed their eval rules: " + ", ".join(bad_gt), file=sys.stderr)
        return 1

    urls = sorted({url for item in records for url in item.get("source_urls", []) + item.get("cross_check_urls", [])})
    cache_dir = Path(args.cache_dir)
    fetched: Dict[str, str] = {}
    for url in urls:
        fetched[url] = fetch_url(url, cache_dir)

    checked = failed = 0
    for item in records:
        item_checked, item_failed = verify_record_sources(item, fetched)
        checked += item_checked
        failed += item_failed

    print(f"source urls checked: {len(urls)}")
    print(f"ground-truth self-check: {len(records)}/{len(records)}")
    print(f"source evidence checks: {checked - sum(len(item.get('negative_targets', [])) for item in records)}")
    print(f"negative absence checks: {sum(len(item.get('negative_targets', [])) for item in records)}")
    if failed:
        print(f"failed source checks: {failed}", file=sys.stderr)
        return 1
    print("second-pass verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
