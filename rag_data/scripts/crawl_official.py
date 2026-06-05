from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from common import RAG_DATA_ROOT, host_of, is_allowed_host, read_json, stable_hash, write_json


USER_AGENT = "Mozilla/5.0 (cs290project3 clean rag builder; educational use)"
HTML_TYPES = ("text/html", "application/xhtml")
DEFAULT_SEEDS = RAG_DATA_ROOT / "config" / "seeds.json"
RAW_DIR = RAG_DATA_ROOT / "raw" / "official_pages"
MANIFEST_PATH = RAG_DATA_ROOT / "processed" / "official_crawl_manifest.jsonl"


@dataclass(frozen=True)
class QueueItem:
    url: str
    category: str
    priority: float
    depth: int
    max_depth: int
    parent_url: str = ""


def canonicalize_url(url: str, base: str = "") -> str:
    joined = urljoin(base, url)
    joined, _fragment = urldefrag(joined)
    parsed = urlparse(joined)
    if parsed.scheme not in {"http", "https"}:
        return ""
    return parsed.geturl()


def should_follow(url: str, allowed_hosts: List[str], allowed_suffixes: List[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".ico", ".zip", ".rar")):
        return False
    return is_allowed_host(parsed.netloc, allowed_hosts, allowed_suffixes)


def extract_links(html: str, base_url: str, allowed_hosts: List[str], allowed_suffixes: List[str]) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for anchor in soup.find_all("a", href=True):
        href = canonicalize_url(anchor.get("href", ""), base=base_url)
        if href and should_follow(href, allowed_hosts, allowed_suffixes):
            links.append(href)
    return links


def fetch(session: requests.Session, url: str, *, timeout: float) -> requests.Response:
    return session.get(
        url,
        timeout=(min(timeout, 5.0), timeout),
        headers={"User-Agent": USER_AGENT},
        allow_redirects=True,
    )


def crawl(
    *,
    seeds_path: Path = DEFAULT_SEEDS,
    raw_dir: Path = RAW_DIR,
    manifest_path: Path = MANIFEST_PATH,
    max_pages: int = 180,
    timeout: float = 12.0,
    sleep: float = 0.25,
) -> Dict[str, Any]:
    config = read_json(seeds_path)
    allowed_hosts = list(config.get("allowed_hosts") or [])
    allowed_suffixes = list(config.get("allowed_host_suffixes") or [])
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    queue: deque[QueueItem] = deque()
    for seed in config.get("seeds", []):
        url = canonicalize_url(str(seed["url"]))
        if not url:
            continue
        queue.append(
            QueueItem(
                url=url,
                category=str(seed.get("category") or "general"),
                priority=float(seed.get("priority") or 0.8),
                depth=0,
                max_depth=int(seed.get("max_depth") or 0),
            )
        )

    seen = set()
    records: List[Dict[str, Any]] = []
    session = requests.Session()
    fetched = 0
    failed = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        while queue and fetched < max_pages:
            item = queue.popleft()
            if item.url in seen:
                continue
            seen.add(item.url)
            if not should_follow(item.url, allowed_hosts, allowed_suffixes):
                skipped += 1
                continue
            record: Dict[str, Any] = {
                "url": item.url,
                "category": item.category,
                "priority": item.priority,
                "depth": item.depth,
                "parent_url": item.parent_url,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            try:
                response = fetch(session, item.url, timeout=timeout)
                response.encoding = response.apparent_encoding or response.encoding
                content_type = response.headers.get("content-type", "")
                html = response.text if response.text else ""
                digest = stable_hash(response.url + "\n" + html, length=32)
                suffix = ".html" if "html" in content_type.lower() or response.url.lower().endswith((".htm", ".html", ".psp")) else ".txt"
                raw_path = raw_dir / f"{digest}{suffix}"
                raw_path.write_text(html, encoding="utf-8", errors="ignore")
                record.update(
                    {
                        "status": "ok",
                        "status_code": response.status_code,
                        "final_url": response.url,
                        "host": host_of(response.url),
                        "content_type": content_type,
                        "encoding": response.encoding,
                        "raw_path": str(raw_path.relative_to(RAG_DATA_ROOT)),
                        "bytes": len(response.content),
                    }
                )
                fetched += 1
                if item.depth < item.max_depth and response.status_code == 200 and any(t in content_type.lower() for t in HTML_TYPES):
                    for link in extract_links(html, response.url, allowed_hosts, allowed_suffixes):
                        if link not in seen:
                            queue.append(
                                QueueItem(
                                    url=link,
                                    category=item.category,
                                    priority=max(item.priority - 0.08, 0.4),
                                    depth=item.depth + 1,
                                    max_depth=item.max_depth,
                                    parent_url=response.url,
                                )
                            )
            except Exception as exc:
                failed += 1
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            records.append(record)
            time.sleep(sleep)

    summary = {
        "seeds_path": str(seeds_path),
        "manifest_path": str(manifest_path),
        "raw_dir": str(raw_dir),
        "records": len(records),
        "fetched": fetched,
        "failed": failed,
        "skipped": skipped,
        "seen": len(seen),
    }
    write_json(RAG_DATA_ROOT / "reports" / "crawl_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl official ShanghaiTech/SIST pages into rag_data/raw.")
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--max-pages", type=int, default=180)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()
    summary = crawl(seeds_path=args.seeds, max_pages=args.max_pages, timeout=args.timeout, sleep=args.sleep)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
