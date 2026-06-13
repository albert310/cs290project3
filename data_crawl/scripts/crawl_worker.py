#!/usr/bin/env python3
"""Single worker - reads URLs from its queue file, crawls and saves them."""
import asyncio
import re
import sys
import pickle
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

OUTPUT_DIR = Path("/Users/leslie/Desktop/data/shanghaitech")
DOMAIN = "shanghaitech.edu.cn"
MAX_DEPTH = 3

SKIP = re.compile(
    r'\.(jpg|jpeg|png|gif|svg|ico|webp|css|js|woff|woff2|ttf|eot|pdf|doc|docx|xls|xlsx|zip|rar|ppt|pptx|mp4|mp3|avi|mov|flv|wmv)$|'
    r'_visitcount|/video/|javascript:|mailto:',
    re.I
)


def normalize(url):
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc, path, p.params, p.query, ""))


def should_crawl(url):
    if DOMAIN not in urlparse(url).netloc:
        return False
    return not SKIP.search(url)


def extract_links(text, base_url):
    links = set()
    for m in re.finditer(r'\[.*?\]\((https?://[^\)]+)\)', text):
        links.add(m.group(1))
    for m in re.finditer(r'(?<!\()https?://[^\s<>"'']+', text):
        url = m.group(0).rstrip('.,;:!?"\'')
        links.add(url)
    resolved = set()
    for link in links:
        try:
            if link.startswith('http'):
                resolved.add(normalize(link))
            else:
                resolved.add(normalize(urljoin(base_url, link)))
        except Exception:
            pass
    return resolved


def url_to_filename(url):
    p = urlparse(url)
    netloc = p.netloc.replace(":", "_")
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)
    if len(path) > 180:
        path = path[:180]
    if p.query:
        q = re.sub(r'[<>:"/\\|?*]', '_', p.query)[:50]
        path = f"{path}__{q}"
    return f"{netloc}/{path}.md"


async def crawl_one(crawler, url, depth):
    filename = url_to_filename(url)
    filepath = OUTPUT_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        try:
            return extract_links(filepath.read_text(encoding="utf-8"), url)
        except Exception:
            pass

    try:
        config = CrawlerRunConfig(page_timeout=30000, cache_mode="bypass")
        result = await crawler.arun(url, config=config)
        if result and result.markdown:
            filepath.write_text(result.markdown, encoding="utf-8")
            return extract_links(result.markdown, url)
    except Exception:
        pass
    return set()


async def run(worker_id, queue_file, result_file):
    """Crawl from queue_file, write new links to result_file."""
    if not queue_file.exists():
        print(f"[W{worker_id}] No queue file, exiting.", flush=True)
        return

    with open(queue_file, 'rb') as f:
        batch = pickle.load(f)

    if not batch:
        print(f"[W{worker_id}] Empty batch, exiting.", flush=True)
        return

    print(f"[W{worker_id}] Starting batch of {len(batch)} URLs", flush=True)
    new_links_all = set()
    count = 0

    async with AsyncWebCrawler() as crawler:
        for url, depth in batch:
            new_links = await crawl_one(crawler, url, depth)
            for link in new_links:
                if should_crawl(link) and link not in new_links_all:
                    nd = depth + 1
                    if nd <= MAX_DEPTH:
                        new_links_all.add((link, nd))
            count += 1

    # Save new links for the dispatcher
    with open(result_file, 'wb') as f:
        pickle.dump(list(new_links_all), f)

    # Mark queue file as done
    queue_file.unlink()
    print(f"[W{worker_id}] Done: {count} crawled, {len(new_links_all)} new links found.", flush=True)


if __name__ == "__main__":
    worker_id = int(sys.argv[1])
    queue_file = Path(sys.argv[2])
    result_file = Path(sys.argv[3])
    asyncio.run(run(worker_id, queue_file, result_file))
