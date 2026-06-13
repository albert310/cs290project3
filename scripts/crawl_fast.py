#!/usr/bin/env python3
"""Fast parallel crawler using arun_many for batch concurrency."""
import asyncio
import re
import sys
import pickle
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

OUTPUT_DIR = Path("/Users/leslie/Desktop/data/shanghaitech")
STATE_FILE = Path("/Users/leslie/Desktop/data/crawl_state_fast2.pkl")
MAX_PAGES = 20000
MAX_DEPTH = 3
BATCH_SIZE = 30  # arun_many processes these concurrently

DOMAIN = "shanghaitech.edu.cn"

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
    return DOMAIN in urlparse(url).netloc and not SKIP.search(url)

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

def load_state():
    visited = set()
    url_queue = []
    url_depth = {}

    if OUTPUT_DIR.exists():
        print("Scanning existing files...", flush=True)
        file_count = 0
        discovered = set()
        for md_file in OUTPUT_DIR.rglob("*.md"):
            file_count += 1
            try:
                text = md_file.read_text(encoding="utf-8")
                for link in extract_links(text, "https://www.shanghaitech.edu.cn"):
                    if should_crawl(link):
                        discovered.add(link)
            except Exception:
                pass
        print(f"  {file_count} files, {len(discovered)} links found", flush=True)
    else:
        discovered = set()
        file_count = 0

    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'rb') as f:
                saved = pickle.load(f)
                visited = saved.get('visited', set())
                url_queue = saved.get('queue', [])
                url_depth = saved.get('url_depth', {})
                print(f"  Loaded state: {len(visited)} visited, {len(url_queue)} queued", flush=True)
        except Exception:
            pass

    # Add newly discovered links to queue
    for link in discovered:
        if link not in visited and should_crawl(link):
            d = url_depth.get(link, 1)
            url_queue.append((link, d))
            url_depth[link] = d

    # Dedupe
    seen = set()
    new_q = []
    for u, d in url_queue:
        if u not in seen and u not in visited:
            seen.add(u)
            new_q.append((u, d))
            url_depth[u] = d
    url_queue = new_q

    return visited, url_queue, url_depth

def save_state(visited, queue, url_depth):
    with open(STATE_FILE, 'wb') as f:
        pickle.dump({'visited': visited, 'queue': queue, 'url_depth': url_depth}, f)

async def main():
    visited, url_queue, url_depth = load_state()

    # Add initial URLs
    start_file = Path("/Users/leslie/Desktop/data/shanghaitech.md")
    if start_file.exists():
        for m in re.finditer(r'https?://[^\s<>"'')]+', start_file.read_text(encoding="utf-8")):
            url = m.group(0).rstrip('.,;:!?"\'')
            if should_crawl(url):
                u = normalize(url)
                if u not in visited:
                    url_queue.append((u, 0))

    print(f"\nFast crawl: {len(visited)} visited, {len(url_queue)} queued, batch_size={BATCH_SIZE}", flush=True)
    start_time = time.time()
    save_counter = 0

    async with AsyncWebCrawler() as crawler:
        while url_queue and len(visited) < MAX_PAGES:
            # Take a batch
            batch = []
            for _ in range(min(BATCH_SIZE, len(url_queue))):
                url, depth = url_queue.pop(0)
                batch.append((url, depth))
                visited.add(url)

            # Prepare URLs and configs
            urls = [url for url, _ in batch]
            config = CrawlerRunConfig(page_timeout=30000, cache_mode="bypass")

            # arun_many - concurrent crawling
            results = await crawler.arun_many(urls=urls, config=config)

            # Process results
            new_links_count = 0
            for (url, depth), result in zip(batch, results):
                if result and result.markdown:
                    # Save to file
                    filepath = OUTPUT_DIR / url_to_filename(url)
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath.write_text(result.markdown, encoding="utf-8")

                    # Extract links
                    new_links = extract_links(result.markdown, url)
                    for link in new_links:
                        if link not in visited and should_crawl(link):
                            nd = depth + 1
                            if nd <= MAX_DEPTH:
                                url_queue.append((link, nd))
                                url_depth[link] = nd
                                new_links_count += 1

            save_counter += len(batch)
            elapsed = time.time() - start_time
            if save_counter >= 100:
                rate = len(visited) / max(elapsed, 1) * 60
                print(f"  [{len(visited)} crawled | {len(url_queue)} queued | {rate:.0f} p/min]", flush=True)
                save_state(visited, url_queue, url_depth)
                save_counter = 0

    save_state(visited, url_queue, url_depth)
    elapsed = time.time() - start_time
    file_count = sum(1 for _ in OUTPUT_DIR.rglob('*.md'))
    print(f"\nDone! {file_count} files, {len(visited)} pages, {elapsed/60:.1f} min.", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
