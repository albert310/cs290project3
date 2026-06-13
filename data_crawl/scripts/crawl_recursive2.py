#!/usr/bin/env python3
"""Recursively crawl shanghaitech.edu.cn pages with resume support."""
import asyncio
import re
import pickle
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

OUTPUT_DIR = Path("/Users/leslie/Desktop/data/shanghaitech")
STATE_FILE = Path("/Users/leslie/Desktop/data/crawl_state.pkl")
MAX_PAGES = 20000
MAX_DEPTH = 3
BATCH_SIZE = 5

DOMAIN = "shanghaitech.edu.cn"

visited = set()
url_queue = []
url_depth = {}

def normalize(url):
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc, path, p.params, p.query, ""))

def should_crawl(url):
    if DOMAIN not in urlparse(url).netloc:
        return False
    skip_patterns = [
        r'\.(jpg|jpeg|png|gif|svg|ico|webp|css|js|woff|woff2|ttf|eot|pdf|doc|docx|xls|xlsx|zip|rar|ppt|pptx|mp4|mp3|avi|mov|flv|wmv)$',
        r'_visitcount',
        r'/video/',
        r'javascript:',
        r'mailto:',
    ]
    for pat in skip_patterns:
        if re.search(pat, url, re.IGNORECASE):
            return False
    return True

def extract_links(markdown_text, base_url):
    links = set()
    for m in re.finditer(r'\[.*?\]\((https?://[^\)]+)\)', markdown_text):
        links.add(m.group(1))
    for m in re.finditer(r'(?<!\()https?://[^\s<>"'']+', markdown_text):
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
    path = p.path.strip("/")
    if not path:
        path = "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)
    if len(path) > 180:
        path = path[:180]
    if p.query:
        q = re.sub(r'[<>:"/\\|?*]', '_', p.query)[:50]
        path = f"{path}__{q}"
    return f"{netloc}/{path}.md"

async def crawl_page(crawler, url, depth):
    if url in visited:
        return set()

    visited.add(url)
    filename = url_to_filename(url)
    filepath = OUTPUT_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        try:
            content = filepath.read_text(encoding="utf-8")
            return extract_links(content, url)
        except Exception:
            pass

    print(f"[d={depth}] {url}", flush=True)
    try:
        config = CrawlerRunConfig(page_timeout=30000, cache_mode="bypass")
        result = await crawler.arun(url, config=config)
        if result and result.markdown:
            filepath.write_text(result.markdown, encoding="utf-8")
            return extract_links(result.markdown, url)
        else:
            return set()
    except Exception as e:
        msg = str(e)[:100]
        print(f"  E: {msg}", flush=True)
        return set()

def load_existing():
    global visited, url_queue, url_depth
    if not OUTPUT_DIR.exists():
        return

    print("Loading existing crawled files...", flush=True)
    crawled_count = 0
    discovered = set()

    for md_file in OUTPUT_DIR.rglob("*.md"):
        crawled_count += 1
        try:
            content = md_file.read_text(encoding="utf-8")
            links = extract_links(content, "https://www.shanghaitech.edu.cn")
            for link in links:
                if should_crawl(link):
                    discovered.add(link)
        except Exception:
            pass

    print(f"Found {crawled_count} files on disk", flush=True)

    # Load saved state
    saved_visited = set()
    saved_queue = []
    saved_depth = {}
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'rb') as f:
                saved = pickle.load(f)
                saved_visited = saved.get('visited', set())
                saved_queue = saved.get('queue', [])
                saved_depth = saved.get('url_depth', {})
            print(f"Loaded saved state: {len(saved_visited)} visited, {len(saved_queue)} queued", flush=True)
        except Exception as e:
            print(f"Could not load state: {e}", flush=True)

    # visited = all URLs with files on disk
    visited = saved_visited | set()

    # Build queue: discovered URLs not yet visited
    all_known = discovered | set(saved_queue)
    seen_in_queue = set()
    for url in all_known:
        if url not in visited and url not in seen_in_queue and should_crawl(url):
            d = saved_depth.get(url, 1)
            url_queue.append((url, d))
            url_depth[url] = d
            seen_in_queue.add(url)

    print(f"Rebuilt: {len(visited)} visited, {len(url_queue)} queued (from {len(discovered)} discovered)", flush=True)

def save_state():
    try:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump({'visited': visited, 'queue': url_queue, 'url_depth': url_depth}, f)
    except Exception as e:
        print(f"Could not save state: {e}", flush=True)

async def main():
    load_existing()

    # Add start URLs from original file
    start_file = Path("/Users/leslie/Desktop/data/shanghaitech.md")
    if start_file.exists():
        text = start_file.read_text(encoding="utf-8")
        for m in re.finditer(r'https?://[^\s<>"'')]+', text):
            url = m.group(0).rstrip('.,;:!?"\'')
            if should_crawl(url):
                u = normalize(url)
                if u not in visited:
                    url_queue.append((u, 0))
                    url_depth[u] = 0

    print(f"Starting crawl: {len(visited)} visited, {len(url_queue)} queued (max {MAX_PAGES})", flush=True)

    async with AsyncWebCrawler() as crawler:
        progress_counter = 0
        while url_queue and len(visited) < MAX_PAGES:
            batch = []
            for _ in range(min(BATCH_SIZE, len(url_queue))):
                url, depth = url_queue.pop(0)
                batch.append((url, depth))

            tasks = [crawl_page(crawler, url, depth) for url, depth in batch]
            results = await asyncio.gather(*tasks)

            for (url, depth), new_links in zip(batch, results):
                if new_links:
                    for link in new_links:
                        if link not in visited and should_crawl(link):
                            nd = depth + 1
                            if nd <= MAX_DEPTH:
                                url_queue.append((link, nd))
                                url_depth[link] = nd

            progress_counter += len(batch)
            if progress_counter >= 50:
                print(f"  [{len(visited)} crawled, {len(url_queue)} queued]", flush=True)
                save_state()
                progress_counter = 0

    save_state()
    print(f"\nDone! {len(visited)} pages total, {len(url_queue)} remaining in queue.", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
