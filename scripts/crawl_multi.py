#!/usr/bin/env python3
"""Multi-instance crawler - 4 AsyncWebCrawler instances running concurrently."""
import asyncio
import re
import sys
import pickle
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

OUTPUT_DIR = Path("/Users/leslie/Desktop/data/shanghaitech")
STATE_FILE = Path("/Users/leslie/Desktop/data/crawl_state_multi.pkl")
MAX_PAGES = 50000
MAX_DEPTH = 6
NUM_WORKERS = 48  # balanced for campus network
BATCH_SIZE = 5    # URLs per worker per iteration
HARD_TIMEOUT = 20  # seconds per page max

DOMAIN = "shanghaitech.edu.cn"

SKIP_RE = re.compile(
    r'\.(jpg|jpeg|png|gif|svg|ico|webp|css|js|woff|woff2|ttf|eot|pdf|doc|docx|xls|xlsx|zip|rar|ppt|pptx|mp4|mp3|avi|mov|flv|wmv)$|'
    r'_visitcount|/video/|javascript:|mailto:|\.png\b|\.jpg\b',
    re.I
)

def normalize(url):
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc, path, p.params, p.query, ""))

def should_crawl(url):
    try:
        return DOMAIN in urlparse(url).netloc and not SKIP_RE.search(url)
    except Exception:
        return False

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
                if is_not_found(text):
                    continue  # Skip 404 pages
                for link in extract_links(text, "https://www.shanghaitech.edu.cn"):
                    if should_crawl(link):
                        discovered.add(link)
            except Exception:
                pass
        print(f"  {file_count} files, {len(discovered)} links", flush=True)
    else:
        discovered = set()

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

    for link in discovered:
        if link not in visited and should_crawl(link):
            url_queue.append((link, url_depth.get(link, 1)))
            url_depth[link] = url_depth.get(link, 1)

    # Dedupe
    seen = set()
    new_q = []
    for u, d in url_queue:
        if u not in seen and u not in visited:
            seen.add(u)
            new_q.append((u, d))
    url_queue = new_q

    # Start URLs
    start_file = Path("/Users/leslie/Desktop/data/shanghaitech.md")
    if start_file.exists():
        for m in re.finditer(r'https?://[^\s<>"'')]+', start_file.read_text(encoding="utf-8")):
            url = m.group(0).rstrip('.,;:!?"\'')
            if should_crawl(url):
                u = normalize(url)
                if u not in visited:
                    url_queue.append((u, 0))

    return visited, url_queue, url_depth

def save_state(visited, queue, url_depth):
    try:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump({'visited': visited, 'queue': queue, 'url_depth': url_depth}, f)
    except Exception:
        pass

NOT_FOUND_INDICATORS = [
    'not found', '404 not found', '页面不存在', '无法找到',
    '请求的url未找到', 'the requested url was not found',
    'page not found', '找不到', 'http 404', 'error 404',
]

def is_not_found(content):
    """Detect actual 404/missing page content."""
    if not content:
        return True
    head = content[:800].lower()
    return any(ind.lower() in head for ind in NOT_FOUND_INDICATORS)

# Config for JS-heavy dynamic pages
CRAWL_CONFIG = CrawlerRunConfig(
    page_timeout=30000,
    cache_mode="bypass",
    delay_before_return_html=3.0,  # Wait for AJAX content to load
    scan_full_page=True,            # Scroll to trigger lazy loading
)

async def worker_crawl(worker_id, urls_with_depth, visited_set, new_links_collector):
    """Crawl a batch of URLs with own browser instance."""
    async with AsyncWebCrawler() as crawler:
        for url, depth in urls_with_depth:
            if url in visited_set:
                continue
            visited_set.add(url)

            filepath = OUTPUT_DIR / url_to_filename(url)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if filepath.exists():
                try:
                    text = filepath.read_text(encoding="utf-8")
                    if not is_not_found(text):
                        links = extract_links(text, url)
                        for link in links:
                            if should_crawl(link):
                                nd = depth + 1
                                if nd <= MAX_DEPTH:
                                    new_links_collector.append((link, nd))
                    continue
                except Exception:
                    pass

            try:
                result = await asyncio.wait_for(
                    crawler.arun(url, config=CRAWL_CONFIG),
                    timeout=HARD_TIMEOUT
                )
                if result and result.markdown and not is_not_found(result.markdown):
                    filepath.write_text(result.markdown, encoding="utf-8")
                    links = extract_links(result.markdown, url)
                    for link in links:
                        if should_crawl(link):
                            nd = depth + 1
                            if nd <= MAX_DEPTH:
                                new_links_collector.append((link, nd))
            except (asyncio.TimeoutError, Exception):
                pass


async def main():
    visited, url_queue, url_depth = load_state()
    total_visited = set(visited)  # Track in this session

    print(f"\nMulti-instance: {len(total_visited)} visited, {len(url_queue)} queued, {NUM_WORKERS} workers", flush=True)
    start_time = time.time()
    save_counter = 0

    while url_queue and len(total_visited) < MAX_PAGES:
        # Split batch across workers
        worker_batches = [[] for _ in range(NUM_WORKERS)]
        for i in range(NUM_WORKERS * BATCH_SIZE):
            if not url_queue:
                break
            url, depth = url_queue.pop(0)
            worker_batches[i % NUM_WORKERS].append((url, depth))

        # Filter empty batches
        worker_batches = [b for b in worker_batches if b]

        if not worker_batches:
            break

        new_links = []
        tasks = [asyncio.create_task(worker_crawl(i, batch, total_visited, new_links)) for i, batch in enumerate(worker_batches)]
        # Wait for all workers with a generous timeout
        done, pending = await asyncio.wait(tasks, timeout=HARD_TIMEOUT + 10)
        for t in pending:
            t.cancel()  # Cancel any workers that exceeded the timeout

        # Add new links to queue
        for link, depth in new_links:
            if link not in total_visited and should_crawl(link):
                url_queue.append((link, depth))
                url_depth[link] = depth

        save_counter += sum(len(b) for b in worker_batches)
        if save_counter >= 100:
            elapsed = time.time() - start_time
            rate = len(total_visited) / max(elapsed, 1) * 60
            print(f"  [{len(total_visited)} crawled | {len(url_queue)} queued | {rate:.0f} p/min]", flush=True)
            save_state(total_visited, url_queue, url_depth)
            save_counter = 0

    save_state(total_visited, url_queue, url_depth)
    elapsed = time.time() - start_time
    file_count = sum(1 for _ in OUTPUT_DIR.rglob('*.md'))
    print(f"\nDone! {file_count} files, {len(total_visited)} pages, {elapsed/60:.1f} min.", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
