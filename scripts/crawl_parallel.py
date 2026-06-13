#!/usr/bin/env python3
"""Multi-process parallel crawler - 8 workers, each with own browser."""
import asyncio
import re
import sys
import time
import pickle
import multiprocessing as mp
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

OUTPUT_DIR = Path("/Users/leslie/Desktop/data/shanghaitech")
STATE_FILE = Path("/Users/leslie/Desktop/data/crawl_state_fast.pkl")
MAX_PAGES = 20000
MAX_DEPTH = 3
NUM_WORKERS = 8
BATCH_PER_WORKER = 10

DOMAIN = "shanghaitech.edu.cn"

SKIP_PATTERNS = [
    r'\.(jpg|jpeg|png|gif|svg|ico|webp|css|js|woff|woff2|ttf|eot|pdf|doc|docx|xls|xlsx|zip|rar|ppt|pptx|mp4|mp3|avi|mov|flv|wmv)$',
    r'_visitcount', r'/video/', r'javascript:', r'mailto:',
]

def normalize(url):
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc, path, p.params, p.query, ""))

def should_crawl(url):
    if DOMAIN not in urlparse(url).netloc:
        return False
    for pat in SKIP_PATTERNS:
        if re.search(pat, url, re.IGNORECASE):
            return False
    return True

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

async def worker_crawl(urls, worker_id):
    """Crawl a batch of URLs. Returns list of (url, markdown_text_or_None, error)."""
    results = []
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            filename = url_to_filename(url)
            filepath = OUTPUT_DIR / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if filepath.exists():
                try:
                    text = filepath.read_text(encoding="utf-8")
                    results.append((url, text, None))
                    continue
                except Exception:
                    pass

            try:
                config = CrawlerRunConfig(page_timeout=30000, cache_mode="bypass")
                result = await crawler.arun(url, config=config)
                if result and result.markdown:
                    filepath.write_text(result.markdown, encoding="utf-8")
                    results.append((url, result.markdown, None))
                else:
                    results.append((url, None, "No content"))
            except Exception as e:
                results.append((url, None, str(e)[:100]))
    return results

def run_worker(urls, worker_id):
    """Entry point for multiprocessing worker."""
    return asyncio.run(worker_crawl(urls, worker_id))

def load_state():
    """Load visited URLs and queue from existing files and pickle."""
    visited = set()
    url_queue = []
    url_depth = {}
    discovered_all = set()

    if OUTPUT_DIR.exists():
        print("Loading existing files...", flush=True)
        for md_file in OUTPUT_DIR.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                links = extract_links(content, "https://www.shanghaitech.edu.cn")
                for link in links:
                    if should_crawl(link):
                        discovered_all.add(link)
            except Exception:
                pass
        print(f"  Scanned {sum(1 for _ in OUTPUT_DIR.rglob('*.md'))} files, {len(discovered_all)} links found", flush=True)

    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'rb') as f:
                saved = pickle.load(f)
                visited = saved.get('visited', set())
                url_queue = saved.get('queue', [])
                url_depth = saved.get('url_depth', {})
                print(f"  Loaded saved state: {len(visited)} visited, {len(url_queue)} queued", flush=True)
        except Exception as e:
            print(f"  Could not load state: {e}", flush=True)

    # Merge: visited = files on disk, queue = discovered but not visited
    for link in discovered_all:
        if link not in visited and should_crawl(link):
            d = url_depth.get(link, 1)
            url_queue.append((link, d))
            url_depth[link] = d

    # Dedupe queue
    seen = set()
    new_q = []
    for u, d in url_queue:
        if u not in seen and u not in visited:
            seen.add(u)
            new_q.append((u, d))
    url_queue = new_q

    # Add start URLs from initial file
    start_file = Path("/Users/leslie/Desktop/data/shanghaitech.md")
    if start_file.exists():
        text = start_file.read_text(encoding="utf-8")
        for m in re.finditer(r'https?://[^\s<>"'')]+', text):
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

def main():
    visited, url_queue, url_depth = load_state()
    initial_queue = len(url_queue)
    print(f"\nStarting parallel crawl: {len(visited)} visited, {initial_queue} queued, {NUM_WORKERS} workers", flush=True)

    start_time = time.time()
    save_counter = 0

    with mp.Pool(NUM_WORKERS) as pool:
        while url_queue and len(visited) < MAX_PAGES:
            # Grab URLs for each worker
            worker_batches = []
            for _ in range(NUM_WORKERS):
                batch = []
                for _ in range(BATCH_PER_WORKER):
                    if url_queue:
                        url, depth = url_queue.pop(0)
                        batch.append(url)
                        visited.add(url)
                    else:
                        break
                if batch:
                    worker_batches.append(batch)

            if not worker_batches:
                break

            # Dispatch to workers
            futures = [pool.apply_async(run_worker, (batch, i)) for i, batch in enumerate(worker_batches)]

            # Collect results
            total_new = 0
            for future, batch in zip(futures, worker_batches):
                try:
                    results = future.get(timeout=120)
                    for url, text, error in results:
                        if text:
                            new_links = extract_links(text, url)
                            for link in new_links:
                                if link not in visited and should_crawl(link):
                                    d = url_depth.get(url, 1) + 1
                                    if d <= MAX_DEPTH:
                                        url_queue.append((link, d))
                                        url_depth[link] = d
                                        total_new += 1
                except Exception as e:
                    print(f"Worker batch failed: {e}", flush=True)

            save_counter += len(worker_batches) * BATCH_PER_WORKER
            elapsed = time.time() - start_time
            rate = (len(visited) - (sum(1 for _ in OUTPUT_DIR.rglob('*.md')) or len(visited))) / max(elapsed, 1) * 60
            # Actually just compute rate from the last save
            if save_counter >= 100:
                print(f"  [{len(visited)} crawled, {len(url_queue)} queued, {len(visited) / max(elapsed, 1) * 60:.0f} pages/min]", flush=True)
                save_state(visited, url_queue, url_depth)
                save_counter = 0

    save_state(visited, url_queue, url_depth)
    elapsed = time.time() - start_time
    file_count = sum(1 for _ in OUTPUT_DIR.rglob('*.md'))
    print(f"\nDone! {file_count} files, {len(visited)} pages, {elapsed/60:.1f} min total.", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
