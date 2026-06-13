#!/usr/bin/env python3
"""Recursively crawl shanghaitech.edu.cn pages and save as markdown."""
import asyncio
import os
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

OUTPUT_DIR = Path("/Users/leslie/Desktop/data/shanghaitech")
START_URLS_FILE = Path("/Users/leslie/Desktop/data/shanghaitech.md")
MAX_PAGES = 500
MAX_DEPTH = 3

DOMAIN = "shanghaitech.edu.cn"

# URL dedup: normalize URLs to avoid re-crawling
visited = set()
url_queue = []
depth_map = {}

def normalize(url):
    """Remove fragments, trailing slashes for consistency."""
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    # Remove jsessionid etc.
    return urlunparse((p.scheme, p.netloc, path, p.params, p.query, ""))

def should_crawl(url):
    """Only crawl HTML pages within shanghaitech.edu.cn."""
    if DOMAIN not in urlparse(url).netloc:
        return False
    # Skip images, static assets, tracking pixels
    skip_patterns = [
        r'\.(jpg|jpeg|png|gif|svg|ico|webp|css|js|woff|woff2|ttf|eot|pdf|doc|docx|xls|xlsx|zip|rar)$',
        r'_visitcount',
        r'/video/',
        r'javascript:',
    ]
    for pat in skip_patterns:
        if re.search(pat, url, re.IGNORECASE):
            return False
    return True

def extract_links(markdown_text, base_url):
    """Extract all http/https links from markdown text."""
    links = set()
    # Markdown links: [text](url)
    for m in re.finditer(r'\[.*?\]\((https?://[^\)]+)\)', markdown_text):
        links.add(m.group(1))
    # Plain URLs
    for m in re.finditer(r'(?<!\()https?://[^\s<>"'']+', markdown_text):
        url = m.group(0).rstrip('.,;:!?"\'')
        links.add(url)
    # Resolve relative to base
    resolved = set()
    for link in links:
        if link.startswith('http'):
            resolved.add(normalize(link))
        else:
            resolved.add(normalize(urljoin(base_url, link)))
    return resolved

def url_to_filename(url):
    """Convert URL to a safe filename."""
    p = urlparse(url)
    # Use netloc as subdirectory
    netloc = p.netloc.replace(":", "_")
    path = p.path.strip("/")
    if not path:
        path = "index"
    # Replace special chars
    path = re.sub(r'[<>:"/\\|?*]', '_', path)
    # Truncate
    if len(path) > 200:
        path = path[:200]
    # Add query hash
    if p.query:
        q = re.sub(r'[<>:"/\\|?*]', '_', p.query)[:50]
        path = f"{path}__{q}"
    return f"{netloc}/{path}.md"

async def crawl_page(crawler, url, depth):
    """Crawl a single page, save markdown, return new links."""
    if url in visited:
        return set()
    if depth > MAX_DEPTH:
        return set()

    visited.add(url)
    filename = url_to_filename(url)
    filepath = OUTPUT_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already crawled
    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        return extract_links(content, url)

    print(f"[depth={depth}] Crawling: {url}", flush=True)

    try:
        config = CrawlerRunConfig(page_timeout=30000, cache_mode="bypass")
        result = await crawler.arun(url, config=config)

        if result and result.markdown:
            # Save markdown
            filepath.write_text(result.markdown, encoding="utf-8")
            print(f"  -> Saved: {filepath}", flush=True)
            return extract_links(result.markdown, url)
        else:
            print(f"  -> No content", flush=True)
            return set()
    except Exception as e:
        print(f"  -> Error: {e}", flush=True)
        return set()

def extract_starting_urls():
    """Extract starting URLs from the initial markdown file."""
    text = START_URLS_FILE.read_text(encoding="utf-8")
    urls = set()
    for m in re.finditer(r'https?://[^\s<>"'')]+', text):
        url = m.group(0).rstrip('.,;:!?"\'')
        if should_crawl(url):
            urls.add(normalize(url))
    return urls

async def main():
    start_urls = extract_starting_urls()
    print(f"Found {len(start_urls)} starting URLs to crawl", flush=True)

    # Initialize queue with starting URLs
    for url in start_urls:
        if url not in visited:
            url_queue.append((url, 0))
            depth_map[url] = 0

    async with AsyncWebCrawler() as crawler:
        while url_queue and len(visited) < MAX_PAGES:
            # Process in batches
            batch_size = min(5, len(url_queue))
            batch = []
            for _ in range(batch_size):
                url, depth = url_queue.pop(0)
                batch.append((url, depth))

            tasks = [crawl_page(crawler, url, depth) for url, depth in batch]
            results = await asyncio.gather(*tasks)

            # Add new links to queue
            for (url, depth), new_links in zip(batch, results):
                if new_links:
                    for link in new_links:
                        if link not in visited and should_crawl(link):
                            new_depth = depth + 1
                            if new_depth <= MAX_DEPTH:
                                url_queue.append((link, new_depth))

            print(f"Progress: {len(visited)} crawled, {len(url_queue)} queued", flush=True)

    print(f"\nDone! Crawled {len(visited)} pages total.", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
