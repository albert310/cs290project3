#!/usr/bin/env python3
"""
Recursive text-file crawler: start from shanghaitech.md seed URLs, crawl
web pages layer by layer (depth=4), discover and download text-type files
(.doc/.docx/.xls/.xlsx/.txt/.csv/.xml/.ppt/.pptx/.rst etc.) along the way.

Uses crawl4ai for JS-capable page crawling + aiohttp for fast file downloads.
"""
import asyncio
import re
import sys
import time
import json
import pickle
import hashlib
import aiohttp
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

# ── Paths ──────────────────────────────────────────────────────────
SEED_FILE = Path("/Users/leslie/Desktop/data/shanghaitech.md")
STATE_FILE = Path("/Users/leslie/Desktop/data/crawl_text_recursive_state.pkl")

# Web page markdown output (for link discovery + future use)
PAGES_OUT = Path("/Users/leslie/Desktop/data/shanghaitech_text_pages")

# Downloaded text files (binary originals)
FILES_OUT = Path("/Users/leslie/Desktop/data/shanghaitech_text_files")

# ── Config ─────────────────────────────────────────────────────────
MAX_DEPTH = 6
WORKERS = 8
PAGE_TIMEOUT = 25  # seconds per web page
FILE_CONCURRENCY = 10

DOMAIN = "shanghaitech.edu.cn"
ALLOWED_DOMAINS = [DOMAIN]  # only follow links on these domains

TARGET_EXTS = {
    "txt", "csv", "xml", "json", "tex", "rtf", "log", "md", "rst", "yaml", "yml",
    "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods", "odp",
}

SKIP_PATH_RE = re.compile(
    r"\.(jpg|jpeg|png|gif|svg|ico|webp|css|js|woff|woff2|ttf|eot|pdf|mp4|mp3|"
    r"avi|mov|flv|wmv|zip|rar|tar|gz|bz2|7z|exe|dmg|pkg|apk|iso|bin)$",
    re.I,
)


# ── Helpers ────────────────────────────────────────────────────────

def normalize(url):
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc, path, p.params, p.query, ""))


def is_text_file(url):
    """Check if URL points to a target text-type file."""
    path = urlparse(url).path.lower()
    for ext in TARGET_EXTS:
        if path.endswith("." + ext):
            return True
    return False


def is_web_page(url):
    """Check if URL is a crawlable web page (not a binary file)."""
    if SKIP_PATH_RE.search(url):
        return False
    path = urlparse(url).path.lower()
    # Exclude known binary/text file extensions
    binary_exts = {
        "jpg", "jpeg", "png", "gif", "svg", "ico", "webp", "css", "js",
        "woff", "woff2", "ttf", "eot", "pdf", "mp4", "mp3", "avi", "mov",
        "flv", "wmv", "zip", "rar", "tar", "gz", "bz2", "7z", "exe", "dmg",
        "pkg", "apk", "iso", "bin", "doc", "docx", "xls", "xlsx", "ppt",
        "pptx", "odt", "ods", "odp",
    }
    for ext in binary_exts:
        if path.endswith("." + ext):
            return False
    return True


def check_domain(url):
    """Check if URL is on an allowed domain."""
    try:
        netloc = urlparse(url).netloc
        return any(d in netloc for d in ALLOWED_DOMAINS)
    except Exception:
        return False


def extract_links(text, base_url):
    """Extract ALL links from markdown/HTML text."""
    links = set()
    # Markdown links [text](url)
    for m in re.finditer(r"\[.*?\]\((https?://[^\)]+)\)", text):
        links.add(m.group(1))
    # Plain URLs
    for m in re.finditer(r"(?<!\()(https?://[^\s<>\"')\]]+)", text):
        url = m.group(0).rstrip(".,;:!?\"')")
        if url.count("http") <= 1:
            links.add(url)
    # href/src attributes
    for m in re.finditer(r'(?:href|src)\s*=\s*["\']([^"\']+)["\']', text):
        path = m.group(1)
        if not path.startswith("http"):
            try:
                path = urljoin(base_url, path)
            except Exception:
                continue
        links.add(path)

    resolved = set()
    for link in links:
        try:
            if link.startswith("http"):
                resolved.add(normalize(link))
            else:
                resolved.add(normalize(urljoin(base_url, link)))
        except Exception:
            pass
    return resolved


def url_to_filename(url):
    """Convert URL to a safe filesystem path."""
    p = urlparse(url)
    netloc = p.netloc.replace(":", "_")
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', "_", path)
    if len(path) > 180:
        path = path[:180]
    if p.query:
        qhash = hashlib.md5(p.query.encode()).hexdigest()[:8]
        path = f"{path}__{qhash}"
    # Ensure extension
    ext = p.path.rsplit(".", 1)[-1] if "." in p.path else ""
    if ext and ext.lower() in TARGET_EXTS:
        return f"{netloc}/{path}"
    return f"{netloc}/{path}.md"


# ── State ──────────────────────────────────────────────────────────

def load_state():
    """Load or initialise crawl state."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    return {
        "visited_pages": set(),    # web pages crawled
        "found_text_files": {},    # url -> source_page
        "downloaded_files": set(), # successfully downloaded text file urls
        "failed_files": {},        # url -> error
        "queue": [],               # [(url, depth)]
        "url_depth": {},
    }


def save_state(s):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(s, f)


# ── Seed URLs ──────────────────────────────────────────────────────

def extract_seed_urls():
    """Extract all shanghaitech URLs from the seed markdown file."""
    if not SEED_FILE.exists():
        print("Seed file not found!")
        return []
    text = SEED_FILE.read_text(encoding="utf-8")
    urls = set()
    for m in re.finditer(r"https?://[^\s<>\"')\]]+", text):
        url = m.group(0).rstrip(".,;:!?\"')>")
        if check_domain(url) and is_web_page(url):
            urls.add(normalize(url))
    # Also extract markdown links
    for m in re.finditer(r"\[.*?\]\((https?://[^\)]+)\)", text):
        url = m.group(1)
        if check_domain(url) and is_web_page(url):
            urls.add(normalize(url))
    seed = sorted(urls)
    print(f"Extracted {len(seed)} seed URLs from {SEED_FILE}")
    return seed


# ── Web Page Crawler Worker ────────────────────────────────────────

async def crawl_page(crawler, url, depth):
    """Crawl a web page and return (markdown_text, set_of_links)."""
    filepath = PAGES_OUT / url_to_filename(url)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Use cached version if available
    if filepath.exists():
        try:
            text = filepath.read_text(encoding="utf-8")
            links = extract_links(text, url)
            return text, links
        except Exception:
            pass

    try:
        config = CrawlerRunConfig(
            page_timeout=PAGE_TIMEOUT * 1000,
            cache_mode="bypass",
        )
        result = await asyncio.wait_for(
            crawler.arun(url, config=config),
            timeout=PAGE_TIMEOUT + 5,
        )
        if result and result.markdown:
            filepath.write_text(result.markdown, encoding="utf-8")
            links = extract_links(result.markdown, url)
            return result.markdown, links
    except Exception:
        pass
    return "", set()


# ── File Downloader ────────────────────────────────────────────────

async def download_file(session, url, filepath, sem):
    """Download a text-type file."""
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath.write_bytes(content)
                    return True, None
                else:
                    return False, f"HTTP {resp.status}"
        except asyncio.TimeoutError:
            return False, "timeout"
        except Exception as e:
            return False, str(e)[:100]


# ── Main Loop ──────────────────────────────────────────────────────

async def main():
    PAGES_OUT.mkdir(parents=True, exist_ok=True)
    FILES_OUT.mkdir(parents=True, exist_ok=True)

    state = load_state()
    print(f"Loaded state: {len(state['visited_pages'])} pages, "
          f"{len(state['downloaded_files'])} files, "
          f"{len(state['queue'])} queued")

    # Seed the queue
    if not state["queue"]:
        seeds = extract_seed_urls()
        for url in seeds:
            if url not in state["visited_pages"]:
                state["queue"].append((url, 0))
                state["url_depth"][url] = 0
    else:
        print(f"Resuming with {len(state['queue'])} URLs in queue")

    print(f"\nStarting recursive crawl: depth≤{MAX_DEPTH}, {WORKERS} workers")
    start_time = time.time()

    # Track new text files found this run
    new_text_files = set()

    async with AsyncWebCrawler() as crawler:
        save_counter = 0

        while state["queue"]:
            # Pick URLs up to current depth, within batch
            batch = []
            while state["queue"] and len(batch) < WORKERS * 3:
                url, depth = state["queue"].pop(0)
                if url not in state["visited_pages"] and depth <= MAX_DEPTH:
                    batch.append((url, depth))

            if not batch:
                break

            # Crawl web pages in parallel
            tasks = [crawl_page(crawler, url, depth) for url, depth in batch]
            results = await asyncio.gather(*tasks)

            # Process results
            for (url, depth), (markdown, links) in zip(batch, results):
                state["visited_pages"].add(url)
                if not links:
                    continue

                for link in links:
                    if not check_domain(link):
                        continue

                    if is_text_file(link):
                        # Found a text file!
                        if link not in state["found_text_files"]:
                            state["found_text_files"][link] = url
                            new_text_files.add(link)
                    elif is_web_page(link):
                        # Queue for crawling at next depth
                        nd = depth + 1
                        if nd <= MAX_DEPTH and link not in state["visited_pages"]:
                            if link not in state["url_depth"]:
                                state["queue"].append((link, nd))
                                state["url_depth"][link] = nd

            save_counter += len(batch)
            if save_counter >= 50:
                elapsed = time.time() - start_time
                rate = len(state["visited_pages"]) / max(elapsed, 1) * 60
                depth_info = {}
                for u, d in state["url_depth"].items():
                    if d not in depth_info:
                        depth_info[d] = 0
                    depth_info[d] += 1
                print(f"  [d={depth}] {len(state['visited_pages'])} pages | "
                      f"{len(state['found_text_files'])} text files found | "
                      f"{len(state['queue'])} queued | {rate:.0f} p/min",
                      flush=True)
                save_state(state)
                save_counter = 0

                # Download batch of newly found text files
                if new_text_files:
                    await download_batch(state, new_text_files)
                    new_text_files.clear()

    # Download remaining text files
    await download_batch(state, new_text_files)
    save_state(state)

    # Final stats
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Crawl complete! {elapsed/60:.1f} minutes")
    print(f"  Web pages crawled: {len(state['visited_pages'])}")
    print(f"  Text files found:  {len(state['found_text_files'])}")
    print(f"  Downloaded:        {len(state['downloaded_files'])}")
    print(f"  Failed:            {len(state['failed_files'])}")
    print(f"  Pages output:      {PAGES_OUT}")
    print(f"  Files output:      {FILES_OUT}")

    # Extension breakdown
    ext_counts = {}
    for url in state["found_text_files"]:
        ext = urlparse(url).path.rsplit(".", 1)[-1].lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    print(f"\nText file types found:")
    for ext, cnt in sorted(ext_counts.items(), key=lambda x: -x[1]):
        downloaded = sum(1 for u in state["downloaded_files"]
                         if urlparse(u).path.lower().endswith("." + ext))
        print(f"  .{ext}: {cnt} found, {downloaded} downloaded")


async def download_batch(state, urls):
    """Download a batch of text files."""
    if not urls:
        return

    sem = asyncio.Semaphore(FILE_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=20, force_close=True)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        url_list = []
        for url in urls:
            if url in state["downloaded_files"] or url in state["failed_files"]:
                continue
            filepath = FILES_OUT / url_to_filename(url).replace(".md", "")
            url_list.append(url)
            tasks.append(download_file(session, url, filepath, sem))

        if not tasks:
            return

        results = await asyncio.gather(*tasks)
        for url, (ok, err) in zip(url_list, results):
            if ok:
                state["downloaded_files"].add(url)
            else:
                state["failed_files"][url] = err

        new_dl = sum(1 for u in url_list if u in state["downloaded_files"])
        print(f"    Downloaded {new_dl}/{len(url_list)} text files", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
