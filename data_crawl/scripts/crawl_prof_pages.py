#!/usr/bin/env python3
"""Crawl all professor profile pages from professors.json"""
import json, asyncio, re
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

PROF_JSON = Path("/Users/leslie/Desktop/data/professors.json")
OUTPUT_DIR = Path("/Users/leslie/Desktop/data/shanghaitech")
CONFIG = CrawlerRunConfig(page_timeout=30000, delay_before_return_html=2.0, cache_mode="bypass")

def url_to_filename(url):
    p = urlparse(url)
    netloc = p.netloc.replace(":", "_")
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)
    if len(path) > 150: path = path[:150]
    if p.query: path = f"{path}__{re.sub(r'[<>:\"/\\|?*]','_',p.query)[:40]}"
    return f"{netloc}/{path}.md"

async def main():
    profs = json.loads(PROF_JSON.read_text(encoding="utf-8"))
    urls = [(p["cnUrl"], p["title"], p.get("school",""), p.get("career",""))
            for p in profs if p.get("cnUrl") and "shanghaitech.edu.cn" in p["cnUrl"]]

    urls = list(set(urls))  # dedupe
    print(f"Crawling {len(urls)} professor profiles...")

    new, skip = 0, 0
    async with AsyncWebCrawler() as crawler:
        for url, name, school, career in urls:
            filepath = OUTPUT_DIR / url_to_filename(url)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            if filepath.exists():
                skip += 1; continue

            try:
                result = await crawler.arun(url, config=CONFIG)
                if result and result.markdown:
                    header = f"# {name}  \n**学院**: {school}  \n**职称**: {career}  \n**URL**: {url}  \n\n---\n\n"
                    filepath.write_text(header + result.markdown, encoding="utf-8")
                    new += 1
                    if new % 20 == 0: print(f"  [{new} new]", flush=True)
            except Exception: pass

    print(f"Done: {new} new, {skip} already existed", flush=True)

asyncio.run(main())
