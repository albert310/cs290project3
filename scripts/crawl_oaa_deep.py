#!/usr/bin/env python3
"""Deep crawl OAA course/curriculum pages"""
import asyncio, re
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright

OUT = Path("/Users/leslie/Desktop/data/shanghaitech")

URLS = [
    "https://oaa.shanghaitech.edu.cn/pyfa_4226/list.htm",
    "https://oaa.shanghaitech.edu.cn/kbxx/list.htm",
    "https://oaa.shanghaitech.edu.cn/kcxx/list.htm",
    "https://oaa.shanghaitech.edu.cn/4094/list.htm",
    "https://oaa.shanghaitech.edu.cn/fwzn_4229/list.htm",
    "https://oaa.shanghaitech.edu.cn/fwzn_4233/list.htm",
    "https://oaa.shanghaitech.edu.cn/wjxz/list.htm",
    "https://oaa.shanghaitech.edu.cn/xl/list.htm",
    "https://sist.shanghaitech.edu.cn/2722/list.htm",
    "https://sist.shanghaitech.edu.cn/glance/list.htm",
    "https://sist.shanghaitech.edu.cn/yzjy/list.htm",
    "https://www.shanghaitech.edu.cn/1054/main.htm",
    "https://www.shanghaitech.edu.cn/dsj/list.htm",
]

def to_fname(url):
    p = urlparse(url)
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)
    return f"{p.netloc.replace(':','_')}/COURSE_{path}.md"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await ctx.new_page()
        saved = 0
        for url in URLS:
            fp = OUT / to_fname(url)
            fp.parent.mkdir(parents=True, exist_ok=True)
            if fp.exists():
                print(f"  SKIP: {fp.name}")
                continue
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(3)
                text = (await page.evaluate("document.body.innerText")).strip()
                if len(text) > 200:
                    title = await page.title()
                    fp.write_text(f"# {title}\n\nURL: {url}\n\n## Content\n\n{text}", encoding="utf-8")
                    saved += 1
                    clean = text[:150].replace("\n", " ")
                    print(f"  OK ({len(text)} chars): {clean[:100]}")
                else:
                    print(f"  SHORT ({len(text)} chars): {url}")
            except Exception as e:
                print(f"  ERR: {url} - {str(e)[:60]}")
        await browser.close()
    print(f"Saved: {saved} new pages")
    total = sum(1 for _ in OUT.rglob("*.md"))
    print(f"Total MD: {total}")

asyncio.run(main())
