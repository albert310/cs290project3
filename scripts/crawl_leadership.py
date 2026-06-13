#!/usr/bin/env python3
"""Extract leadership pages rendered via Playwright (handles JS-heavy pages)."""
import asyncio, re
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright

OUTPUT = Path("/Users/leslie/Desktop/data/shanghaitech")

# Schools and their leadership page URLs
LEADERSHIP_URLS = {
    "spst": ["https://spst.shanghaitech.edu.cn/xyld/list.htm",
             "https://spst.shanghaitech.edu.cn/administration/list.htm",
             "https://spst.shanghaitech.edu.cn/about/list.htm"],
    "slst": ["https://slst.shanghaitech.edu.cn/leadership/list.htm",
             "https://slst.shanghaitech.edu.cn/lrld/list.htm",
             "https://slst.shanghaitech.edu.cn/about/list.htm",
             "https://slst.shanghaitech.edu.cn/xztd/list.htm"],
    "sist": ["https://sist.shanghaitech.edu.cn/xyld/list.htm",
             "https://sist.shanghaitech.edu.cn/about/list.htm",
             "https://sist.shanghaitech.edu.cn/xztd/list.htm"],
    "sem":  ["https://sem.shanghaitech.edu.cn/xyld/list.htm",
             "https://sem.shanghaitech.edu.cn/about/list.htm",
             "https://sem.shanghaitech.edu.cn/xztd/list.htm"],
    "sca":  ["https://sca.shanghaitech.edu.cn/xyld/list.htm",
             "https://sca.shanghaitech.edu.cn/about/list.htm",
             "https://sca.shanghaitech.edu.cn/xztd/list.htm"],
    "ih":   ["https://ih.shanghaitech.edu.cn/yld/list.htm",
             "https://ih.shanghaitech.edu.cn/about/list.htm"],
    "bme":  ["https://bme.shanghaitech.edu.cn/xyld/list.htm",
             "https://bme.shanghaitech.edu.cn/about/list.htm",
             "https://bme.shanghaitech.edu.cn/xztd/list.htm",
             "https://bme.shanghaitech.edu.cn/leadership/list.htm"],
    "siais":["https://siais.shanghaitech.edu.cn/xyld/list.htm",
             "https://siais.shanghaitech.edu.cn/about/list.htm",
             "https://siais.shanghaitech.edu.cn/xztd/list.htm"],
    "ihuman":["https://ihuman.shanghaitech.edu.cn/xyld/list.htm",
              "https://ihuman.shanghaitech.edu.cn/xztd/list.htm",
              "https://ihuman.shanghaitech.edu.cn/ldd/main.htm"],
    "ims":  ["https://ims.shanghaitech.edu.cn/xyld/list.htm",
             "https://ims.shanghaitech.edu.cn/about/list.htm",
             "https://ims.shanghaitech.edu.cn/xztd/list.htm"],
    "cts":  ["https://cts.shanghaitech.edu.cn/xyld/list.htm",
             "https://cts.shanghaitech.edu.cn/about/list.htm",
             "https://cts.shanghaitech.edu.cn/xztd/list.htm"],
    "smdl": ["https://smdl.shanghaitech.edu.cn/xyld/list.htm",
             "https://smdl.shanghaitech.edu.cn/about/list.htm",
             "https://smdl.shanghaitech.edu.cn/xztd/list.htm"],
}

def url_to_filename(url):
    p = urlparse(url)
    netloc = p.netloc.replace(":", "_")
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)
    return f"{netloc}/PLAYWRIGHT_{path}.md"


async def render_page(page, url):
    """Navigate, wait for JS, scroll, return text content."""
    print(f"  Loading: {url}", flush=True)
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(3)  # Extra time for dynamic content

        # Scroll to trigger lazy loading
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1)
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(1)

        # Get rendered text
        title = await page.title()
        text = await page.evaluate("document.body.innerText")
        html = await page.content()

        links = []
        # Extract links from rendered page
        hrefs = await page.evaluate("""
            Array.from(document.querySelectorAll('a[href]')).map(a => ({
                text: a.textContent.trim(),
                href: a.href
            })).filter(l => l.text && l.text.length > 0 && l.text.length < 200)
        """)
        for l in hrefs:
            links.append(f"[{l['text']}]({l['href']})")

        md = f"# {title}\n\nURL: {url}\n\n## Content\n\n{text}\n\n## Links\n\n" + "\n".join(links)
        return md
    except Exception as e:
        return f"# Error\n\nURL: {url}\n\n**Error**: {str(e)[:200]}"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )
        page = await context.new_page()

        saved = 0
        for school, urls in LEADERSHIP_URLS.items():
            print(f"\n{school.upper()}:", flush=True)
            for url in urls:
                filepath = OUTPUT / url_to_filename(url)
                filepath.parent.mkdir(parents=True, exist_ok=True)
                if filepath.exists():
                    size = len(filepath.read_text(encoding="utf-8"))
                    print(f"  SKIP (exists, {size} chars): {url}", flush=True)
                    continue

                content = await render_page(page, url)
                filepath.write_text(content, encoding="utf-8")
                saved += 1
                content_len = len(content.split("## Content\n")[-1].split("## Links")[0].strip())
                print(f"  SAVED ({content_len} chars content): {filepath.name}", flush=True)

        await browser.close()
        print(f"\nDone! {saved} new pages saved.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
