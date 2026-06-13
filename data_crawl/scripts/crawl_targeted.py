#!/usr/bin/env python3
"""Targeted crawl for: course catalogs, training plans, lectures, news."""
import asyncio, re, requests
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

OUT = Path("/Users/leslie/Desktop/data/shanghaitech")
BASE_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# ====== KEY TARGETS ======
CATEGORIES = {
    "教务与课程": [
        # OAA - 教务处核心页面
        "https://oaa.shanghaitech.edu.cn/pyfa/list.htm",       # 培养方案
        "https://oaa.shanghaitech.edu.cn/xkzy/list.htm",       # 学科专业
        "https://oaa.shanghaitech.edu.cn/xkzy_4225/list.htm",  # 学科专业2
        "https://oaa.shanghaitech.edu.cn/4082/list.htm",       # 课程目录
        "https://oaa.shanghaitech.edu.cn/4092/list.htm",       # 课程
        "https://oaa.shanghaitech.edu.cn/xwlw/list.htm",       # 学位论文
        "https://oaa.shanghaitech.edu.cn/4107/list.htm",       # 规章制度
        "https://oaa.shanghaitech.edu.cn/xjyxw/list.htm",      # 学籍与学位
        "https://oaa.shanghaitech.edu.cn/4083/list.htm",       # 选课
        "https://oaa.shanghaitech.edu.cn/bylwwsjw/list.htm",   # 毕业论文
        # 各学院培养方案
        "https://sist.shanghaitech.edu.cn/undergraduate/list.htm",
        "https://sist.shanghaitech.edu.cn/graduate/list.htm",
        "https://slst.shanghaitech.edu.cn/undergraduate/list.htm",
        "https://spst.shanghaitech.edu.cn/undergraduate/list.htm",
        "https://sem.shanghaitech.edu.cn/undergraduate/list.htm",
    ],
    "师资与科研": [
        # Research directions
        "https://sist.shanghaitech.edu.cn/research/list.htm",
        "https://sist.shanghaitech.edu.cn/2725/list.htm",      # Research centers
        "https://slst.shanghaitech.edu.cn/researchfields/list.htm",
        "https://spst.shanghaitech.edu.cn/field/list.htm",
        "https://bme.shanghaitech.edu.cn/research/list.htm",
        "https://sist.shanghaitech.edu.cn/crjs/list.htm",      # 常任教授列表
    ],
    "动态与时讯": [
        # News and lectures
        "https://sist.shanghaitech.edu.cn/news/list.htm",
        "https://sist.shanghaitech.edu.cn/seminar/list.htm",
        "https://www.shanghaitech.edu.cn/1001/list.htm",       # 新闻
        "https://www.shanghaitech.edu.cn/hd/list.htm",          # 活动
        "https://www.shanghaitech.edu.cn/1006/list.htm",        # 科研进展
        "https://slst.shanghaitech.edu.cn/seminar/list.htm",
        "https://spst.shanghaitech.edu.cn/seminar/list.htm",
    ],
    "基础概况": [
        "https://www.shanghaitech.edu.cn/1054/main.htm",        # 学校概况
        "https://www.shanghaitech.edu.cn/jgsz/list.htm",         # 机构设置
        "https://sist.shanghaitech.edu.cn/2722/list.htm",        # SIST使命愿景
        "https://sist.shanghaitech.edu.cn/glance/list.htm",      # SIST概况
    ],
}


def url_to_filename(url):
    p = urlparse(url)
    netloc = p.netloc.replace(":", "_")
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)[:150]
    q = re.sub(r'[<>:"/\\|?*]', '_', p.query)[:50] if p.query else ""
    suffix = f"__{q}" if q else ""
    return f"{netloc}/FOCUS_{path}{suffix}.md"


async def crawl_with_playwright(urls, page):
    """Use Playwright for JS-heavy pages."""
    saved = 0
    for url in urls:
        fp = OUT / url_to_filename(url)
        fp.parent.mkdir(parents=True, exist_ok=True)
        if fp.exists():
            continue
        try:
            resp = await page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(2)
            text = (await page.evaluate("document.body.innerText")).strip()
            if len(text) > 100:
                title = await page.title()
                fp.write_text(f"# {title}\n\nURL: {url}\n\n## Content\n\n{text}", encoding="utf-8")
                saved += 1
                print(f"  PW: {url.split('/')[-2]}/{url.split('/')[-1]} ({len(text)} chars)")
        except:
            pass
    return saved


def crawl_with_http(urls):
    """Use HTTP+BS4 for static pages."""
    saved = 0
    for url in urls:
        fp = OUT / url_to_filename(url)
        fp.parent.mkdir(parents=True, exist_ok=True)
        if fp.exists():
            continue
        try:
            r = requests.get(url, headers=BASE_HEADERS, timeout=30)
            if r.status_code == 200 and len(r.text) > 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                text = soup.get_text(separator="\n", strip=True)
                if len(text) > 100:
                    fp.write_text(f"# {soup.title.string if soup.title else '?'}\n\nURL: {url}\n\n## Content\n\n{text}", encoding="utf-8")
                    saved += 1
                    print(f"  HTTP: {url.split('/')[-2]}/{url.split('/')[-1]} ({len(text)} chars)")
        except:
            pass
    return saved


async def main():
    print("=== Phase 1: HTTP (static pages) ===")
    all_urls = []
    for cat, urls in CATEGORIES.items():
        all_urls.extend(urls)
    http_saved = crawl_with_http(all_urls)
    print(f"HTTP saved: {http_saved}")

    print("\n=== Phase 2: Playwright (JS-heavy pages) ===")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1920, "height": 1080}, user_agent=BASE_HEADERS["User-Agent"])
        page = await ctx.new_page()
        pw_saved = await crawl_with_playwright(all_urls, page)
        await browser.close()
    print(f"Playwright saved: {pw_saved}")

    # Scan for NEW PDFs discovered on newly crawled pages
    print("\n=== Phase 3: Discover new PDFs ===")
    new_pdfs = set()
    for md_file in OUT.rglob("FOCUS_*.md"):
        text = md_file.read_text(encoding="utf-8")
        for m in re.finditer(r'https?://[^\s<>"\')]+\.pdf', text):
            new_pdfs.add(m.group(0).rstrip(".,;:!?\")]"))
    print(f"New PDFs discovered: {len(new_pdfs)}")

    # Download new PDFs
    pdf_dir = Path("/Users/leslie/Desktop/data/shanghaitech_pdfs")
    dl = 0
    for url in new_pdfs:
        fname = re.sub(r'[<>:"/\\|?*%]', '_', url.split("/")[-1])[:200]
        fp = pdf_dir / fname
        if fp.exists():
            continue
        try:
            r = requests.get(url, headers=BASE_HEADERS, timeout=60, stream=True)
            if r.status_code == 200:
                with open(fp, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                dl += 1
        except:
            pass
    print(f"New PDFs downloaded: {dl}")

    total = sum(1 for _ in OUT.rglob("*.md"))
    pdf_total = sum(1 for _ in pdf_dir.rglob("*.pdf"))
    print(f"\nFinal: {total} MD files, {pdf_total} PDFs")


if __name__ == "__main__":
    asyncio.run(main())
