#!/usr/bin/env python3
"""Try all possible leadership URL patterns for remaining 6 schools."""
import asyncio, re
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright

OUTPUT = Path("/Users/leslie/Desktop/data/shanghaitech")

MISSING = {
    "sist": ["sist.shanghaitech.edu.cn", "信息科学"],
    "sem":  ["sem.shanghaitech.edu.cn", "创业管理"],
    "bme":  ["bme.shanghaitech.edu.cn", "生医工"],
    "ims":  ["ims.shanghaitech.edu.cn", "数学"],
    "cts":  ["cts.shanghaitech.edu.cn", "大科学"],
    "smdl": ["smdl.shanghaitech.edu.cn", "材料器件"],
}

# All patterns to try
PATTERNS = [
    "xyld/list.htm", "xzld/list.htm", "ld/list.htm", "leadership/list.htm",
    "xrld/list.htm", "lrld/list.htm", "deanswelcome/list.htm",
    "glance/list.htm", "committees/list.htm", "about/list.htm",
    "xztd/list.htm", "administration/list.htm", "js/list.htm",
    "overview_13310/list.htm", "zzjg/list.htm", "zzjg_13274/list.htm",
    "xxld/list.htm", "leader/list.htm", "leaders/list.htm",
    "presentleader/list.htm",
]

def url_to_filename(url):
    p = urlparse(url)
    netloc = p.netloc.replace(":", "_")
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)
    return f"{netloc}/LEADER_{path}.md"


async def try_url(page, url):
    """Return rendered text if page has meaningful content."""
    try:
        await page.goto(url, wait_until="networkidle", timeout=20000)
        await asyncio.sleep(2)
        text = await page.evaluate("document.body.innerText")
        text = text.strip()
        # Skip empty / error pages
        if len(text) < 50 or "访问地址无效" in text or "Not Found" in text or "找不到" in text:
            return None
        return text
    except Exception:
        return None


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        page = await context.new_page()

        found = {}
        for school_code, (domain, name) in MISSING.items():
            print(f"\n{'='*60}")
            print(f"{name} ({domain})")
            print(f"{'='*60}")
            found[school_code] = []

            for pattern in PATTERNS:
                url = f"https://{domain}/{pattern}"
                text = await try_url(page, url)
                if text:
                    # Check for leadership keywords
                    kw = ["院长", "副院长", "所长", "副所长", "主任", "领导",
                          "助理院长", "教授", "Dean", "Director", "Chair",
                          "书记", "行政", "管理团队", "现任"]
                    has_kw = [k for k in kw if k in text]
                    if has_kw:
                        filepath = OUTPUT / url_to_filename(url)
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        title = await page.title()
                        md = f"# {title}\n\nURL: {url}\n\n## Content\n\n{text}"
                        filepath.write_text(md, encoding="utf-8")
                        found[school_code].append((url, len(text), has_kw))
                        print(f"  FOUND: {pattern} ({len(text)} chars) [{', '.join(has_kw[:5])}]", flush=True)
                    else:
                        print(f"  SKIP: {pattern} ({len(text)} chars) - no leadership keywords", flush=True)

        await browser.close()

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for school_code, (domain, name) in MISSING.items():
            if found[school_code]:
                print(f"\n{name}:")
                for url, size, kws in found[school_code]:
                    print(f"  {url} ({size} chars) [{', '.join(kws[:3])}]")
            else:
                print(f"\n{name}: NOTHING FOUND")


if __name__ == "__main__":
    asyncio.run(main())
