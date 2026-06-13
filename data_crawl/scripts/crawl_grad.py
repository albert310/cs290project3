#!/usr/bin/env python3
"""CAS login then crawl graduate training plan system + OAA."""
import asyncio, re, json
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright

OUT = Path("/Users/leslie/Desktop/data/shanghaitech")
CAS_URL = "https://ids.shanghaitech.edu.cn/authserver/login?service=https%3A%2F%2Fgraduate.shanghaitech.edu.cn%2Fgsapp%2Fsys%2Fyjsemaphome%2Fportal%2Findex.do"

# Content pages to crawl after login
TARGETS = [
    # Graduate training plan system
    ("研-培养方案查询", "https://graduate.shanghaitech.edu.cn/gsapp/sys/wdpyfaapp/*default/index.do#/pyfaxq"),
    ("研-个人培养方案", "https://graduate.shanghaitech.edu.cn/gsapp/sys/wdpyfaapp/*default/index.do#/pyfa"),
    ("研-主页", "https://graduate.shanghaitech.edu.cn/gsapp/sys/yjsemaphome/portal/index.do"),
    # OAA pages (may show more after auth)
    ("OAA-培养方案", "https://oaa.shanghaitech.edu.cn/pyfa_4226/list.htm"),
    ("OAA-课表信息", "https://oaa.shanghaitech.edu.cn/kbxx/list.htm"),
    ("OAA-课程信息", "https://oaa.shanghaitech.edu.cn/kcxx/list.htm"),
    # Try course query
    ("研-课程查询", "https://graduate.shanghaitech.edu.cn/gsapp/sys/wdkbapp/*default/index.do"),
]


def to_fname(url):
    p = urlparse(url)
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*#\s]+', '_', path)[:150]
    return f"{p.netloc.replace(':','_')}/GRAD_{path}.md"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx = await browser.new_context(viewport={"width": 1920, "height": 1080}, locale="zh-CN")
        page = await ctx.new_page()

        # Login
        print("=== 打开CAS登录页面 ===")
        await page.goto(CAS_URL, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)

        # Auto-fill if possible
        try:
            el = await page.query_selector("input#username,input[name='username']")
            if el and await el.is_visible():
                await el.fill("2025222203")
                print("已填充账号")
        except:
            pass
        try:
            el = await page.query_selector("input#password,input[name='password']")
            if el and await el.is_visible():
                await el.fill("vusjef-wupke6-heWjoj")
                print("已填充密码")
        except:
            pass

        print("请在浏览器中完成登录（如已填充可直接点登录）")
        login_url = page.url
        for i in range(120):
            await asyncio.sleep(2)
            url = page.url
            if url != login_url and "ids.shanghaitech.edu.cn" not in url:
                print(f"登录成功！-> {url}")
                break

        await asyncio.sleep(3)

        # Crawl targets
        print("\n=== 爬取研究生系统 ===")
        saved = 0
        for name, url in TARGETS:
            try:
                await page.goto(url, wait_until="networkidle", timeout=60000)
                await asyncio.sleep(5)

                current = page.url
                title = await page.title()
                text = (await page.evaluate("document.body.innerText")).strip()

                fp = OUT / to_fname(url)
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(f"# {title}\n\nURL: {url}\nRedirect: {current}\n\n## Content\n\n{text}", encoding="utf-8")
                saved += 1

                # Also save screenshot for JS-heavy pages
                if "gsapp" in current:
                    ss_path = OUT / to_fname(url).replace(".md", ".png")
                    await page.screenshot(path=str(ss_path), full_page=True)

                # Try to extract structured data if it's a list page
                if "pyfa" in current.lower() or "培养方案" in title:
                    # Check if there are links to specific programs
                    links = await page.evaluate("""() => {
                        return Array.from(document.querySelectorAll('a[href], div[class*=\"item\"], li[class*=\"item\"]'))
                            .map(el => el.textContent.trim().substring(0, 150))
                            .filter(t => t.length > 3);
                    }""")
                    if links:
                        print(f"  [{name}] {len(text)} chars, {len(links)} items")
                        for l in links[:10]:
                            print(f"    - {l[:100]}")
                else:
                    print(f"  [{name}] {len(text)} chars")

                # Click into sub-items if it's a SPA
                if len(text) < 500 and "gsapp" in current:
                    # Try clicking common elements
                    for sel in ['a[href*="pyfa"]', '.list-item', '[class*="item"]', 'button', 'a']:
                        try:
                            els = await page.query_selector_all(sel)
                            if len(els) > 0 and len(els) < 50:
                                for el in els[:5]:
                                    try:
                                        txt = await el.text_content()
                                        if txt and len(txt.strip()) > 3:
                                            print(f"    Clickable: {txt.strip()[:80]}")
                                    except:
                                        pass
                                break
                        except:
                            pass

            except Exception as e:
                print(f"  [{name}] ERR: {str(e)[:80]}")

        await browser.close()
        total = sum(1 for _ in OUT.rglob("*.md"))
        print(f"\nDone! {saved} pages, total {total} MD files")


if __name__ == "__main__":
    asyncio.run(main())
