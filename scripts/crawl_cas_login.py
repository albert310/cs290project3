#!/usr/bin/env python3
"""Manual CAS login then crawl OAA authenticated pages."""
import asyncio, re
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright

OUT = Path("/Users/leslie/Desktop/data/shanghaitech")

CAS_LOGIN = "https://ids.shanghaitech.edu.cn/authserver/login?service=https%3A%2F%2Fegate.shanghaitech.edu.cn%2Findex.html"

TARGETS = [
    ("选课通知","https://oaa.shanghaitech.edu.cn/4083/list.htm"),
    ("课表信息","https://oaa.shanghaitech.edu.cn/kbxx/list.htm"),
    ("课程信息","https://oaa.shanghaitech.edu.cn/kcxx/list.htm"),
    ("培养方案","https://oaa.shanghaitech.edu.cn/pyfa_4226/list.htm"),
    ("研究生培养","https://oaa.shanghaitech.edu.cn/4078/list.htm"),
    ("学科专业","https://oaa.shanghaitech.edu.cn/xkzy_4225/list.htm"),
    ("规章制度","https://oaa.shanghaitech.edu.cn/4094/list.htm"),
    ("文件下载","https://oaa.shanghaitech.edu.cn/wjxz/list.htm"),
    ("毕业论文","https://oaa.shanghaitech.edu.cn/bylwwsjw/list.htm"),
    ("Egate教学","https://egate.shanghaitech.edu.cn/index.html"),
]

def to_fname(url):
    p = urlparse(url)
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)[:150]
    return f"{p.netloc.replace(':','_')}/CAS_{path}.md"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
        )
        page = await ctx.new_page()

        # Step 1: Open CAS login
        print("=== 正在打开CAS登录页面 ===")
        await page.goto(CAS_LOGIN, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)
        print(f"当前页面: {await page.title()}")

        # Try to auto-fill credentials
        try:
            user_input = await page.query_selector("input#username,input[name='username']")
            if user_input and await user_input.is_visible():
                await user_input.fill("2025222203")
                print("已填充账号")
        except: pass
        try:
            pass_input = await page.query_selector("input#password,input[name='password']")
            if pass_input and await pass_input.is_visible():
                await pass_input.fill("vusjef-wupke6-heWjoj")
                print("已填充密码")
        except: pass

        # Step 2: Wait for manual login (or auto-click if we filled credentials)
        print("\n请在浏览器窗口中完成登录（如已自动填充，可直接点击登录按钮）")
        print("登录成功后将自动继续...")

        login_url = page.url
        for i in range(120):
            await asyncio.sleep(2)
            url = page.url
            title = await page.title()
            if url != login_url and "ids.shanghaitech.edu.cn" not in url:
                print(f"\n检测到登录完成！")
                print(f"当前页面: {url}")
                print(f"标题: {title}")
                break
            if i % 15 == 14:
                print(f"  等待登录中... ({i*2}秒)")

        await asyncio.sleep(3)

        # Step 3: Crawl authenticated pages
        print("\n=== 开始爬取OAA页面 ===")
        saved = 0
        for name, url in TARGETS:
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(3)

                # Extract all article links from rendered page
                links = await page.evaluate("""() => {
                    return Array.from(document.querySelectorAll('a[href]')).map(a => ({
                        text: a.textContent.trim().substring(0, 150),
                        href: a.href
                    })).filter(l => l.text.length > 3 && (
                        l.href.includes('_page.htm') || l.href.includes('.pdf')
                    ));
                }""")

                # Save the list page
                text = (await page.evaluate("document.body.innerText")).strip()
                title = await page.title()
                fp = OUT / to_fname(url)
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(f"# {title}\n\nURL: {url}\n\n## Content\n\n{text}", encoding="utf-8")
                saved += 1

                articles = [l for l in links if "_page.htm" in l["href"]]
                pdfs = [l for l in links if ".pdf" in l["href"]]
                print(f"\n[{name}] {len(text)} chars, {len(articles)} articles, {len(pdfs)} PDFs")

                if articles:
                    for a in articles[:8]:
                        print(f"  📄 {a['text'][:70]}")
                        print(f"     {a['href'][:130]}")

                        # Crawl individual articles too
                        afp = OUT / to_fname(a["href"])
                        afp.parent.mkdir(parents=True, exist_ok=True)
                        if not afp.exists():
                            try:
                                await page.goto(a["href"], wait_until="domcontentloaded", timeout=20000)
                                await asyncio.sleep(1)
                                atext = (await page.evaluate("document.body.innerText")).strip()
                                atitle = await page.title()
                                afp.write_text(f"# {atitle}\n\nTitle: {a['text']}\nURL: {a['href']}\n\n## Content\n\n{atext}", encoding="utf-8")
                                saved += 1
                                print(f"     -> 已保存 ({len(atext)} chars)")
                            except:
                                pass

                if pdfs:
                    import requests
                    PDF_OUT = Path("/Users/leslie/Desktop/data/shanghaitech_pdfs")
                    for p in pdfs[:5]:
                        pfname = re.sub(r'[<>:"/\\|?*]', '_', p["href"].split("/")[-1])[:200]
                        pfp = PDF_OUT / pfname
                        if not pfp.exists():
                            try:
                                r = requests.get(p["href"], timeout=60, headers={"User-Agent":"Mozilla/5.0"})
                                if r.status_code == 200:
                                    pfp.write_bytes(r.content)
                                    print(f"  📥 PDF: {pfname[:60]} ({len(r.content)//1024}KB)")
                            except:
                                pass

            except Exception as e:
                print(f"  [{name}] ERR: {str(e)[:80]}")

        await browser.close()
        total = sum(1 for _ in OUT.rglob("*.md"))
        pdfs = sum(1 for _ in Path("/Users/leslie/Desktop/data/shanghaitech_pdfs").rglob("*.pdf"))
        print(f"\n=== 完成！{total} MD, {pdfs} PDFs ===")

if __name__ == "__main__":
    asyncio.run(main())
