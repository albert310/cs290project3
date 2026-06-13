#!/usr/bin/env python3
"""Open browser for manual CAS login, then auto-crawl authenticated pages."""
import asyncio, re
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright

OUT = Path("/Users/leslie/Desktop/data/shanghaitech")

# Pages to crawl AFTER login
TARGETS = [
    # Egate portal - course registration system
    ("Egate首页", "https://egate-new.shanghaitech.edu.cn"),
    ("Egate教学服务", "https://egate-new.shanghaitech.edu.cn/red/index.html"),
    # OAA authenticated pages
    ("选课系统", "https://oaa.shanghaitech.edu.cn/4083/list.htm"),
    ("课表信息", "https://oaa.shanghaitech.edu.cn/kbxx/list.htm"),
    ("课程信息", "https://oaa.shanghaitech.edu.cn/kcxx/list.htm"),
    ("研究生课表", "https://oaa.shanghaitech.edu.cn/kbxx_4234/list.htm"),
    ("本科培养方案", "https://oaa.shanghaitech.edu.cn/pyfa_4226/list.htm"),
    ("研究生培养方案", "https://oaa.shanghaitech.edu.cn/4078/list.htm"),
    ("学籍与学位", "https://oaa.shanghaitech.edu.cn/xjyxw/list.htm"),
    ("成绩查询", "https://oaa.shanghaitech.edu.cn/4106/list.htm"),
    # Common course system URLs
    ("教务系统主站", "https://oaa.shanghaitech.edu.cn/main.htm"),
]


def to_fname(url):
    p = urlparse(url)
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)[:150]
    return f"{p.netloc.replace(':','_')}/AUTH_{path}.md"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
        )
        page = await ctx.new_page()

        # Step 1: Navigate to Egate - triggers CAS redirect
        print("正在打开浏览器，请在窗口中手动登录...")
        await page.goto("https://egate-new.shanghaitech.edu.cn", wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)

        # Step 2: Try to navigate to a service that requires CAS auth
        print("尝试触发CAS认证跳转...")
        await page.goto("https://egate-new.shanghaitech.edu.cn/red/index.html", wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)

        # Step 3: Navigate to OAA to trigger login
        print("尝试访问教务系统...")
        await page.goto("https://oaa.shanghaitech.edu.cn/4083/list.htm", wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)

        current_url = page.url
        print(f"当前页面: {current_url}")
        print(f"页面标题: {await page.title()}")

        # Step 4: Wait for user to manually log in
        if "cas" in current_url.lower() or "login" in current_url.lower():
            print("\n=== 请在浏览器窗口中手动输入账号密码登录 ===")
            print(f"   账号: 2025222203")
            print(f"   密码: vusjef-wupke6-heWjoj")
            print("   登录完成后请等待，脚本会自动继续...\n")

            # Wait until redirected away from CAS login page
            for i in range(60):
                await asyncio.sleep(2)
                url = page.url
                if "cas" not in url.lower() and "login" not in url.lower():
                    print(f"检测到登录成功！当前页面: {url}")
                    break
                if i % 10 == 9:
                    print(f"  ... 等待登录中 ({i*2}秒) ...")
            else:
                print("超时：仍未离开登录页面。请在浏览器确认登录成功后按回车继续...")
        else:
            print("\n=== 如看到登录页面，请手动登录 ===")
            print("   登录完成后按回车键继续...")
            input("")

        await asyncio.sleep(3)
        print(f"\n登录后页面: {page.url}")
        print(f"页面标题: {await page.title()}")

        # Step 5: Crawl all target pages
        print("\n=== 开始爬取认证页面 ===")
        saved = 0
        for name, url in TARGETS:
            fp = OUT / to_fname(url)
            fp.parent.mkdir(parents=True, exist_ok=True)

            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(3)
                text = (await page.evaluate("document.body.innerText")).strip()
                title = await page.title()
                current = page.url

                fp.write_text(
                    f"# {title}\n\nURL: {url}\nRedirect: {current}\n\n## Content\n\n{text}",
                    encoding="utf-8",
                )

                # Show key course-related content
                keywords = ["课程", "培养", "选课", "学分", "课表", "教师", "上课", "成绩", "课程名称", "授课"]
                found = [kw for kw in keywords if kw in text[:3000]]
                clean = text[:200].replace("\n", " ")
                print(f"\n[{name}] {len(text)} chars")
                if found:
                    print(f"  关键词: {found}")
                print(f"  内容预览: {clean[:150]}")

                # If it's a list page with article links, extract and crawl them too
                if "/list.htm" in url and len(text) > 300:
                    links = await page.evaluate("""
                        Array.from(document.querySelectorAll('a[href]')).map(a => ({
                            text: a.textContent.trim().substring(0, 100),
                            href: a.href
                        })).filter(l => l.text.length > 2 && l.href.includes('_page.htm'))
                    """)
                    if links:
                        print(f"  发现 {len(links)} 篇文章链接，正在爬取...")
                        for i, l in enumerate(links[:10]):  # Limit to 10 articles per page
                            try:
                                await page.goto(l["href"], wait_until="domcontentloaded", timeout=20000)
                                await asyncio.sleep(1)
                                art_text = (await page.evaluate("document.body.innerText")).strip()

                                afp = OUT / to_fname(l["href"])
                                afp.parent.mkdir(parents=True, exist_ok=True)
                                art_title = await page.title()
                                afp.write_text(
                                    f"# {art_title}\n\nArticle: {l['text']}\nURL: {l['href']}\n\n## Content\n\n{art_text}",
                                    encoding="utf-8",
                                )
                                print(f"    [{i+1}] {l['text'][:60]} ({len(art_text)} chars)")
                            except:
                                pass

                saved += 1
            except Exception as e:
                print(f"  [{name}] ERR: {str(e)[:80]}")

        # Step 6: Also save a screenshot of the main pages for reference
        try:
            await page.goto("https://oaa.shanghaitech.edu.cn/4083/list.htm", wait_until="networkidle", timeout=30000)
            await page.screenshot(path=str(OUT / "oaa_courses_screenshot.png"), full_page=True)
            print(f"\n保存选课系统截图")
        except:
            pass

        await browser.close()

        total = sum(1 for _ in OUT.rglob("*.md"))
        print(f"\n=== 完成！总文件数: {total} ===")
        print(f"数据保存在: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
