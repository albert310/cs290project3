#!/usr/bin/env python3
"""Login to ShanghaiTech CAS and access course/academic systems."""
import asyncio, re
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path("/Users/leslie/Desktop/data/shanghaitech")
USERNAME = "2025222203"
PASSWORD = "vusjef-wupke6-heWjoj"

# Target systems to access after login
TARGETS = [
    ("Egate门户", "https://egate-new.shanghaitech.edu.cn"),
    ("Egate旧版", "https://egate.shanghaitech.edu.cn"),
    ("教务系统OAA", "https://oaa.shanghaitech.edu.cn"),
    ("选课系统", "https://oaa.shanghaitech.edu.cn/4083/list.htm"),
    ("研究生培养", "https://oaa.shanghaitech.edu.cn/4078/list.htm"),
    ("课程信息", "https://oaa.shanghaitech.edu.cn/kcxx/list.htm"),
    ("课表信息", "https://oaa.shanghaitech.edu.cn/kbxx/list.htm"),
    ("培养方案", "https://oaa.shanghaitech.edu.cn/pyfa_4226/list.htm"),
]

def to_fname(url):
    from urllib.parse import urlparse
    p = urlparse(url)
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)
    return f"{p.netloc.replace(':','_')}/AUTH_{path}.md"


async def try_login_and_crawl():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Show browser!
        ctx = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            locale="zh-CN",
        )
        page = await ctx.new_page()

        # Step 1: Navigate to Egate - this should trigger CAS redirect
        print("=== Step 1: Login via CAS ===")
        await page.goto("https://egate-new.shanghaitech.edu.cn", wait_until="networkidle", timeout=30000)
        await asyncio.sleep(3)
        current_url = page.url
        print(f"Current URL: {current_url}")
        print(f"Title: {await page.title()}")

        # Check if we're on a CAS login page
        if "cas" in current_url.lower() or "login" in current_url.lower() or "oauth" in current_url.lower():
            print("Detected CAS login page!")
            await asyncio.sleep(2)

            # Try to fill login form
            username_selectors = [
                "input#username", "input[name='username']", "input[name='user']",
                "input[placeholder*='学号']", "input[placeholder*='工号']",
                "input[placeholder*='账号']", "input[placeholder*='用户']",
                "input[type='text']", "input[type='number']",
            ]
            for sel in username_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el and await el.is_visible():
                        await el.click()
                        await el.fill(USERNAME)
                        print(f"Filled username: {sel}")
                        break
                except:
                    continue

            password_selectors = [
                "input#password", "input[name='password']", "input[name='pass']",
                "input[type='password']",
            ]
            for sel in password_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el and await el.is_visible():
                        await el.click()
                        await el.fill(PASSWORD)
                        print(f"Filled password: {sel}")
                        break
                except:
                    continue

            # Click login
            btn_selectors = [
                "button[type='submit']", "input[type='submit']",
                "button:has-text('登录')", "button:has-text('登 录')",
                "a:has-text('登录')", ".btn-login", "#loginBtn",
                "button:has-text('Login')", "button:has-text('Sign in')",
            ]
            for sel in btn_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el and await el.is_visible():
                        await el.click()
                        print(f"Clicked: {sel}")
                        break
                except:
                    continue

            await asyncio.sleep(5)
            print(f"After login: {page.url}")
            print(f"Title: {await page.title()}")
        else:
            print("No CAS redirect detected - checking for inline login...")
            # Save screenshot for debugging
            await page.screenshot(path=str(OUT / "egate_screenshot.png"))
            print(f"Screenshot saved to egate_screenshot.png")

        # Step 2: Navigate to target pages with the authenticated session
        print("\n=== Step 2: Crawling authenticated pages ===")
        saved = 0
        for name, url in TARGETS:
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(3)
                current = page.url
                text = (await page.evaluate("document.body.innerText")).strip()

                fp = OUT / to_fname(url)
                fp.parent.mkdir(parents=True, exist_ok=True)
                title = await page.title()
                fp.write_text(f"# {title}\n\nURL: {url}\nRedirect: {current}\n\n## Content\n\n{text}", encoding="utf-8")

                # Show key info
                kw_found = []
                for kw in ["课程", "培养", "选课", "学分", "课表", "教师", "上课", "课程名称"]:
                    if kw in text:
                        kw_found.append(kw)
                print(f"  [{name}] {len(text)} chars -> {url[:80]}")
                if kw_found:
                    print(f"    Keywords: {kw_found}")
                    # Show snippet
                    for kw in kw_found[:2]:
                        idx = text.find(kw)
                        if idx >= 0:
                            snippet = text[max(0,idx-30):idx+150].replace("\n"," ")
                            print(f"    [{kw}] ...{snippet[:150]}...")
                saved += 1
            except Exception as e:
                print(f"  [{name}] ERR: {str(e)[:80]}")

        await browser.close()
        print(f"\nDone! {saved} pages saved.")


if __name__ == "__main__":
    asyncio.run(try_login_and_crawl())
