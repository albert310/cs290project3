#!/usr/bin/env python3
"""Handle CAS SSO login using Playwright directly, then crawl authenticated pages."""
import asyncio
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Please install: pip install playwright && playwright install chromium")
    raise

OUTPUT_DIR = Path("/Users/leslie/Desktop/data/shanghaitech")
USERNAME = "2025222203"
PASSWORD = "vusjef-wupke6-heWjoj"

# Egate entry point
EGATE_URL = "https://egate-new.shanghaitech.edu.cn"

# After login, crawl these internal pages
AUTH_URLS = [
    EGATE_URL,
    f"{EGATE_URL}/red/index.html",
    f"{EGATE_URL}/green/index.html",
    f"{EGATE_URL}/blue/index.html",
]


def normalize(url):
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc, path, p.params, p.query, ""))


def url_to_filename(url):
    p = urlparse(url)
    netloc = p.netloc.replace(":", "_")
    path = p.path.strip("/")
    if not path:
        path = "index_auth"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)
    if len(path) > 180:
        path = path[:180]
    if p.query:
        q = re.sub(r'[<>:"/\\|?*]', '_', p.query)[:50]
        path = f"{path}__{q}"
    return f"{netloc}/{path}.md"


def extract_links(html_text, base_url):
    links = set()
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html_text):
        link = m.group(1)
        if link.startswith("http"):
            links.add(normalize(link))
        elif link.startswith("/"):
            links.add(normalize(urljoin(base_url, link)))
    return links


async def try_login(page):
    """Try to log in via CAS or direct form."""
    print("Navigating to Egate...", flush=True)
    await page.goto(EGATE_URL, wait_until="networkidle", timeout=60000)
    await asyncio.sleep(3)

    current_url = page.url
    print(f"Current URL: {current_url}", flush=True)
    title = await page.title()
    print(f"Title: {title}", flush=True)

    # Check if redirected to CAS login
    if "cas" in current_url.lower() or "login" in current_url.lower() or "oauth" in current_url.lower():
        print("Detected CAS/SSO redirect. Looking for login form...", flush=True)
        await asyncio.sleep(2)

        # Try to fill in credentials on the CAS page
        try:
            # Find username field
            user_selectors = [
                'input#username', 'input#user', 'input[name="username"]',
                'input[name="user"]', 'input[name="j_username"]',
                'input[type="text"]', 'input[placeholder*="用户"]',
                'input[placeholder*="学号"]', 'input[placeholder*="工号"]',
                'input[placeholder*="账号"]',
            ]
            for sel in user_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        await el.fill(USERNAME)
                        print(f"Filled username via {sel}", flush=True)
                        break
                except Exception:
                    continue

            # Find password field
            pass_selectors = [
                'input#password', 'input#pass', 'input[name="password"]',
                'input[name="pass"]', 'input[name="j_password"]',
                'input[type="password"]',
            ]
            for sel in pass_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        await el.fill(PASSWORD)
                        print(f"Filled password via {sel}", flush=True)
                        break
                except Exception:
                    continue

            # Click login button
            btn_selectors = [
                'button[type="submit"]', 'input[type="submit"]',
                'button:has-text("登录")', 'button:has-text("登 录")',
                'a:has-text("登录")', 'button:has-text("Login")',
                '.btn-login', '#btnLogin', '[class*="login-btn"]',
            ]
            for sel in btn_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        await el.click()
                        print(f"Clicked login via {sel}", flush=True)
                        break
                except Exception:
                    continue

            await asyncio.sleep(5)
            print(f"After login attempt, URL: {page.url}", flush=True)
            print(f"Title: {await page.title()}", flush=True)
        except Exception as e:
            print(f"Login attempt error: {e}", flush=True)

    elif "egate" in current_url.lower():
        print("Already on Egate page (no redirect)...", flush=True)
        # Try to find and fill login if there's an inline login form
        try:
            inputs = await page.query_selector_all("input")
            for inp in inputs:
                tp = await inp.get_attribute("type") or ""
                name = await inp.get_attribute("name") or ""
                ph = await inp.get_attribute("placeholder") or ""
                if tp == "text" or "user" in name.lower() or "账号" in ph or "学号" in ph:
                    await inp.fill(USERNAME)
                    print("Filled username on inline form", flush=True)
                if tp == "password":
                    await inp.fill(PASSWORD)
                    print("Filled password on inline form", flush=True)

            # Click login
            btns = await page.query_selector_all("button, input[type='submit']")
            for btn in btns:
                text = await btn.inner_text()
                text = text.strip()
                if "登录" in text or "Login" in text:
                    await btn.click()
                    print("Clicked inline login button", flush=True)
                    break

            await asyncio.sleep(5)
        except Exception as e:
            print(f"Inline login error: {e}", flush=True)

    return current_url


async def crawl_auth_page(page, url, depth, visited, max_depth=2):
    if url in visited or depth > max_depth:
        return set()
    visited.add(url)

    filename = url_to_filename(url)
    filepath = OUTPUT_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    print(f"[auth d={depth}] {url}", flush=True)
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)
        html = await page.content()
        text = await page.evaluate("document.body.innerText")

        if text:
            md_content = f"# {await page.title()}\n\nURL: {url}\n\n{text}"
            filepath.write_text(md_content, encoding="utf-8")
            print(f"  -> Saved: {filepath} ({len(md_content)} chars)", flush=True)
        else:
            print(f"  -> No text content", flush=True)

        return extract_links(html, url)
    except Exception as e:
        print(f"  -> Error: {str(e)[:120]}", flush=True)
        return set()


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )
        page = await context.new_page()

        # Step 1: Login
        await try_login(page)

        # Step 2: Crawl authenticated pages
        visited = set()
        queue = [(url, 0) for url in AUTH_URLS]
        max_depth = 2

        while queue:
            url, depth = queue.pop(0)
            new_links = await crawl_auth_page(page, url, depth, visited, max_depth)
            if new_links:
                for link in new_links:
                    if "egate" in link and link not in visited:
                        queue.append((link, depth + 1))

        print(f"\nAuth crawl done! {len(visited)} pages.", flush=True)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
