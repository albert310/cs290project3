#!/usr/bin/env python3
"""Crawl authenticated pages on shanghaitech.edu.cn requiring login."""
import asyncio
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig

OUTPUT_DIR = Path("/Users/leslie/Desktop/data/shanghaitech")

USERNAME = "2025222203"
PASSWORD = "vusjef-wupke6-heWjoj"

# Login JS - simple synchronous script, no async wrapper needed
LOGIN_JS = f"""
// Wait a bit for page to load
await new Promise(r => setTimeout(r, 3000));

// Find username input
var inputs = document.querySelectorAll('input');
for (var i = 0; i < inputs.length; i++) {{
    var inp = inputs[i];
    var type = (inp.type || '').toLowerCase();
    var ph = (inp.placeholder || '').toLowerCase();
    var name = (inp.name || '').toLowerCase();
    var id = (inp.id || '').toLowerCase();
    if (type === 'text' || type === 'number' || ph.includes('用户') || ph.includes('学号') || ph.includes('工号') || ph.includes('账号') || name.includes('user') || id.includes('user') || name.includes('account') || id.includes('account')) {{
        inp.value = '{USERNAME}';
        inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
        inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
        break;
    }}
}}

// Find password input and fill
var pwInputs = document.querySelectorAll('input[type="password"]');
for (var j = 0; j < pwInputs.length; j++) {{
    pwInputs[j].value = '{PASSWORD}';
    pwInputs[j].dispatchEvent(new Event('input', {{ bubbles: true }}));
    pwInputs[j].dispatchEvent(new Event('change', {{ bubbles: true }}));
    break;
}}

await new Promise(r => setTimeout(r, 1000));

// Try to click login button
var buttons = document.querySelectorAll('button, input[type="submit"], a.btn, .login-btn, [class*="login"], [id*="login"]');
for (var k = 0; k < buttons.length; k++) {{
    var btn = buttons[k];
    var text = (btn.textContent || btn.value || '').trim();
    if (text.includes('登录') || text.includes('登 录') || text.includes('Login') || text.includes('Sign')) {{
        btn.click();
        break;
    }}
}}

// If no specific login button found, try clicking any submit button
if (buttons.length === 0 || true) {{
    var submitBtns = document.querySelectorAll('button[type="submit"], input[type="submit"]');
    for (var m = 0; m < submitBtns.length; m++) {{
        submitBtns[m].click();
        break;
    }}
}}

await new Promise(r => setTimeout(r, 5000));
"""

AUTH_START_URLS = [
    "https://egate-new.shanghaitech.edu.cn",
    "https://egate-new.shanghaitech.edu.cn/index.html",
]

# Browser config - use non-headless to avoid anti-bot
BROWSER_CFG = BrowserConfig(
    headless=True,
    viewport_width=1920,
    viewport_height=1080,
    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    user_agent_mode="random",
)

CRAWLER_CFG = CrawlerRunConfig(
    js_code=LOGIN_JS,
    page_timeout=60000,
    delay_before_return_html=8.0,
    cache_mode="bypass",
    session_id="auth_session_v2",
    scan_full_page=True,
)


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


def extract_links(markdown_text, base_url):
    links = set()
    for m in re.finditer(r'\[.*?\]\((https?://[^\)]+)\)', markdown_text):
        links.add(m.group(1))
    for m in re.finditer(r'(?<!\()https?://[^\s<>"'']+', markdown_text):
        url = m.group(0).rstrip('.,;:!?"\'')
        links.add(url)
    resolved = set()
    for link in links:
        try:
            if link.startswith('http'):
                resolved.add(normalize(link))
            else:
                resolved.add(normalize(urljoin(base_url, link)))
        except Exception:
            pass
    return resolved


async def crawl_auth_pages():
    visited = set()
    queue = [(url, 0) for url in AUTH_START_URLS]
    max_depth = 2

    async with AsyncWebCrawler(config=BROWSER_CFG) as crawler:
        while queue:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            filename = url_to_filename(url)
            filepath = OUTPUT_DIR / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            print(f"[auth d={depth}] {url}", flush=True)
            try:
                result = await crawler.arun(url, config=CRAWLER_CFG)
                if result and result.markdown:
                    filepath.write_text(result.markdown, encoding="utf-8")
                    print(f"  -> Saved: {filepath} ({len(result.markdown)} chars)", flush=True)

                    new_links = extract_links(result.markdown, url)
                    for link in new_links:
                        if "egate" in link and link not in visited:
                            queue.append((link, depth + 1))
                else:
                    print(f"  -> No content", flush=True)
            except Exception as e:
                print(f"  -> Error: {str(e)[:120]}", flush=True)

    print(f"\nAuth crawl done! {len(visited)} pages.", flush=True)


if __name__ == "__main__":
    asyncio.run(crawl_auth_pages())
