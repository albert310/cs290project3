#!/usr/bin/env python3
"""
Login to graduate system and crawl training plan details from pyfacxapp.
Uses Playwright with visible browser for manual login (CAPTCHA).
Navigation uses hash routing within the SPA.
"""
import asyncio
import re
import json
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path("/Users/leslie/Desktop/data/shanghaitech_training_plans")
OUT.mkdir(parents=True, exist_ok=True)

CAS_URL = (
    "https://ids.shanghaitech.edu.cn/authserver/login"
    "?service=https%3A%2F%2Fgraduate.shanghaitech.edu.cn"
    "%2Fgsapp%2Fsys%2Fyjsemaphome%2Fportal%2Findex.do"
)

BASE_URL = "https://graduate.shanghaitech.edu.cn/gsapp/sys/pyfacxapp/*default/index.do"

USERNAME = "2025222203"
PASSWORD = "vusjef-wupke6-heWjoj"

# (hash_route, school_name)
TARGETS = [
    ("/pyfacxlb/2025/物质科学与技术学院/all/0201", "物质科学与技术学院"),
    ("/pyfacxlb/2025/创意与艺术学院/all/0205", "创意与艺术学院"),
    ("/pyfacxlb/2025/生命科学与技术学院/all/0202", "生命科学与技术学院"),
    ("/pyfacxlb/2025/信息科学与技术学院/all/0203", "信息科学与技术学院"),
    ("/pyfacxlb/2025/生物医学工程学院/all/0209", "生物医学工程学院"),
    ("/pyfacxlb/2025/数学科学研究所/all/0303", "数学科学研究所"),
    ("/pyfacxlb/2025/创业与管理学院/all/0204", "创业与管理学院"),
]


def safe_name(text):
    text = re.sub(r'[<>:"/\\|?*#\s]+', '_', text).strip("_")
    return text[:60] if text else "unknown"


async def login(page):
    """Login via CAS SSO."""
    print("=== CAS 登录 ===")
    await page.goto(CAS_URL, wait_until="networkidle", timeout=30000)
    await asyncio.sleep(2)

    try:
        el = await page.query_selector("input#username,input[name='username']")
        if el and await el.is_visible():
            await el.fill(USERNAME)
            print("  已填充账号")
    except Exception:
        pass
    try:
        el = await page.query_selector("input#password,input[name='password']")
        if el and await el.is_visible():
            await el.fill(PASSWORD)
            print("  已填充密码")
    except Exception:
        pass

    print("  请在浏览器中完成登录 (如有验证码)...")
    login_url = page.url
    for i in range(120):
        await asyncio.sleep(2)
        url = page.url
        if url != login_url and "ids.shanghaitech.edu.cn" not in url:
            print("  登录成功!")
            return True
        if i % 15 == 14:
            print(f"  等待登录中... ({i*2}s)")
    return False


async def process_school(page, hash_route, school_name):
    """
    Navigate to the school's plan listing page, click all '详情' buttons,
    and extract the expanded content.
    """
    full_url = BASE_URL + "#" + hash_route
    print(f"\n{'='*50}")
    print(f"学校: {school_name}")
    print(f"路由: {hash_route}")

    # Force full reload by navigating away first, then to target
    # This prevents SPA from showing stale cached content from previous hash
    await page.goto(
        "https://graduate.shanghaitech.edu.cn/gsapp/sys/yjsemaphome/portal/index.do",
        wait_until="networkidle", timeout=30000
    )
    await asyncio.sleep(2)

    # Now navigate to target with full page load
    await page.goto(full_url, wait_until="networkidle", timeout=60000)
    await asyncio.sleep(5)

    # Verify we're on the right page by checking for school name in text
    body_text = await page.evaluate("document.body.innerText")
    print(f"  页面字符数: {len(body_text)}")
    # Show first 200 chars to verify page loaded correctly
    first_line = body_text.strip().split("\n")[0] if body_text else "EMPTY"
    print(f"  首行: {first_line[:100]}")

    if school_name[:2] not in body_text[:800]:
        print(f"  警告: 页面中未找到 '{school_name[:2]}'")
        # Try hash navigation as fallback
        await page.evaluate(f"window.location.hash = '{hash_route}'")
        await asyncio.sleep(5)
        body_text = await page.evaluate("document.body.innerText")
        print(f"  重试后页面字符数: {len(body_text)}")

    # Screenshot
    ss_path = OUT / f"_screenshot_{safe_name(school_name)}.png"
    await page.screenshot(path=str(ss_path), full_page=False)

    # Save the list page content
    school_dir = OUT / safe_name(school_name)
    school_dir.mkdir(parents=True, exist_ok=True)

    # Count detail buttons using text selector; we'll click by index (DOM-safe)
    btn_count = await page.evaluate("""() => {
        return Array.from(document.querySelectorAll('*'))
            .filter(el => el.innerText === '详情' && el.children.length === 0)
            .length;
    }""")
    print(f"  找到 {btn_count} 个'详情'按钮")

    # Save the list page content
    (school_dir / "_list_page.md").write_text(
        f"# {school_name} - 培养方案列表\n\n**URL**: {full_url}\n\n---\n\n{body_text}",
        encoding="utf-8",
    )

    # Remember the list page body text length (before any clicks)
    list_page_len = len(body_text)
    plans = []

    for i in range(btn_count):
        try:
            # Click the nth detail button and extract content using pure JS
            # This avoids Playwright DOM staleness issues after drawer opens/closes
            result = await page.evaluate(f"""(async () => {{
                // Find all leaf "详情" elements
                const btns = Array.from(document.querySelectorAll('*'))
                    .filter(el => el.innerText === '详情' && el.children.length === 0);

                if ({i} >= btns.length) return null;

                const btn = btns[{i}];
                btn.scrollIntoView({{block: 'center'}});
                btn.click();
                await new Promise(r => setTimeout(r, 3000));

                // Try to get drawer/dialog content
                const selectors = [
                    '.el-drawer__body', '.el-dialog__body',
                    '[class*=drawer]', '[class*=detail]',
                ];
                for (let sel of selectors) {{
                    const d = document.querySelector(sel);
                    if (d && d.offsetParent !== null && d.innerText.trim().length > 50) {{
                        return d.innerText.trim();
                    }}
                }}
                return document.body.innerText;
            }})()""")

            if not result:
                continue

            # Get row context
            row_summary = await page.evaluate(f"""(async () => {{
                const btns = Array.from(document.querySelectorAll('*'))
                    .filter(el => el.innerText === '详情' && el.children.length === 0);
                if ({i} >= btns.length) return '';
                const el = btns[{i}];
                let p = el.parentElement;
                for (let j=0; j<5; j++) {{
                    if (!p) break;
                    let txt = (p.innerText || '').trim();
                    if (txt.length > 20 && txt.length < 500) return txt;
                    p = p.parentElement;
                }}
                return '';
            }})()""")

            print(f"  [{i+1}/{btn_count}] {row_summary[:80]}...")
            print(f"    debug: detail_len={len(result)}")

            # Save if we got new content (different from list page)
            if result and len(result) > list_page_len + 50:
                # Extract plan name from row summary
                plan_name = ""
                m = re.search(r'(\d{4}级[^\n]{0,60})', row_summary)
                if not m:
                    m = re.search(r'([^\n]{5,60}培养方案[^\n]{0,30})', row_summary)
                if m:
                    plan_name = m.group(1).strip()

                plans.append({
                    "summary": row_summary[:150],
                    "plan_name": plan_name or row_summary[:80],
                })

                fname = f"{i+1:02d}_{safe_name(plan_name or row_summary[:40])}.md"
                fp = school_dir / fname
                fp.write_text(
                    f"# {row_summary[:80]}\n\n"
                    f"**学校**: {school_name}\n\n"
                    f"---\n\n"
                    f"{result}",
                    encoding="utf-8",
                )
                print(f"    -> 保存: {fname} ({len(result)} chars)")
            else:
                print(f"    -> 无新内容 (got {len(result)} chars, list page was {list_page_len} chars)")

            # Close drawer via JS
            await page.evaluate("""() => {
                const closeBtn = document.querySelector('.el-drawer__close-btn, .el-dialog__close, .el-icon-close, [class*=close-btn]');
                if (closeBtn) closeBtn.click();
            }""")
            await asyncio.sleep(1.5)

        except Exception as e:
            print(f"    -> 错误: {str(e)[:80]}")
            await page.evaluate("""() => {
                const closeBtn = document.querySelector('.el-drawer__close-btn, .el-dialog__close, .el-icon-close, [class*=close-btn]');
                if (closeBtn) closeBtn.click();
            }""")
            await asyncio.sleep(1)
            await page.keyboard.press("Escape")
            await asyncio.sleep(1)

    return plans


async def main():
    all_results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
        )
        page = await ctx.new_page()

        if not await login(page):
            print("登录失败")
            await browser.close()
            return

        await asyncio.sleep(3)

        for hash_route, school_name in TARGETS:
            try:
                plans = await process_school(page, hash_route, school_name)
                all_results.append({
                    "school": school_name,
                    "hash": hash_route,
                    "plans_found": len(plans),
                })
            except Exception as e:
                print(f"  处理失败 {school_name}: {e}")
                import traceback
                traceback.print_exc()

        await browser.close()

    # Summary
    summary_file = OUT / "_summary.json"
    summary_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*50}")
    print(f"完成!")
    for r in all_results:
        print(f"  {r['school']}: {r['plans_found']} 个培养方案")
    print(f"  输出目录: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
