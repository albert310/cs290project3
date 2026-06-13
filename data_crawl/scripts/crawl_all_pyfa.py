#!/usr/bin/env python3
"""Login to graduate system and extract ALL training plans across all majors."""
import asyncio, re, json
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright

OUT = Path("/Users/leslie/Desktop/data/shanghaitech")

CAS_URL = "https://ids.shanghaitech.edu.cn/authserver/login?service=https%3A%2F%2Fgraduate.shanghaitech.edu.cn%2Fgsapp%2Fsys%2Fyjsemaphome%2Fportal%2Findex.do"

PYFA_URL = "https://graduate.shanghaitech.edu.cn/gsapp/sys/wdpyfaapp/*default/index.do#/pyfaxq"


def to_fname(text, idx):
    text = re.sub(r'[<>:"/\\|?*#\s]+', '_', text)[:80]
    return f"PYFA_{idx:03d}_{text}.md"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx = await browser.new_context(viewport={"width": 1920, "height": 1080}, locale="zh-CN")
        page = await ctx.new_page()

        # Step 1: Login
        print("=== 登录研究生系统 ===")
        await page.goto(CAS_URL, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)

        try:
            el = await page.query_selector("input#username,input[name='username']")
            if el and await el.is_visible():
                await el.fill("2025222203")
                print("已填充账号")
        except: pass
        try:
            el = await page.query_selector("input#password,input[name='password']")
            if el and await el.is_visible():
                await el.fill("vusjef-wupke6-heWjoj")
                print("已填充密码")
        except: pass

        print("请在浏览器中完成登录...")
        login_url = page.url
        for i in range(120):
            await asyncio.sleep(2)
            url = page.url
            if url != login_url and "ids.shanghaitech.edu.cn" not in url:
                print(f"登录成功！")
                break

        await asyncio.sleep(3)

        # Step 2: Navigate to training plan query
        print("\n=== 进入培养方案查询 ===")
        await page.goto(PYFA_URL, wait_until="networkidle", timeout=60000)
        await asyncio.sleep(5)
        print(f"当前页面: {await page.title()}")

        # Save initial screenshot
        await page.screenshot(path=str(OUT / "pyfa_initial.png"), full_page=True)

        all_plans = []

        # Step 3: Explore the page - find clickable elements
        print("\n=== 探索页面结构 ===")

        # Try to find dropdown/select for major selection
        selectors_to_try = [
            "select", "input[type='text']", ".el-select", ".el-input__inner",
            "[class*='select']", "[class*='dropdown']", "[class*='picker']",
            "input", "button", "[class*='tab']", "[class*='menu']",
            ".van-dropdown-menu", ".van-field",
        ]

        found_elements = []
        for sel in selectors_to_try:
            try:
                els = await page.query_selector_all(sel)
                for el in els:
                    try:
                        visible = await el.is_visible()
                        if visible:
                            tag = await el.evaluate("el => el.tagName")
                            cls = await el.evaluate("el => el.className")
                            ph = await el.evaluate("el => el.placeholder || ''")
                            txt = (await el.text_content()).strip()[:60]
                            rect = await el.bounding_box()
                            if rect:
                                found_elements.append({
                                    "selector": sel, "tag": tag, "class": str(cls)[:60],
                                    "text": txt, "placeholder": str(ph)[:40],
                                    "x": rect["x"], "y": rect["y"]
                                })
                    except: pass
            except: pass

        print(f"Found {len(found_elements)} interactive elements")
        for e in found_elements[:30]:
            print(f"  [{e['tag']}] {e['selector']} | text='{e['text'][:50]}' | ph='{e['placeholder']}' | ({e['x']:.0f},{e['y']:.0f})")

        # Step 4: Try to interact with dropdowns to select different majors
        print("\n=== 尝试选择不同专业 ===")

        # Try clicking elements that look like major selectors
        click_targets = [e for e in found_elements if
            any(kw in (e['text'] + e['placeholder']).lower()
                for kw in ['专业', '学科', 'major', '选择', '请选择', '年级', '学院'])]

        if not click_targets:
            # Try clicking all inputs
            click_targets = [e for e in found_elements if e['tag'] in ('INPUT', 'SELECT')]

        extracted_texts = set()

        for target in click_targets[:10]:
            try:
                x, y = target["x"], target["y"]
                print(f"\n点击: {target['text'][:50]} at ({x:.0f},{y:.0f})")
                await page.mouse.click(x + 10, y + 10)
                await asyncio.sleep(2)

                # Take screenshot after click
                ss_name = f"pyfa_click_{target['text'][:20]}.png"
                ss_name = re.sub(r'[<>:"/\\|?*]', '_', ss_name)
                await page.screenshot(path=str(OUT / ss_name), full_page=True)

                # Get new text content
                text = (await page.evaluate("document.body.innerText")).strip()
                if len(text) > 200 and text not in extracted_texts:
                    extracted_texts.add(text)
                    idx = len(all_plans)
                    fp = OUT / to_fname(target["text"], idx)
                    fp.write_text(f"# {target['text'][:80]}\n\n## Content\n\n{text}", encoding="utf-8")
                    all_plans.append({"trigger": target["text"][:80], "length": len(text)})
                    print(f"  获取到 {len(text)} chars 新内容")

                    # Look for sub-options (dropdown items)
                    opts = await page.evaluate("""() => {
                        return Array.from(document.querySelectorAll(
                            '.el-select-dropdown__item, .van-picker__option, option, ' +
                            '[class*=option], [class*=item], li[class*=item], .cell'
                        )).map(el => ({
                            text: el.textContent.trim().substring(0, 100),
                            visible: el.offsetParent !== null
                        }));
                    }""")

                    for o in opts[:20]:
                        if o["visible"] and len(o["text"]) > 1:
                            print(f"    选项: {o['text'][:80]}")

            except Exception as e:
                print(f"  错误: {str(e)[:60]}")

        # Step 5: Also try the "我的培养方案" page for the user's own plan
        print("\n=== 获取个人培养方案详情 ===")
        await page.goto("https://graduate.shanghaitech.edu.cn/gsapp/sys/wdpyfaapp/*default/index.do#/pyfa",
                       wait_until="networkidle", timeout=60000)
        await asyncio.sleep(5)
        text = (await page.evaluate("document.body.innerText")).strip()
        fp = OUT / "PYFA_个人培养方案.md"
        fp.write_text(f"# 个人培养方案\n\n## Content\n\n{text}", encoding="utf-8")
        await page.screenshot(path=str(OUT / "pyfa_personal.png"), full_page=True)
        print(f"个人培养方案: {len(text)} chars")

        # Save results
        result_file = OUT / "pyfa_summary.json"
        result_file.write_text(json.dumps(all_plans, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== 完成！提取了 {len(all_plans)} 个培养方案 ===")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
