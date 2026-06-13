#!/usr/bin/env python3
"""Extract ALL professors from ShanghaiTech schools via WebPlus CMS API."""
import requests
import json
import asyncio
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

OUTPUT_DIR = Path("/Users/leslie/Desktop/data/shanghaitech")
PROF_DATA_FILE = Path("/Users/leslie/Desktop/data/professors.json")

# School definitions: (subdomain, siteId, name)
SCHOOLS = [
    ("spst.shanghaitech.edu.cn", "siteId=46", "物质科学与技术学院"),
    ("slst.shanghaitech.edu.cn", "siteId=23", "生命科学与技术学院"),
    ("sist.shanghaitech.edu.cn", "siteId=43", "信息科学与技术学院"),
    ("sem.shanghaitech.edu.cn", "siteId=55", "创业与管理学院"),
    ("sca.shanghaitech.edu.cn", "siteId=66", "创意与艺术学院"),
    ("ih.shanghaitech.edu.cn", "siteId=129", "人文科学研究院"),
    ("bme.shanghaitech.edu.cn", "siteId=95", "生物医学工程学院"),
    ("siais.shanghaitech.edu.cn", "siteId=41", "免疫化学研究所"),
    ("ihuman.shanghaitech.edu.cn", "siteId=123", "iHuman研究所"),
    ("ims.shanghaitech.edu.cn", "siteId=34", "数学科学研究所"),
    ("cts.shanghaitech.edu.cn", "siteId=101", "大科学中心"),
    ("smdl.shanghaitech.edu.cn", "siteId=137", "材料器件中心"),
]

API_PATH = "/_wp3services/generalQuery?queryObj=teacherHome"

POST_CONDITIONS = json.dumps([
    {"field": "published", "value": "1", "judge": "="},
    {"field": "language", "value": "1", "judge": "="},
])

POST_RETURN = json.dumps([
    {"field": "title", "name": "title"},
    {"field": "headerPic", "name": "headerPic"},
    {"field": "career", "name": "career"},
    {"field": "cnUrl", "name": "cnUrl"},
    {"field": "phone", "name": "phone"},
    {"field": "email", "name": "email"},
    {"field": "exField1", "name": "exField1"},
    {"field": "exField2", "name": "exField2"},
    {"field": "exField3", "name": "exField3"},
    {"field": "exField4", "name": "exField4"},
    {"field": "exField6", "name": "exField6"},
    {"field": "exField7", "name": "exField7"},
    {"field": "exField8", "name": "exField8"},
])

POST_ORDERS = json.dumps([{"field": "exField6", "type": "asc"}])


def fetch_professors(subdomain, site_param):
    """Fetch all professors for a school via API."""
    url = f"https://{subdomain}{API_PATH}"
    data = {
        site_param.split("=")[0]: site_param.split("=")[1],
        "pageIndex": "1",
        "rows": "500",
        "conditions": POST_CONDITIONS,
        "orders": POST_ORDERS,
        "returnInfos": POST_RETURN,
        "articleType": "1",
        "level": "1",
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    try:
        resp = requests.post(url, data=data, headers=headers, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            return result.get("data", [])
        else:
            print(f"  HTTP {resp.status_code} for {subdomain}")
            return []
    except Exception as e:
        print(f"  Error: {e}")
        return []


def normalize(url):
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc, path, p.params, p.query, ""))


def url_to_filename(url):
    p = urlparse(url)
    netloc = p.netloc.replace(":", "_")
    path = p.path.strip("/") or "index"
    path = re.sub(r'[<>:"/\\|?*\s]+', '_', path)
    if len(path) > 150:
        path = path[:150]
    if p.query:
        q = re.sub(r'[<>:"/\\|?*]', '_', p.query)[:40]
        path = f"{path}__{q}"
    return f"{netloc}/{path}.md"


async def crawl_professor_pages(all_professors):
    """Crawl individual professor profile pages."""
    config = CrawlerRunConfig(page_timeout=30000, delay_before_return_html=2.0)

    to_crawl = []
    for prof in all_professors:
        url = prof.get("cnUrl", "")
        if url and "shanghaitech.edu.cn" in url:
            to_crawl.append((url, prof.get("title", "unknown"), prof.get("school", "")))

    print(f"\nCrawling {len(to_crawl)} individual professor pages...")
    crawled = 0
    async with AsyncWebCrawler() as crawler:
        for url, name, school in to_crawl:
            filename = url_to_filename(url)
            filepath = OUTPUT_DIR / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if filepath.exists():
                crawled += 1
                continue

            try:
                result = await crawler.arun(url, config=config)
                if result and result.markdown:
                    header = f"# {name}\n\n**学院**: {school}\n**URL**: {url}\n\n---\n\n"
                    filepath.write_text(header + result.markdown, encoding="utf-8")
                    crawled += 1
                    if crawled % 50 == 0:
                        print(f"  [{crawled}/{len(to_crawl)} profiles crawled]", flush=True)
            except Exception as e:
                pass

    print(f"  Done: {crawled} profiles saved.", flush=True)


def main():
    all_professors = []

    for subdomain, site_param, school_name in SCHOOLS:
        print(f"\nFetching: {school_name} ({subdomain})...", flush=True)
        professors = fetch_professors(subdomain, site_param)
        for p in professors:
            p["school"] = school_name
            p["subdomain"] = subdomain
        all_professors.extend(professors)
        print(f"  -> {len(professors)} professors", flush=True)

    print(f"\n=== TOTAL: {len(all_professors)} professors across {len(SCHOOLS)} schools ===")

    # Save as JSON
    with open(PROF_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_professors, f, ensure_ascii=False, indent=2)
    print(f"Saved to {PROF_DATA_FILE}")

    # Crawl individual professor pages
    asyncio.run(crawl_professor_pages(all_professors))


if __name__ == "__main__":
    main()
