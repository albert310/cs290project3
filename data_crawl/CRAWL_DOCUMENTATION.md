# ShanghaiTech University Data Crawling Project

## Skill Used

We used the **[crawl4ai](https://github.com/unclecode/crawl4ai)** skill for web crawling and data extraction.

- **GitHub**: [https://github.com/unclecode/crawl4ai](https://github.com/unclecode/crawl4ai)
- **Skill Documentation**: [https://github.com/brettdavies/crawl4ai-skill](https://github.com/brettdavies/crawl4ai-skill)
- **Version Used**: crawl4ai 0.8.6
- **Key Capabilities**:
  - JavaScript rendering via Playwright (handles SPAs and dynamic content)
  - Markdown generation from HTML
  - Concurrent crawling with `arun_many`
  - Session/Cookie management for authenticated pages
  - CSS selector and LLM-based data extraction

### Supporting Tools

| Tool | Purpose |
|------|---------|
| **Playwright** (`playwright 1.60.0`) | Direct browser automation for CAS login and SPA interaction |
| **playwright-stealth** (`2.0.3`) | Anti-detection measures for bot-protected pages |
| **MarkItDown** (`markitdown[pdf]`) | PDF to Markdown conversion |
| **BeautifulSoup4** | Static HTML parsing and text extraction |
| **requests** | HTTP fallback when Playwright was blocked |
| **PyPDF2** | PDF text extraction |
| **SQLite** | Structured knowledge base storage (SIST data) |

---

## Scripts Built

All scripts are in `scripts/`. Below is what each one does:

### Core Crawlers

| Script | Purpose |
|--------|---------|
| `crawl_multi.py` | **Main production crawler** — 48 parallel AsyncWebCrawler instances, depth 6, with resume support and SHA256 deduplication. Discovers and crawls all reachable pages across all subdomains. |
| `crawl_fast.py` | Fast crawler using `arun_many` batch processing for higher throughput with a single browser instance. |
| `crawl_parallel.py` | Multi-process crawler using Python multiprocessing (experimental). |
| `crawl_recursive.py` | Original recursive crawler with depth 3 (first version). |
| `crawl_recursive2.py` | Improved recursive crawler with pickle state saving for resume capability. |
| `crawl_worker.py` | Single worker process that reads URLs from a queue file, crawls them, and saves results. |

### Targeted Crawlers

| Script | Purpose |
|--------|---------|
| `crawl_prof_pages.py` | Crawls individual professor profile pages from URLs in `professors.json`. Saved 183 detailed professor homepages. |
| `crawl_leadership.py` | Playwright-based crawler for JS-rendered leadership/administration pages across all 12 schools. |
| `crawl_leadership2.py` | Brute-force URL pattern search (18 patterns × 6 schools) to find hidden leadership pages. |
| `crawl_targeted.py` | Targeted crawl of OAA academic affairs pages: course catalogs, training plans, schedules, news. |
| `crawl_training_plans.py` | Crawls graduate training plan detail pages from the pyfacx system. |
| `crawl_text_files.py` | BFS crawl (depth 5) specifically hunting for `.doc`, `.docx`, `.xls`, `.xlsx`, `.ppt`, `.pptx`, `.csv`, `.txt` files. |
| `crawl_text_recursive.py` | Recursive text file crawler with state persistence. |

### Authentication Crawlers

| Script | Purpose |
|--------|---------|
| `crawl_auth.py` | Attempts CAS authentication via JavaScript injection (crawl4ai `js_code`). |
| `crawl_auth_login.py` | Opens visible browser for manual CAS login, then crawls authenticated OAA pages. |
| `crawl_cas_login.py` | CAS login + automatic post-login crawl of course selection system. |
| `crawl_manual_auth.py` | Manual browser login with auto-detection of login completion. |
| `crawl_grad.py` | CAS login + graduate training plan system crawling (gsapp). |

### OAA & Course Data

| Script | Purpose |
|--------|---------|
| `crawl_all_pyfa.py` | Clicks through all graduate training plan query options to extract plans per school. |
| `crawl_oaa_deep.py` | Deep crawl of OAA article pages: course schedules, training plans, requirements. |

### Data Extraction

| Script | Purpose |
|--------|---------|
| `extract_professors.py` | Calls the WebPlus CMS API (`_wp3services/generalQuery`) to extract structured professor data (name, email, career, URL) for all 12 schools. |

---

## Data Collected

### Final Output: `/Users/leslie/Desktop/data/shanghaitech_data/`

| Directory | Files | Size | Content |
|-----------|-------|------|---------|
| `web/` | 5,832 | 78 MB | Cleaned web pages as Markdown (105 subdomains) |
| `pdf_md/` | 1,281 | 57 MB | PDF files converted to Markdown |
| `sist_raw/` | 2,194 | 10 MB | SIST knowledge base raw HTML → MD |
| `text_pages/` | 318 | 4 MB | Text-only extracted pages |
| `text/` | 127 | 20 MB | Plain text documents |
| `text_files/` | 22 | 3 MB | Downloaded Office files (.docx, .xlsx, .xls) |
| `text_md/` | 6 | 1 MB | Office files converted to Markdown |
| `training/` | 42 | 1 MB | Graduate training plan detail pages |
| `data/` | 15 | 120 MB | Structured data (JSON + SQLite) |

| File Type | Count |
|-----------|-------|
| `.md` (Markdown) | **7,474** |
| `.png` (screenshots) | 8 |
| `.json` (structured data) | 2 |
| `.sqlite` (knowledge base) | 1 |

### PDF Files

| Location | Files | Size |
|----------|-------|------|
| `shanghaitech_pdfs/` | **1,447** | **2.9 GB** |

### Structured Data (in `data/`)

| File | Records | Description |
|------|---------|-------------|
| `professors_enriched.json` | 419 | Faculty: names, emails, titles, research areas (57% coverage) |
| `courses_clean.json` | 552 | Courses with teacher mappings (90% have instructors) |
| `course_schedule.json` | 538 | Course schedules with time, classroom, instructor |
| `sist_kb.sqlite` | 118 MB | SIST knowledge base: 2,110 courses, 103 faculty, 15,048 requirements, 4,065 events |

### Coverage

- **Subdomains crawled**: 186
- **Web pages crawled**: 6,054 raw → 3,819 after cleaning
- **Total pages scanned during crawl**: ~23,099
- **Crawl time**: ~8 hours total (across all phases)
- **Total text characters**: ~91 million

---

## Crawl Configuration

### Main Production Crawler (`crawl_multi.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_WORKERS` | 48 | Parallel browser instances |
| `MAX_DEPTH` | 6 | Maximum link depth from start page |
| `MAX_PAGES` | 50,000 | Upper limit (queue exhausted before reaching) |
| `BATCH_SIZE` | 5 | URLs per worker per iteration |
| `HARD_TIMEOUT` | 20s | Maximum wait per page |
| `delay_before_return_html` | 3s | Extra wait for JS content |
| `scan_full_page` | true | Scroll to trigger lazy loading |

### Text File Crawler (`crawl_text_files.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_DEPTH` | 5 | BFS depth from starting page |
| Target Extensions | `.doc`, `.docx`, `.xls`, `.xlsx`, `.ppt`, `.pptx`, `.csv`, `.txt` | Downloadable document files |

---

## Manually Downloaded Content

The following data required manual browser interaction (CAS login):

1. **Graduate Training Plans** (via `https://graduate.shanghaitech.edu.cn/gsapp/sys/pyfacxapp/`)
   - 7 schools' training plan lists with program counts
   - 42 individual training plan detail pages
   - Requires CAS SSO authentication

2. **Course Selection System** (via `https://oaa.shanghaitech.edu.cn/`)
   - 116 course-related articles (training plans, schedules, course recommendations)
   - 12 semesters of course schedules (2021-2026)
   - Content loaded via CMS article system

3. **Leadership Pages** (via Playwright rendering)
   - 8/12 schools' leadership information successfully extracted
   - 4 schools (SIST, SEM, BME, SMDL) blocked automated access — manually accessed

4. **SIST Dean's Message** and **SEM Dean's Message**
   - Manually copied from browser due to server blocking

### File Types Downloaded

| Extension | Count | Description |
|-----------|-------|-------------|
| `.pdf` | 1,447 | Course handbooks, schedules, forms, publications |
| `.md` (converted) | 7,474 | All web pages and PDFs converted to Markdown |
| `.docx` | 10 | Word documents (mostly forms and templates) |
| `.xlsx` | 6 | Excel spreadsheets |
| `.xls` | 4 | Legacy Excel files |
| `.doc` | 3 | Legacy Word documents |
| `.txt` | 1 | Plain text file |
| `.json` | 9 | Structured data files |
| `.sqlite` | 1 | Knowledge base database |

### Total Data Size

| Category | Size |
|----------|------|
| Web Markdown | 78 MB |
| PDF → Markdown | 57 MB |
| SIST raw → Markdown | 10 MB |
| Other text/MD | 28 MB |
| Structured data (JSON) | 1 MB |
| SQLite database | 120 MB |
| **Subtotal (text/searchable)** | **~294 MB** |
| Original PDF files | 2.9 GB |
| **Grand Total** | **~3.2 GB** |
