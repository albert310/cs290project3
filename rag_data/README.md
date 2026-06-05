# Clean ShanghaiTech/SIST RAG Data

This directory contains a clean, independent RAG database for questions about
ShanghaiTech University and the School of Information Science and Technology
(SIST). It is separate from the older `data/rag/knowledge.sqlite` database.

## Current Database

- SQLite database: `rag_data/db/shanghaitech_sist.sqlite`
- FTS table: `chunks_fts`
- Main content table: `chunks`
- Build report: `rag_data/reports/build_report.md`
- Verification report: `rag_data/reports/verification_report.md`

The current build contains:

- 2069 source documents
- 10025 searchable chunks
- 0 chunks without URL
- Source tiers:
  - `verified_seed`: manually web-checked official facts
  - `live_official`: pages crawled directly from official sites
  - `local_official_mirror`: cleaned pages from the course-provided official SIST mirror

## Data Scope

The database focuses on official, directly traceable material:

- ShanghaiTech overview and contact information
- SIST overview
- SIST faculty profiles
- SIST courses and course schedules
- undergraduate and graduate degree programs
- research, seminars, news, and official notices

The first build intentionally excludes broad PDF dumps, external academic
papers, static assets, pages with no URL, and obvious navigation/copyright
noise. Every searchable chunk stores `source_url`, `source_tier`, `category`,
`quality`, `date`, and `source_path`.

## Build Pipeline

Run from the project root:

```bash
python3 rag_data/scripts/crawl_official.py --max-pages 80 --timeout 10 --sleep 0.15
python3 rag_data/scripts/build_database.py
python3 rag_data/scripts/verify_database.py --top-k 8
```

The crawler reads `rag_data/config/seeds.json` and writes:

- raw HTML: `rag_data/raw/official_pages/`
- crawl manifest: `rag_data/processed/official_crawl_manifest.jsonl`
- crawl summary: `rag_data/reports/crawl_summary.json`

The builder merges three sources:

- `rag_data/config/verified_seed_facts.json`
- crawled official pages
- local official mirror pages from `data/sist/jsonl/documents.jsonl` and `data/sist/texts`

## Search

Use the standalone search script:

```bash
python3 rag_data/scripts/search_db.py 'CS282 机器学习 春季 任课教师' --top-k 5
python3 rag_data/scripts/search_db.py '王浩宇 教授 邮箱 办公室 研究方向' --top-k 5
python3 rag_data/scripts/search_db.py '2024 Bachelor Degree Programs CS 信息学院 培养方案' --top-k 5
```

JSON output:

```bash
python3 rag_data/scripts/search_db.py '上海科技大学由谁共同举办建立' --json
```

The search layer uses SQLite FTS plus intent-aware supplemental recall. This is
important for long official pages such as degree programs, where exact URL/year
evidence can be more reliable than raw BM25 ranking.

## Verification

`rag_data/config/verified_seed_facts.json` stores a small set of web-checked
official facts. The current verification result is:

```text
7/7 passed
```

Covered checks include:

- ShanghaiTech was jointly founded by Shanghai Municipal Government and Chinese
  Academy of Sciences.
- ShanghaiTech was officially established in 2013.
- The Pudong campus address includes 393 Huaxia Middle Road.
- SIST means School of Information Science and Technology.
- SIST's 2024-2025 courses can be retrieved from the official course page.
- Wang Haoyu's official profile retrieves `wanghy@shanghaitech.edu.cn`.

## Optional LLM Curation

This first clean build does not require large-scale LLM processing. The main
failure mode of the previous database was mixed-source noise, missing URLs,
navigation text, and weak source ranking; deterministic filtering fixes most of
that without spending API calls.

If a later pass needs LLM curation, use `/2022533109/chenyuhan/evas/gpt_api`
with model `gpt-5.4-mini` only for low-confidence or conflicting records, for
example:

- identify whether a page is relevant to ShanghaiTech/SIST
- summarize a very long page into evidence-focused chunks
- classify ambiguous pages into `sist_faculty`, `sist_courses`, or
  `sist_degree_programs`
- flag conflicting facts for manual review

The recommended policy is not to ask the LLM to invent facts. It should only
filter, classify, or compress evidence while preserving the original
`source_url`.

## Suggested RAG Integration

For the baseline RAG system, replace the old DB path with:

```text
rag_data/db/shanghaitech_sist.sqlite
```

Use `rag_data/scripts/search_db.py` as the reference retrieval behavior. In code,
the fields needed for prompt construction are:

- `title`
- `category`
- `source_tier`
- `quality`
- `source_url`
- `text`

Prompt source headers should include the URL and source tier so the model can
prefer official evidence and reject unsupported answers.
