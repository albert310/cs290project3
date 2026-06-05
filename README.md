# cs290project3
cs290project3

## Provided SIST dataset

The course dataset archive is kept under `data/sist.zip` and extracted to
`data/sist/`. The whole `data/` directory is ignored by Git because raw data
must not be included in the final submission zip.

Dataset layout:

```text
data/sist/
  jsonl/   structured JSONL tables and ready-to-index RAG chunks
  raw/     original crawled HTML/PDF/PSP/TXT/ASP files
  texts/   extracted plain-text files referenced by documents.jsonl
```

Current extracted size is about 630 MB. The raw directory contains mostly HTML
pages and PDFs:

| extension | files |
|---|---:|
| htm | 1737 |
| html | 364 |
| pdf | 201 |
| psp | 96 |
| txt | 2 |
| asp | 1 |

JSONL tables:

| table | rows | use |
|---|---:|---|
| `chunks` | 23146 | Main RAG indexing table. Each row has `id`, `document_id`, `chunk_index`, `title`, `url`, `category`, `text`, `char_count`. |
| `documents` | 2566 | Source document metadata, including URL, title, language, category, raw path, text path, timestamps. |
| `facts` | 19495 | Extracted subject-predicate-object style facts. |
| `program_requirements` | 15048 | Extracted degree/program requirement snippets. |
| `events` | 4065 | News/event-like records. |
| `courses` | 2110 | Course-like records. |
| `leadership_roles` | 2114 | Leadership or role records. |
| `entities` | 2130 | Extracted entities. |
| `program_sources` | 1965 | Program source pages/snippets. |
| `contacts` | 741 | Contact-like records. |
| `faculty_members` | 103 | Faculty profile records. |
| `facilities` | 42 | Facility/procurement-like records. |
| `staff_members` | 2 | Staff records. |
| `crawl_runs` | 1 | Crawl metadata. |

Chunk categories include `program`, `faculty`, `leadership`, `news`,
`admission`, `career`, `international`, `research`, `campus_life`, `facility`,
`pdf`, and several smaller categories. There are no missing `document_id`
references from `chunks.jsonl` to `documents.jsonl`.

### Dataset loader

Use the zero-dependency `sist_data` package to inspect and load the dataset.

```bash
PYTHONPATH=. python3 -m sist_data.loader --summary
PYTHONPATH=. python3 -m sist_data.loader --table chunks --limit 2
PYTHONPATH=. python3 -m sist_data.loader --rag-sample --limit 2
```

Python usage:

```python
from sist_data import SISTDataset

dataset = SISTDataset("data/sist")

# Iterate normalized records for embedding/indexing.
for record in dataset.iter_rag_records(categories=["program", "faculty"]):
    text = record["text"]
    metadata = record["metadata"]

# Load any structured table.
courses = dataset.load_jsonl("courses", limit=10)

# Resolve source metadata and extracted text.
doc = dataset.get_document(54)
text = dataset.read_document_text(54)
```

## Unified cleaned RAG database

`scripts/build_rag_database.py` builds a cleaned SQLite knowledge store from the
course-provided `data/sist` dataset and the self-crawled
`data/shanghaitech_data` dataset.

```bash
python3 scripts/build_rag_database.py
```

Outputs:

```text
data/rag/knowledge.sqlite      unified SQLite database
data/rag/build_report.md       build counts and source/category distribution
data/rag/review_samples.md     deterministic samples for manual inspection
data/rag/manual_review.md      manual review notes and caveats
```

Current reviewed build:

| item | count |
|---|---:|
| documents | 31,217 |
| chunks | 46,199 |
| structured records | 15,458 |
| FTS rows | 46,199 |

SQLite schema:

| table | use |
|---|---|
| `documents` | one row per cleaned source document or structured record document |
| `chunks` | retrievable RAG chunks with source metadata, category, URL, host, date, quality score, and content hash |
| `chunks_fts` | FTS5 index over tokenized title/text/metadata for lexical retrieval |
| `structured_records` | raw JSON and metadata for structured rows linked to `chunks.id` |
| `metadata` | build summary and schema version |
| `build_events` | reserved for build diagnostics |

The builder keeps high-value sources (`training`, `sist_text`,
`structured_table`, `structured_json`, official web/text pages, and relevant
PDFs) and filters common crawler noise: navigation/share widgets, static assets,
JS/CSS chunks, mojibake, low-confidence structured rows, irrelevant PDFs, and
known teacher-field parsing artifacts. Manual review details are in
`data/rag/manual_review.md`.

This database is intended for the next RAG version. Retrieval should combine FTS
with structured hard filters/reranking for course codes, people, dates, and
program-year constraints; plain OR-only FTS is not reliable enough for final QA.

## Keyword retrieval baseline

The `retrieval.keyword_search` module implements a zero-dependency BM25F
keyword retriever over `chunks.jsonl` plus selected structured JSONL tables
(`contacts`, `courses`, `faculty_members`, `program_requirements`, etc.).
Use `--chunks-only` when you want a pure chunk baseline.

It uses mixed Chinese/English tokenization:

- English words and course-like codes are lowercased as normal tokens.
- Chinese text contributes character unigrams, bigrams, and trigrams.
- BM25F field weights are `title=3.0`, `text=1.0`, `category=0.8`, `url=0.2`.

A small Chinese/English query expansion map is available as an optional
experiment, e.g. `邮箱 -> email`, `学分 -> credits`, `任课老师 -> instructor`,
`信息学院 -> SIST`. Keep it off for the baseline and enable it with `--expand`
when measuring query expansion as an optimization.

CLI:

```bash
PYTHONPATH=. python3 -m retrieval.keyword_search "深度学习 任课老师" --top-k 5
PYTHONPATH=. python3 -m retrieval.keyword_search "计算机科学与技术 毕业 学分" --top-k 5
PYTHONPATH=. python3 -m retrieval.keyword_search "屠可伟 邮箱" --top-k 5
PYTHONPATH=. python3 -m retrieval.keyword_search "屠可伟 邮箱" --chunks-only --top-k 5
PYTHONPATH=. python3 -m retrieval.keyword_search "屠可伟 邮箱" --expand --top-k 5
```

Python:

```python
from sist_data import SISTDataset
from retrieval import BM25FIndex

dataset = SISTDataset("data/sist")
index = BM25FIndex.from_dataset(dataset)
hits = index.search("深度学习 任课老师", top_k=5)

for hit in hits:
    print(hit.score, hit.metadata["title"], hit.metadata["url"])
```

## Web-verified evaluation set

`eval/testset_web_verified.jsonl` contains 100 ShanghaiTech/SIST questions built
from public official web pages, independent of the provided `data/sist`
archive. The set covers university profile facts, SIST profile/admission facts,
course-catalog questions, faculty-profile questions, comparative questions,
conditional questions, multi-hop questions, time-sensitive facts, and
negative-refusal questions where the user asks for nonexistent information.

Current question type distribution is `factual=50`, `comparative=10`,
`conditional=10`, `multi-hop=10`, `time_sensitive=10`, and
`negative_refusal=10`.

Regenerate, validate, source-check, and export the project-required result
table:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 eval/build_web_testset.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate_testset.py --testset eval/testset_web_verified.jsonl --validate
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_testset_sources.py --testset eval/testset_web_verified.jsonl
PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate_testset.py --testset eval/testset_web_verified.jsonl --export-csv eval/testset_web_verified.csv
```

After filling model outputs into `sys_resp_before_opt` and
`sys_resp_after_opt`, score the table automatically:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate_testset.py --testset eval/testset_web_verified.jsonl --answers-csv eval/testset_web_verified.csv --output-csv eval/testset_scored.csv
```

See `eval/README.md` for the schema, sources, and evaluation rules.

## Text-only baseline RAG

The first baseline RAG system uses only preprocessed plain-text files under
`data/sist/texts`. It builds a local SQLite FTS5 index over text chunks, retrieves
the top-k chunks, inserts them into a grounded prompt, and asks the local Qwen
server to answer with thinking enabled by default.

By default the baseline does not pass `max_tokens` to the local Qwen server.
Generation therefore stops on the model/server stopping condition rather than a
project-side answer-length cap. You can still set a cap manually with
`--max-tokens N` when running the scripts.

Run one question:

```bash
python3 scripts/run_baseline_rag.py "上海科技大学校园占地约多少亩？" --show-context
```

Run validation on the 100-question test set:

```bash
python3 scripts/eval_baseline_rag.py \
  --testset eval/testset_web_verified.jsonl \
  --output-csv eval/baseline_rag_before_opt.csv \
  --top-k 6
```

The historical before-optimization validation result is `27/100 = 0.270`. See
`eval/baseline_rag_summary.md` and `eval/baseline_rag_before_opt.csv` for the
detailed result table and failure breakdown.

## Unified RAG and optional query-keyword planning

The unified RAG system reads `data/rag/knowledge.sqlite`. By default, it uses the
original user question directly for retrieval. The first optional optimization is
LLM query-keyword planning: before retrieval, local Qwen reads the user question
and outputs short search keywords. The retriever then searches with:

```text
original question + generated keywords
```

This optimization is off by default and can be enabled per run:

```bash
python3 scripts/run_unified_rag.py \
  "CS282春季任课教师王浩的研究方向是什么？" \
  --llm-query-keywords \
  --show-keywords \
  --show-context
```

Evaluate with the switch enabled:

```bash
python3 scripts/eval_unified_rag.py \
  --output-csv eval/unified_rag_llm_keywords.csv \
  --top-k 8 \
  --llm-query-keywords
```

The output CSV includes `search_query`, `llm_query_keywords`,
`llm_query_keyword_raw`, and `llm_query_keyword_error` so the optimization can be
audited and compared against runs where the switch is disabled.

The second optional optimization is iterative search rollout. After the first
retrieval, local Qwen receives the user question and current evidence, then
chooses either to stop searching or to request another search with explicit
keywords. The system allows at most five model-requested searches; after that,
the final answer must be generated or refused from the accumulated evidence.

```bash
python3 scripts/run_unified_rag.py \
  "CS282春季任课教师王浩的研究方向是什么？" \
  --iterative-search \
  --show-rollout \
  --show-context
```

Evaluate the rollout version:

```bash
python3 scripts/eval_unified_rag.py \
  --output-csv eval/unified_rag_rollout.csv \
  --top-k 8 \
  --iterative-search \
  --max-search-steps 5
```

The third optional optimization is answer verification. The system first drafts
an answer from the current evidence, asks local Qwen to extract verification
keywords from that draft, retrieves again, and then generates the final answer
after cross-checking the original and verification evidence. The verifier also
searches a course-code-stripped version of the verification query so course
codes do not block faculty profile or homepage evidence.

```bash
python3 scripts/run_unified_rag.py \
  "CS282春季任课教师王浩的博士学位来自哪所大学、哪一年？" \
  --verify-answer \
  --show-verification \
  --show-context
```

Evaluate the verification version:

```bash
python3 scripts/eval_unified_rag.py \
  --output-csv eval/unified_rag_verify_answer.csv \
  --top-k 8 \
  --verify-answer
```

The output CSV includes `answer_verification`, which records the draft answer,
verification keywords, verification search query, hit count, and number of new
evidence chunks. `--llm-query-keywords`, `--iterative-search`, and
`--verify-answer` are independent switches and can be enabled together.

## Web chat UI

The web chat frontend is under `web/`. The TypeScript source is
`web/src/app.ts`, and `web/static/app.js` is the browser-ready build used by the
server. The local server serves both the frontend and the RAG API. By default it
uses the unified SQLite RAG database. The web demo uses `/api/chat/stream` for
streaming output: optional `query_keywords` events are rendered as search terms,
`think_delta` is rendered into the expandable thinking block, and `answer_delta`
is rendered into the answer bubble.

```bash
python3 scripts/serve_web_chat.py --host 127.0.0.1 --port 7860
python3 scripts/serve_web_chat.py --host 127.0.0.1 --port 7860 --llm-query-keywords
python3 scripts/serve_web_chat.py --host 127.0.0.1 --port 7860 --iterative-search
python3 scripts/serve_web_chat.py --host 127.0.0.1 --port 7860 --verify-answer
```

Then open:

```text
http://127.0.0.1:7860
```

## Local Qwen API wrapper

The `qwen_api` package wraps the local OpenAI-compatible vLLM server and splits
model output into `think` and final `answer` parts.

Non-streaming:

```python
from qwen_api import QwenClient

client = QwenClient()
result = client.chat("2+3 equals what? Answer with one number.")
print(result.answer)
print(result.think)
print(result.raw)
```

Streaming:

```python
from qwen_api import QwenClient

client = QwenClient()
for event in client.stream_chat("2+3 equals what? Answer with one number."):
    if event.kind == "answer":
        print(event.delta, end="", flush=True)
```

Convenience helpers:

```python
client.ask("2+3 equals what?")
client.stream_answer("2+3 equals what?")
client.stream_think("2+3 equals what?")
```

Defaults can be overridden with environment variables:

```bash
export QWEN_BASE_URL=http://127.0.0.1:8000
export QWEN_MODEL=qwen3.6-27b
```

Demo:

```bash
PYTHONPATH=. python3 scripts/demo_qwen_api.py
PYTHONPATH=. python3 scripts/demo_qwen_api.py --stream
PYTHONPATH=. python3 scripts/demo_qwen_api.py --show-think
```
