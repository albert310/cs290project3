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
