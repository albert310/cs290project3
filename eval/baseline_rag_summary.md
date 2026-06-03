# Baseline RAG validation summary

Date: 2026-06-02

This historical validation run used only preprocessed plain-text files under
`data/sist/texts`. It does not read `raw/`, structured JSONL tables, or external
web pages at inference time.

Pipeline:

```text
user query -> SQLite FTS5 lexical retrieval over text chunks -> top-k context
-> local Qwen generation -> automatic answer scoring
```

Configuration:

- Text source: `data/sist/texts`
- Retriever: local SQLite FTS5 over pre-tokenized Chinese/English text
- Chunk size: 900 chars
- Chunk overlap: 120 chars
- `top_k`: 6
- Generator: local `qwen3.6-27b`
- `max_tokens`: 220
- Thinking: disabled for this historical run; current RAG defaults enable
  thinking.

Overall result on `eval/testset_web_verified.jsonl`:

| metric | value |
|---|---:|
| questions | 100 |
| correct before optimization | 27 |
| accuracy before optimization | 0.270 |
| average latency | 2.072 s |
| max latency | 4.964 s |

Accuracy by question type:

| question_type | correct / total | accuracy |
|---|---:|---:|
| factual | 19 / 50 | 0.380 |
| comparative | 4 / 10 | 0.400 |
| conditional | 2 / 10 | 0.200 |
| multi-hop | 0 / 10 | 0.000 |
| time_sensitive | 2 / 10 | 0.200 |
| negative_refusal | 0 / 10 | 0.000 |

Accuracy by category:

| category | correct / total | accuracy |
|---|---:|---:|
| university_profile | 12 / 26 | 0.462 |
| sist_profile | 2 / 15 | 0.133 |
| graduate_admission | 11 / 18 | 0.611 |
| course_catalog | 1 / 12 | 0.083 |
| faculty_profile | 1 / 19 | 0.053 |
| negative_refusal | 0 / 10 | 0.000 |

Main observed weaknesses:

- Plain lexical retrieval often ranks old or less authoritative pages above
  newer pages, e.g. old campus-area text conflicts with newer `900亩` facts.
- Course and faculty questions perform poorly because the baseline ignores the
  structured tables and only indexes plain text.
- Multi-hop questions fail because retrieved chunks are not coordinated across
  course pages and faculty profile pages.
- Negative/refusal questions fail because lexical retrieval still returns
  partially related chunks for common terms such as `学院` or `课程`.
- Some semantically acceptable answers fail strict automatic scoring because the
  baseline omits required key phrases.

Useful next optimizations:

- Add structured-course/faculty records or metadata mapping while keeping text
  chunks as the main corpus.
- Add query expansion for `任课老师/instructor`, `学分/credit`, `邮箱/email`.
- Add a simple entity-existence guard for course codes, research centers,
  schools, and faculty names.
- Rerank retrieved chunks using exact phrase/course-code matches and recency or
  authority heuristics.
- Add a multi-hop retrieval pass for course-teacher and teacher-profile links.
