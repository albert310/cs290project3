# Web-verified evaluation set

`testset_web_verified.jsonl` is a 100-question evaluation set for the
ShanghaiTech/SIST RAG system. It is intentionally independent of the provided
`data/sist` archive: questions and ground-truth answers were built from public
official web pages checked on 2026-06-02.

Question type distribution:

| question_type | count | meaning |
|---|---:|---|
| `factual` | 50 | ordinary factual QA |
| `comparative` | 10 | compare two or more official facts |
| `conditional` | 10 | answer under a user condition |
| `multi-hop` | 10 | combine facts from multiple official pages |
| `time_sensitive` | 10 | facts with an explicit timestamp or as-of date |
| `negative_refusal` | 10 | nonexistent user premise; system should refuse to fabricate |

## Sources

Main official sources:

- ShanghaiTech University profile:
  https://www.shanghaitech.edu.cn/1054/main.psp
- SIST English "About SIST":
  https://sist.shanghaitech.edu.cn/sist_en/2724/list.psp
- SIST graduate admission page:
  https://sist.shanghaitech.edu.cn/yjszs/
- SIST 2025-2026 course catalog:
  https://faculty.sist.shanghaitech.edu.cn/office/Academics/Courses/SIST_2025-2026_Courses.htm
- Kewei Tu faculty page:
  https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/
- Hao Wang faculty page:
  https://faculty.sist.shanghaitech.edu.cn/faculty/wanghao/
- Xuming He faculty page:
  https://faculty.sist.shanghaitech.edu.cn/faculty/hexm/index.html
- Jingya Wang faculty page:
  https://faculty.sist.shanghaitech.edu.cn/faculty/wangjingya/
- Haipeng Zhang faculty page:
  https://faculty.sist.shanghaitech.edu.cn/zhanghp/
- Quan Li faculty page:
  https://faculty.sist.shanghaitech.edu.cn/liquan/
- Siting Liu faculty page:
  https://faculty.sist.shanghaitech.edu.cn/liust/

Where a fact appears on more than one official page, the JSONL row includes
`cross_check_urls`. Course-row and faculty-contact facts usually have one
authoritative primary official page, so their `cross_check_urls` may be empty.

## Schema

Each JSONL row has:

- `id`: stable question id.
- `category`: topic group, such as `university_profile`, `course_catalog`.
- `question_type`: factual, comparative, conditional, multi-hop,
  time-sensitive, or negative-refusal.
- `query`: test question.
- `gt_answer`: ground-truth answer.
- `eval`: automatic matching rule.
- `source_urls`: primary official sources.
- `cross_check_urls`: secondary official sources when available.
- `verified_at`: verification date.
- `notes`: short provenance note.
- `source_evidence`: optional source-side terms used by the second-pass
  verifier.
- `negative_targets`: optional nonexistent terms that must not appear in
  official sources for `negative_refusal` items.

The evaluator supports these `eval.type` values:

- `contains_all`: every target group must appear.
- `contains_any`: at least one target group must appear.
- `exact_any`: normalized answer must equal one target.
- `regex`: at least one regex target must match.

For `contains_all` and `contains_any`, a target can be either a string or a
list of acceptable alternatives. This is used for Chinese/English names and
official abbreviations, for example:

```json
{"type": "contains_all", "targets": [["Lan Xu", "许岚"], ["Yin Cao", "曹迎"]]}
```

## Commands

Regenerate the JSONL file from the maintained source list:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 eval/build_web_testset.py
```

Validate the JSONL file:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate_testset.py \
  --testset eval/testset_web_verified.jsonl \
  --validate
```

Run second-pass source verification. This refetches official URLs, checks that
ground-truth answers satisfy their eval rules, verifies `source_evidence`, and
checks that `negative_targets` are absent from official sources.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_testset_sources.py \
  --testset eval/testset_web_verified.jsonl
```

Export the project-required result table template:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate_testset.py \
  --testset eval/testset_web_verified.jsonl \
  --export-csv eval/testset_web_verified.csv
```

After running the RAG system, fill `sys_resp_before_opt` and
`sys_resp_after_opt` in the CSV, then score it automatically:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate_testset.py \
  --testset eval/testset_web_verified.jsonl \
  --answers-csv eval/testset_web_verified.csv \
  --output-csv eval/testset_scored.csv
```

The exported CSV uses UTF-8 with BOM so it opens cleanly in Excel.
