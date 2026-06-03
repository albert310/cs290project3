# Unified RAG evaluation summary

Evaluation date: 2026-06-03

Database: `data/rag/knowledge.sqlite`

Output CSV: `eval/unified_rag_after_opt.csv`

Command:

```bash
python3 scripts/eval_unified_rag.py --output-csv eval/unified_rag_after_opt.csv --top-k 8
```

## Overall result

| Metric | Result |
| --- | ---: |
| Total accuracy | 47 / 100 |
| Accuracy | 0.470 |

## Accuracy by question type

| Type | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| factual | 29 | 50 | 0.580 |
| comparative | 4 | 10 | 0.400 |
| conditional | 2 | 10 | 0.200 |
| multi-hop | 0 | 10 | 0.000 |
| time_sensitive | 3 | 10 | 0.300 |
| negative_refusal | 9 | 10 | 0.900 |

## Main observations

1. Negative refusal is strong. Exact nonexistent course filtering and no-evidence refusal work well on 9 of 10 reverse questions.
2. Factual questions are usable but uneven. Basic ShanghaiTech facts and many course facts are answered correctly, while SIST English introduction, SIST admission-page statistics, and several faculty-page facts are often missed.
3. Multi-hop is the weakest part. Current retrieval usually finds the course row, but does not automatically follow the instructor name to the corresponding faculty profile.
4. Conditional and comparative questions need better evidence selection. They often require multiple aligned snippets, but the current top-k prompt is still mostly single-query retrieval.
5. Time-sensitive questions partially benefit from year reranking, but queries about exact dated school statistics and faculty news still miss the right page.

## Suggested next improvements

1. Add query decomposition for multi-hop questions: retrieve the course first, extract instructor names, then retrieve faculty profiles.
2. Add stronger boosts for SIST overview, English introduction, graduate admission pages, and faculty profile pages.
3. Add structured aliases for Chinese and English faculty names, course names, and research-center abbreviations.
4. Add evidence compression before generation so comparison and conditional questions include the relevant fields instead of long unrelated snippets.
5. Keep refusal logic, but adjust the negative evaluator or answer template for questions where a correct negation is not counted as refusal.
