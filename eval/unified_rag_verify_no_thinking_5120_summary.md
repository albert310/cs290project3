# Unified RAG eval: no thinking, 5120 tokens

Command:

```bash
conda run -n rag python scripts/eval_unified_rag.py \
  --testset eval/testset_web_verified.jsonl \
  --output-csv eval/unified_rag_verify_no_thinking_5120.csv \
  --top-k 8 \
  --max-context-chars 7200 \
  --max-tokens 5120 \
  --no-thinking \
  --verify-answer
```

Scoring command:

```bash
conda run -n rag python scripts/evaluate_testset.py \
  --testset eval/testset_web_verified.jsonl \
  --answers-csv eval/unified_rag_verify_no_thinking_5120.csv \
  --output-csv eval/unified_rag_verify_no_thinking_5120_scored.csv
```

Overall result:

- after_opt_accuracy: 60/100 = 0.600
- before_opt_accuracy: 0/100 = 0.000
- mean latency: 13.53s
- median latency: 11.51s
- min latency: 5.01s
- max latency: 82.55s

By question type:

| question_type | correct | total | accuracy |
|---|---:|---:|---:|
| comparative | 4 | 10 | 0.400 |
| conditional | 6 | 10 | 0.600 |
| factual | 31 | 50 | 0.620 |
| multi-hop | 7 | 10 | 0.700 |
| negative_refusal | 9 | 10 | 0.900 |
| time_sensitive | 3 | 10 | 0.300 |

By category:

| category | correct | total | accuracy |
|---|---:|---:|---:|
| course_catalog | 8 | 12 | 0.667 |
| faculty_profile | 11 | 19 | 0.579 |
| graduate_admission | 8 | 18 | 0.444 |
| negative_refusal | 9 | 10 | 0.900 |
| sist_profile | 4 | 15 | 0.267 |
| university_profile | 20 | 26 | 0.769 |

Failed IDs:

fact_007, fact_015, fact_021, fact_023, fact_025, fact_026, fact_027, fact_028, fact_029, fact_030, fact_031, fact_032, fact_033, fact_034, fact_038, fact_041, fact_042, fact_043, fact_049, comp_001, comp_003, comp_004, comp_005, comp_006, comp_008, cond_001, cond_006, cond_009, cond_010, multi_002, multi_003, multi_007, time_002, time_003, time_004, time_005, time_007, time_008, time_010, neg_007.

Baseline note:

`scripts/eval_baseline_rag.py` needs `data/sist/texts`. The current workspace only has `data/rag` symlinked to the provided SQLite artifacts, so a baseline run would build an empty text index and produce invalid results.
