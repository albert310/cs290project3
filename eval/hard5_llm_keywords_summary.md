# Hard-5 LLM query keyword test

Evaluation date: 2026-06-03

Selected questions:

| id | type | reason |
| --- | --- | --- |
| fact_026 | factual | SIST English introduction and research-area list |
| comp_006 | comparative | compare two similar course codes and fields |
| cond_004 | conditional | admission condition plus exam-subject lookup |
| multi_002 | multi-hop | course instructor to faculty profile |
| time_008 | time_sensitive | faculty homepage news with 2026 timestamp |

## Results

| Run | CSV | Correct | Avg latency |
| --- | --- | ---: | ---: |
| unified RAG, no LLM query keywords | `eval/hard5_unified_no_keywords.csv` | 0 / 5 | 33.35 s |
| unified RAG, LLM query keywords enabled | `eval/hard5_unified_llm_keywords.csv` | 0 / 5 | 32.61 s |
| unified RAG, iterative search rollout enabled | `eval/hard5_unified_iterative_search.csv` | 0 / 5 | 42.00 s |

## Findings

1. The LLM keyword planner produced reasonable keywords, for example `SIST`,
   `English introduction`, `research areas`, `CS290U`, `CS290S`, `Instructor`,
   `408`, and `CVPR 2026`.
2. The hard-5 automatic score did not improve. These failures are not mainly
   caused by missing query terms.
3. `cond_004` produced the core answer `408计算机学科专业基础`, but the strict
   evaluator still marked it incorrect because the response omitted the expected
   official-directory caveat.
4. `multi_002` still fails because the current system retrieves the `CS282`
   course row but does not follow `Hao Wang/王浩` to the faculty profile.
5. `fact_026` and `time_008` still miss the correct pages, suggesting that the
   database needs stronger page-level metadata, aliases, or targeted boosts for
   SIST English introduction and faculty homepages.

## Conclusion

LLM keyword planning works as an auditable switch, but this first version has
little effect on the hardest representative questions. The next useful
optimization should be multi-step retrieval and entity linking rather than only
more query keywords.

## Iterative search rollout update

The second optimization adds a model-controlled rollout loop. After seeing the
question and current evidence, the model can either stop searching or request
another search with keywords. The loop is capped at five model-requested search
steps; after the cap, the system must answer or refuse from accumulated evidence.

On `multi_002`, the rollout behaved as intended:

1. Initial retrieval found the `CS282` course records and identified `王浩 (Hao Wang)`.
2. Step 1 requested another search for `王浩`, `Hao Wang`, `研究方向`,
   `research interests`, `profile`, and `homepage`.
3. Retrieval then found the Wang Hao profile page.
4. Step 2 selected `answer`, and the final answer used both the course record and
   the faculty profile.

The automatic score remains 0 / 5 on this hard subset because the local profile
page states `机器学习与优化算法`, while the test-set answer expects the more
specific externally verified wording about nonlinear optimization and its
applications in operations research, computer science, and statistics. This shows
that rollout improves evidence chaining, but remaining failures need stronger
source coverage, entity linking, and page-specific normalization.
