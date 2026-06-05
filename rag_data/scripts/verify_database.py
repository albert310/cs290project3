from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from common import DEFAULT_DB_PATH, RAG_DATA_ROOT, read_json
from search_db import search


FACTS_PATH = RAG_DATA_ROOT / "config" / "verified_seed_facts.json"
REPORT_PATH = RAG_DATA_ROOT / "reports" / "verification_report.md"
CSV_PATH = RAG_DATA_ROOT / "reports" / "verification_results.csv"


def verify(db_path: Path = DEFAULT_DB_PATH, *, top_k: int = 8) -> Dict[str, Any]:
    facts = read_json(FACTS_PATH)
    rows: List[Dict[str, Any]] = []
    passed = 0
    for fact in facts:
        query = str(fact["question"])
        expected_terms = [str(term) for term in fact.get("expected_terms") or []]
        hits = search(db_path, query, top_k=top_k)
        joined = "\n".join(json.dumps(hit, ensure_ascii=False) for hit in hits).lower()
        matched_terms = [term for term in expected_terms if term.lower() in joined]
        source_url = str(fact.get("source_url") or "")
        source_hit = any(source_url and source_url in str(hit.get("url") or "") for hit in hits)
        ok = bool(hits) and len(matched_terms) >= max(1, min(len(expected_terms), 2)) and (source_hit or len(matched_terms) == len(expected_terms))
        if ok:
            passed += 1
        top = hits[0] if hits else {}
        rows.append(
            {
                "id": fact.get("id"),
                "question": query,
                "ok": ok,
                "matched_terms": "; ".join(matched_terms),
                "expected_terms": "; ".join(expected_terms),
                "source_hit": source_hit,
                "top_title": top.get("title", ""),
                "top_category": top.get("category", ""),
                "top_tier": top.get("source_tier", ""),
                "top_url": top.get("url", ""),
                "top_snippet": top.get("snippet", ""),
            }
        )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Clean RAG Verification Report",
        "",
        f"- Database: `{db_path}`",
        f"- Cases: **{len(rows)}**",
        f"- Passed: **{passed}/{len(rows)}**",
        "",
        "| id | ok | matched_terms | top_tier | top_category | top_title |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['ok']} | {row['matched_terms']} | {row['top_tier']} | {row['top_category']} | {str(row['top_title'])[:80]} |"
        )
    lines.extend(["", "## Details", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['id']} {'PASS' if row['ok'] else 'FAIL'}",
                "",
                f"- Question: {row['question']}",
                f"- Expected terms: {row['expected_terms']}",
                f"- Matched terms: {row['matched_terms']}",
                f"- Top URL: {row['top_url']}",
                f"- Top snippet: {row['top_snippet']}",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return {"cases": len(rows), "passed": passed, "report": str(REPORT_PATH), "csv": str(CSV_PATH)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify clean RAG DB against web-checked seed facts.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(verify(args.db, top_k=args.top_k), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
