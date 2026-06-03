from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


PROJECT_RESULT_COLUMNS = [
    "id",
    "query",
    "gt_answer",
    "category",
    "question_type",
    "eval_type",
    "eval_targets",
    "source_urls",
    "sys_resp_before_opt",
    "sys_resp_after_opt",
    "is_correct_before_opt",
    "is_correct_after_opt",
]

REQUIRED_FIELDS = {
    "id",
    "category",
    "question_type",
    "query",
    "gt_answer",
    "eval",
    "source_urls",
    "verified_at",
}
SUPPORTED_EVAL_TYPES = {"contains_all", "contains_any", "exact_any", "regex"}


def normalize_text(text: Any) -> str:
    """Normalize answers for tolerant factual matching."""

    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = value.replace("（", "(").replace("）", ")")
    value = value.replace("，", ",").replace("。", ".")
    value = value.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", "", value)


def target_groups(targets: Sequence[Any]) -> List[List[str]]:
    groups: List[List[str]] = []
    for target in targets:
        if isinstance(target, str):
            groups.append([target])
        elif isinstance(target, list) and all(isinstance(item, str) for item in target):
            groups.append(target)
        else:
            raise ValueError(f"invalid target entry: {target!r}")
    return groups


def contains_group(answer_norm: str, group: Sequence[str]) -> bool:
    return any(normalize_text(item) in answer_norm for item in group)


def score_answer(answer: str, eval_spec: Mapping[str, Any]) -> bool:
    eval_type = str(eval_spec.get("type") or "")
    targets = eval_spec.get("targets") or []
    if eval_type not in SUPPORTED_EVAL_TYPES:
        raise ValueError(f"unsupported eval type: {eval_type}")
    if not isinstance(targets, list) or not targets:
        raise ValueError("eval.targets must be a non-empty list")

    answer_norm = normalize_text(answer)
    groups = target_groups(targets)

    if eval_type == "contains_all":
        return all(contains_group(answer_norm, group) for group in groups)
    if eval_type == "contains_any":
        return any(contains_group(answer_norm, group) for group in groups)
    if eval_type == "exact_any":
        return any(answer_norm == normalize_text(item) for group in groups for item in group)
    if eval_type == "regex":
        return any(
            re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL)
            or re.search(pattern, answer_norm, flags=re.IGNORECASE | re.DOTALL)
            for group in groups
            for pattern in group
        )
    raise AssertionError(eval_type)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL row must be an object")
            item["_line_no"] = line_no
            records.append(item)
    return records


def validate_records(records: Sequence[Mapping[str, Any]]) -> Tuple[Counter, Counter]:
    seen_ids = set()
    categories: Counter = Counter()
    question_types: Counter = Counter()

    for item in records:
        line = item.get("_line_no", "?")
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            raise ValueError(f"line {line}: missing required fields: {sorted(missing)}")

        item_id = str(item["id"])
        if item_id in seen_ids:
            raise ValueError(f"line {line}: duplicate id {item_id!r}")
        seen_ids.add(item_id)

        if not isinstance(item["source_urls"], list) or not item["source_urls"]:
            raise ValueError(f"line {line}: source_urls must be a non-empty list")
        if "cross_check_urls" in item and not isinstance(item["cross_check_urls"], list):
            raise ValueError(f"line {line}: cross_check_urls must be a list when present")

        eval_spec = item["eval"]
        if not isinstance(eval_spec, dict):
            raise ValueError(f"line {line}: eval must be an object")
        eval_type = eval_spec.get("type")
        if eval_type not in SUPPORTED_EVAL_TYPES:
            raise ValueError(f"line {line}: unsupported eval.type {eval_type!r}")
        targets = eval_spec.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"line {line}: eval.targets must be a non-empty list")
        target_groups(targets)

        categories[str(item["category"])] += 1
        question_types[str(item["question_type"])] += 1

    return categories, question_types


def as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def export_project_csv(records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROJECT_RESULT_COLUMNS)
        writer.writeheader()
        for item in records:
            eval_spec = item["eval"]
            writer.writerow(
                {
                    "id": item["id"],
                    "query": item["query"],
                    "gt_answer": item["gt_answer"],
                    "category": item["category"],
                    "question_type": item["question_type"],
                    "eval_type": eval_spec["type"],
                    "eval_targets": as_json(eval_spec["targets"]),
                    "source_urls": as_json(item["source_urls"]),
                    "sys_resp_before_opt": "",
                    "sys_resp_after_opt": "",
                    "is_correct_before_opt": "",
                    "is_correct_after_opt": "",
                }
            )


def load_csv_rows(path: Path) -> List[MutableMapping[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def evaluate_answers(
    records: Sequence[Mapping[str, Any]],
    answers_path: Path,
    output_path: Path | None = None,
) -> Tuple[int, int, int | None, int | None]:
    by_id = {str(item["id"]): item for item in records}
    rows = load_csv_rows(answers_path)

    before_total = before_correct = after_total = after_correct = 0
    single_total = single_correct = 0

    for row in rows:
        item_id = row.get("id", "")
        if item_id not in by_id:
            raise ValueError(f"answers CSV has unknown id: {item_id!r}")
        spec = by_id[item_id]["eval"]

        if "sys_answer" in row:
            single_total += 1
            ok = score_answer(row.get("sys_answer", ""), spec)
            single_correct += int(ok)
            row["is_correct"] = str(int(ok))

        if "sys_resp_before_opt" in row:
            before_total += 1
            ok = score_answer(row.get("sys_resp_before_opt", ""), spec)
            before_correct += int(ok)
            row["is_correct_before_opt"] = str(int(ok))

        if "sys_resp_after_opt" in row:
            after_total += 1
            ok = score_answer(row.get("sys_resp_after_opt", ""), spec)
            after_correct += int(ok)
            row["is_correct_after_opt"] = str(int(ok))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if before_total or after_total:
        return before_correct, before_total, after_correct, after_total
    return single_correct, single_total, None, None


def print_summary(records: Sequence[Mapping[str, Any]], categories: Counter, question_types: Counter) -> None:
    print(f"validated {len(records)} questions")
    print("categories:")
    for name, count in sorted(categories.items()):
        print(f"  {name}: {count}")
    print("question_types:")
    for name, count in sorted(question_types.items()):
        print(f"  {name}: {count}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and score the web-verified ShanghaiTech/SIST test set.")
    parser.add_argument("--testset", default="eval/testset_web_verified.jsonl", help="Path to the JSONL test set.")
    parser.add_argument("--validate", action="store_true", help="Validate schema and print summary.")
    parser.add_argument("--export-csv", help="Export a project-format CSV template.")
    parser.add_argument("--answers-csv", help="Score a CSV containing id plus sys_answer or before/after response columns.")
    parser.add_argument("--output-csv", help="Write scored answer CSV to this path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    testset_path = Path(args.testset)

    records = read_jsonl(testset_path)
    categories, question_types = validate_records(records)

    did_action = False
    if args.validate or not any([args.export_csv, args.answers_csv]):
        print_summary(records, categories, question_types)
        did_action = True

    if args.export_csv:
        export_project_csv(records, Path(args.export_csv))
        print(f"wrote CSV template: {args.export_csv}")
        did_action = True

    if args.answers_csv:
        before_correct, before_total, after_correct, after_total = evaluate_answers(
            records,
            Path(args.answers_csv),
            Path(args.output_csv) if args.output_csv else None,
        )
        if after_total is None:
            acc = before_correct / before_total if before_total else 0.0
            print(f"accuracy: {before_correct}/{before_total} = {acc:.3f}")
        else:
            before_acc = before_correct / before_total if before_total else 0.0
            after_acc = after_correct / after_total if after_total else 0.0
            print(f"before_opt_accuracy: {before_correct}/{before_total} = {before_acc:.3f}")
            print(f"after_opt_accuracy: {after_correct}/{after_total} = {after_acc:.3f}")
        if args.output_csv:
            print(f"wrote scored CSV: {args.output_csv}")
        did_action = True

    return 0 if did_action else 1


if __name__ == "__main__":
    raise SystemExit(main())
