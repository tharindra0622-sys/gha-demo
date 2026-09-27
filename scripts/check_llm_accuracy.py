#!/usr/bin/env python3
"""
check_llm_accuracy.py

Checks how accurate the LLM diagnosis agent's answers are, across your 30
test runs.

Since only YOU know the real, deliberately-triggered cause behind each
run (e.g. "I checked inject_failure on this one", "this one was the
Server Error Test"), this script works in two steps:

STEP 1 (first run): reads diagnosis_report.csv (produced by
compile_all_diagnoses.py) and creates a fill-in template,
accuracy_checklist.csv, with one row per diagnosed run and an empty
"correct" column for you to mark.

STEP 2 (after you fill it in): open accuracy_checklist.csv in Excel /
Google Sheets, and for each row, look at the AI's root_cause and
category, compare it against what you actually know is true, and type
"yes" or "no" in the "correct" column. Save the file, then run this
script again — it will read your marks and print the final accuracy.

Usage:
    python check_llm_accuracy.py --report diagnosis_report.csv --checklist accuracy_checklist.csv
"""

import argparse
import csv
from pathlib import Path


def build_template(report_path, checklist_path):
    with open(report_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fieldnames = [
        "issue_number", "run", "workflow", "actual_outcome",
        "predicted_failure_probability", "category", "root_cause",
        "suggested_fix", "correct", "notes",
    ]

    with open(checklist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "issue_number": row.get("issue_number", ""),
                "run": row.get("run", ""),
                "workflow": row.get("workflow", ""),
                "actual_outcome": row.get("actual_outcome", ""),
                "predicted_failure_probability": row.get("predicted_failure_probability", ""),
                "category": row.get("category", ""),
                "root_cause": row.get("root_cause", ""),
                "suggested_fix": row.get("suggested_fix", ""),
                "correct": "",   # <-- fill this in: yes / no
                "notes": "",     # <-- optional: what you actually triggered
            })

    print(f"Created {checklist_path} with {len(rows)} rows.")
    print("\nNext steps:")
    print(f"  1. Open {checklist_path} in Excel or Google Sheets.")
    print("  2. For each row, read 'root_cause' and compare it against what")
    print("     you know you actually triggered for that run.")
    print("  3. Type 'yes' or 'no' in the 'correct' column for every row.")
    print("  4. Optionally note what you actually triggered in 'notes'")
    print("     (e.g. 'inject_failure test', 'server error test').")
    print(f"  5. Save the file, then run this script again:")
    print(f"     python check_llm_accuracy.py --report {report_path} --checklist {checklist_path}")


def compute_accuracy(checklist_path):
    with open(checklist_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    marked = [r for r in rows if r.get("correct", "").strip().lower() in ("yes", "no", "y", "n")]
    unmarked = len(rows) - len(marked)

    if not marked:
        print(f"No rows in {checklist_path} have been marked yet.")
        print("Fill in the 'correct' column with 'yes' or 'no' for each row, then run this again.")
        return

    correct_count = sum(1 for r in marked if r["correct"].strip().lower() in ("yes", "y"))
    total = len(marked)
    accuracy = correct_count / total

    print(f"\n{'='*50}")
    print(f"LLM Diagnosis Accuracy Report")
    print(f"{'='*50}")
    print(f"Total runs marked:     {total}")
    if unmarked:
        print(f"Still unmarked:        {unmarked}  <-- fill these in too")
    print(f"Correct diagnoses:     {correct_count}")
    print(f"Incorrect diagnoses:   {total - correct_count}")
    print(f"Accuracy:              {accuracy:.1%}")

    # Breakdown by category
    print(f"\n{'-'*50}")
    print("Breakdown by category:")
    by_category = {}
    for r in marked:
        cat = r.get("category", "unknown") or "unknown"
        by_category.setdefault(cat, {"correct": 0, "total": 0})
        by_category[cat]["total"] += 1
        if r["correct"].strip().lower() in ("yes", "y"):
            by_category[cat]["correct"] += 1

    for cat, stats in sorted(by_category.items(), key=lambda x: -x[1]["total"]):
        pct = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"  {cat:25s}  {stats['correct']}/{stats['total']}  ({pct:.0%})")

    # List the incorrect ones for easy review
    incorrect = [r for r in marked if r["correct"].strip().lower() in ("no", "n")]
    if incorrect:
        print(f"\n{'-'*50}")
        print("Runs marked INCORRECT (worth discussing in your thesis):")
        for r in incorrect:
            print(f"  Issue #{r['issue_number']} (run {r['run']}): {r.get('notes', '') or r.get('category', '')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="diagnosis_report.csv")
    ap.add_argument("--checklist", default="accuracy_checklist.csv")
    args = ap.parse_args()

    if not Path(args.checklist).exists():
        if not Path(args.report).exists():
            print(f"Error: {args.report} not found. Run compile_all_diagnoses.py first.")
            return
        build_template(args.report, args.checklist)
    else:
        compute_accuracy(args.checklist)


if __name__ == "__main__":
    main()
