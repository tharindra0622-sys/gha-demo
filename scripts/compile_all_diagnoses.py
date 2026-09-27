#!/usr/bin/env python3
"""
compile_all_diagnoses.py

Pulls EVERY issue labeled "aiops-diagnosis" from the repo and compiles
them into one clean report — a table with run number, actual outcome,
predicted probability, category, confidence, root cause, evidence used,
and suggested fix — so the whole set of LLM answers can be reviewed in
one place instead of clicking through issues one by one.

Usage:
    export GITHUB_TOKEN=ghp_xxx
    export GITHUB_REPOSITORY=tharindra0622-sys/gha-demo
    python compile_all_diagnoses.py --output diagnosis_report.csv
"""

import os
import re
import csv
import argparse
import requests

API = "https://api.github.com"


def gh_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_all_diagnosis_issues(owner, repo, token):
    """Fetches every issue (open and closed) labeled aiops-diagnosis."""
    issues = []
    page = 1
    while True:
        r = requests.get(
            f"{API}/repos/{owner}/{repo}/issues",
            headers=gh_headers(token),
            params={"labels": "aiops-diagnosis", "state": "all", "per_page": 100, "page": page},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        # skip pull requests, which the issues endpoint also returns
        batch = [i for i in batch if "pull_request" not in i]
        issues.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return issues


def extract_field(body, field_name):
    """Pulls a single field's value out of the markdown table/body of a diagnosis issue."""
    # Try table row format: | Field | Value |
    table_match = re.search(rf"\|\s*{re.escape(field_name)}\s*\|\s*(.+?)\s*\|", body, re.IGNORECASE)
    if table_match:
        return table_match.group(1).strip("` ")
    # Try "### Field\ncontent" format
    section_match = re.search(rf"###\s*{re.escape(field_name)}\s*\n+(.+?)(?=\n###|\n---|\Z)", body, re.IGNORECASE | re.DOTALL)
    if section_match:
        return section_match.group(1).strip()
    return ""


def parse_issue(issue):
    body = issue.get("body") or ""
    return {
        "issue_number": issue["number"],
        "issue_title": issue["title"],
        "issue_state": issue["state"],
        "issue_url": issue["html_url"],
        "created_at": issue["created_at"],
        "run": extract_field(body, "Run"),
        "workflow": extract_field(body, "Workflow"),
        "actual_outcome": extract_field(body, "Actual outcome"),
        "predicted_failure_probability": extract_field(body, "Predicted failure probability"),
        "failing_job": extract_field(body, "Failing job"),
        "failing_step": extract_field(body, "Failing step"),
        "category": extract_field(body, "Category"),
        "confidence": extract_field(body, "Confidence"),
        "root_cause": extract_field(body, "Root cause"),
        "evidence_considered": extract_field(body, "Evidence considered"),
        "suggested_fix": extract_field(body, "Suggested fix"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="diagnosis_report.csv")
    args = ap.parse_args()

    token = os.environ["GITHUB_TOKEN"]
    owner, repo = os.environ["GITHUB_REPOSITORY"].split("/")

    print(f"Fetching all aiops-diagnosis issues from {owner}/{repo}...")
    issues = get_all_diagnosis_issues(owner, repo, token)
    print(f"Found {len(issues)} issues.")

    rows = [parse_issue(i) for i in issues]

    fieldnames = [
        "issue_number", "issue_title", "issue_state", "created_at", "run", "workflow",
        "actual_outcome", "predicted_failure_probability", "failing_job", "failing_step",
        "category", "confidence", "root_cause", "evidence_considered", "suggested_fix",
        "issue_url",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.output}")

    # Quick summary printed to console too
    categories = {}
    for row in rows:
        cat = row["category"] or "unknown"
        categories[cat] = categories.get(cat, 0) + 1
    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
