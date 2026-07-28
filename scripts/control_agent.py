#!/usr/bin/env python3
"""
control_agent.py

Orchestrates the AIOps diagnosis loop:
  1. Reads data/predictions.csv (produced by predict_on_new_data.py)
  2. Finds rows that are either an ACTUAL failure (metadata_conclusion
     == "failure") or a PREDICTED failure (predicted_label == 1) that
     haven't been processed yet (tracked via a checkpoint file)
  3. Calls llm_agent.run_diagnosis() for each
  4. Posts the recommendation as a GitHub Issue (never applies a fix —
     this keeps a human in the loop)
  5. Saves a JSON report per run under aiops/reports/

Run this AFTER predict_on_new_data.py in your workflow.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import requests

import llm_agent

API = "https://api.github.com"


def gh_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def create_issue(owner, repo, token, title, body, labels=None):
    r = requests.post(
        f"{API}/repos/{owner}/{repo}/issues",
        headers=gh_headers(token),
        json={"title": title, "body": body, "labels": labels or []},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["html_url"]


def format_issue_body(row, diagnosis):
    return f"""**Automated CI/CD failure diagnosis** (AIOps agent — recommendation only, no changes applied)

| Field | Value |
|---|---|
| Run | [`#{row['run_number']}`]({row.get('logs_archive_path', '')}) |
| Workflow | `{row['workflow_path']}` |
| Actual outcome | `{row.get('metadata_conclusion', 'n/a')}` |
| Predicted failure probability | `{row.get('predicted_failure_probability', 'n/a')}` |
| Failing job | `{diagnosis.get('failing_job', 'n/a')}` |
| Failing step | `{diagnosis.get('failing_step', 'n/a')}` |
| Category | `{diagnosis.get('category', 'n/a')}` |
| Confidence | `{diagnosis.get('confidence', 'n/a')}` |

### Root cause
{diagnosis.get('root_cause', 'n/a')}

### Suggested fix
{diagnosis.get('suggested_fix', 'n/a')}

---
_This issue was created automatically by the AIOps Control Agent as part of an early-prediction CI/CD research pipeline. It is a recommendation for a human to review — no code was changed._
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="data/predictions.csv")
    ap.add_argument("--checkpoint", default="aiops/state/diagnosed_runs.json")
    ap.add_argument("--reports-dir", default="aiops/reports")
    args = ap.parse_args()

    github_token = os.environ["GITHUB_TOKEN"]
    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
    owner, repo = os.environ["GITHUB_REPOSITORY"].split("/")

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)

    processed = set()
    if Path(args.checkpoint).exists():
        processed = set(json.loads(Path(args.checkpoint).read_text()))

    df = pd.read_csv(args.predictions)

    needs_diagnosis = df[
        (df["metadata_conclusion"] == "failure")
        | (df.get("predicted_label", 0) == 1)
    ]
    needs_diagnosis = needs_diagnosis[~needs_diagnosis["metadata_id"].astype(str).isin(processed)]

    if needs_diagnosis.empty:
        print("No new failed/predicted-failure runs to diagnose.")
        return

    for _, row in needs_diagnosis.iterrows():
        run_id = str(row["metadata_id"])
        print(f"Diagnosing run {run_id} ({row['workflow_path']})...")

        try:
            diagnosis = llm_agent.run_diagnosis(
                owner, repo, run_id, row.to_dict(),
                github_token, anthropic_api_key,
            )
        except Exception as e:
            print(f"  ! diagnosis failed: {e}")
            diagnosis = {
                "root_cause": f"Diagnosis agent error: {e}",
                "category": "unknown",
                "suggested_fix": "n/a — diagnosis step failed",
                "confidence": "low",
            }

        # Save report
        report_path = Path(args.reports_dir) / f"{run_id}.json"
        report_path.write_text(json.dumps({"run": row.to_dict(), "diagnosis": diagnosis}, indent=2, default=str))

        # Post issue
        title = f"CI Failure Diagnosis: {row['workflow_path']} #{row['run_number']}"
        body = format_issue_body(row, diagnosis)
        try:
            url = create_issue(owner, repo, github_token, title, body, labels=["aiops-diagnosis"])
            print(f"  -> issue created: {url}")
        except Exception as e:
            print(f"  ! could not create issue: {e}")

        processed.add(run_id)

    Path(args.checkpoint).write_text(json.dumps(sorted(processed)))
    print(f"\nProcessed {len(needs_diagnosis)} run(s). Checkpoint updated.")


if __name__ == "__main__":
    main()
