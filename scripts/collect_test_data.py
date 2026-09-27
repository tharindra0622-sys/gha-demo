#!/usr/bin/env python3
"""
collect_test_data.py

Pulls completed GitHub Actions runs for this repo via the REST API,
downloads their raw logs, extracts the same log_* engineered features
used in runs_200_2.csv (real training data), and appends rows to a
growing CSV that can be used as fresh/unseen test data for the thesis
ML model.

Designed to run inside a GitHub Actions job (uses GITHUB_TOKEN / gh api),
but also runs locally if you export GITHUB_TOKEN and GITHUB_REPOSITORY.

State: a checkpoint file (state/last_checkpoint.txt) stores the ISO
timestamp of the last processed run, so re-runs only pick up new runs.

Assumptions / things to double check against your original feature
engineering code (adjust if your definitions differ):
  - "shell step" vs "action step" is determined by parsing the workflow
    YAML for each job (a step with `run:` = shell, a step with `uses:` =
    action). GitHub's Jobs API does not expose this directly.
  - The auto-injected "Set up job" / "Complete job" / "Post ..." steps
    are excluded before aligning job steps to the YAML step list.
  - "early3" features are computed from the first 3 steps in the run,
    ordered chronologically across all jobs (mirrors the thesis's
    "first 20% of execution" early-prediction framing at small scale).
  - OS detection comes from each job's `labels` field (runs-on).
"""

import os
import sys
import csv
import json
import time
import zipfile
import io
import re
from pathlib import Path
from datetime import datetime, timezone

import requests
import yaml

API = "https://api.github.com"
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY")  # "owner/repo"
OUT_CSV = os.environ.get("OUT_CSV", "test_data.csv")
STATE_FILE = os.environ.get("STATE_FILE", "state/last_checkpoint.txt")
MAX_RUNS_PER_INVOCATION = int(os.environ.get("MAX_RUNS", "20"))

if not TOKEN or not REPO:
    sys.exit("GITHUB_TOKEN and GITHUB_REPOSITORY must be set")

OWNER, REPO_NAME = REPO.split("/")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# Exact column order must match runs_200_2.csv so rows can be concatenated
# directly onto the training data as held-out test rows.
FIELDNAMES = [
    "_id", "repository_name", "workflow_path", "run_number", "run_attempt",
    "metadata_id", "metadata_name", "metadata_node_id", "metadata_head_branch",
    "metadata_head_sha", "metadata_path", "metadata_display_title",
    "metadata_run_number", "metadata_event", "metadata_status",
    "metadata_conclusion", "metadata_workflow_id", "metadata_check_suite_id",
    "metadata_check_suite_node_id", "metadata_pull_requests",
    "metadata_created_at", "metadata_updated_at", "metadata_actor_login",
    "metadata_actor_id", "metadata_actor_node_id", "metadata_actor_gravatar_id",
    "metadata_actor_type", "metadata_actor_site_admin", "metadata_run_attempt",
    "metadata_referenced_workflows", "metadata_run_started_at",
    "metadata_triggering_actor_login", "metadata_triggering_actor_id",
    "metadata_triggering_actor_node_id", "metadata_triggering_actor_gravatar_id",
    "metadata_triggering_actor_type", "metadata_triggering_actor_site_admin",
    "metadata_head_commit_id", "metadata_head_commit_tree_id",
    "metadata_head_commit_message", "metadata_head_commit_timestamp",
    "metadata_head_commit_author_name", "metadata_head_commit_author_email",
    "metadata_head_commit_committer_name", "metadata_head_commit_committer_email",
    "metadata_repository_id", "metadata_repository_node_id",
    "metadata_repository_name", "metadata_repository_full_name",
    "metadata_repository_private", "metadata_repository_owner_login",
    "metadata_repository_owner_id", "metadata_repository_owner_node_id",
    "metadata_repository_owner_gravatar_id", "metadata_repository_owner_type",
    "metadata_repository_owner_site_admin", "metadata_repository_description",
    "metadata_repository_fork", "metadata_head_repository_id",
    "metadata_head_repository_node_id", "metadata_head_repository_name",
    "metadata_head_repository_full_name", "metadata_head_repository_private",
    "metadata_head_repository_owner_login", "metadata_head_repository_owner_id",
    "metadata_head_repository_owner_node_id",
    "metadata_head_repository_owner_gravatar_id",
    "metadata_head_repository_owner_type",
    "metadata_head_repository_owner_site_admin",
    "metadata_head_repository_description", "metadata_head_repository_fork",
    "logs_archive_path", "total_logs_size",
    "log_num_jobs", "log_total_steps", "log_total_duration_sec",
    "log_shell_steps", "log_action_steps", "log_error_steps",
    "log_total_lines", "log_has_linux", "log_has_macos", "log_has_windows",
    "log_num_os_types", "log_early3_total_dur", "log_early3_max_dur",
    "log_early3_min_dur", "log_early3_shell_count", "log_early3_action_count",
    "log_early3_error_count", "log_early3_avg_dur", "log_error_rate",
    "log_shell_ratio", "log_avg_step_dur", "log_max_step_dur",
]


def gh_get(url, **kwargs):
    r = requests.get(url, headers=HEADERS, timeout=30, **kwargs)
    r.raise_for_status()
    return r


def list_recent_completed_runs(since_iso):
    """List completed runs across all workflows, newest first, updated after since_iso."""
    runs = []
    page = 1
    while True:
        r = gh_get(
            f"{API}/repos/{OWNER}/{REPO_NAME}/actions/runs",
            params={"status": "completed", "per_page": 50, "page": page},
        )
        data = r.json()
        batch = data.get("workflow_runs", [])
        if not batch:
            break
        stop = False
        for run in batch:
            if since_iso and run["updated_at"] <= since_iso:
                stop = True
                continue
            runs.append(run)
        if stop or len(batch) < 50:
            break
        page += 1
        if len(runs) >= MAX_RUNS_PER_INVOCATION * 3:
            break
    runs.sort(key=lambda r: r["updated_at"])
    return runs[:MAX_RUNS_PER_INVOCATION]


def get_jobs(run_id):
    jobs = []
    page = 1
    while True:
        r = gh_get(
            f"{API}/repos/{OWNER}/{REPO_NAME}/actions/runs/{run_id}/jobs",
            params={"per_page": 100, "page": page},
        )
        data = r.json()
        batch = data.get("jobs", [])
        jobs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return jobs


def get_workflow_yaml(path, ref):
    try:
        r = gh_get(
            f"{API}/repos/{OWNER}/{REPO_NAME}/contents/{path}",
            params={"ref": ref},
        )
        import base64
        content = base64.b64decode(r.json()["content"]).decode("utf-8", errors="ignore")
        return yaml.safe_load(content)
    except Exception:
        return None


def classify_steps_from_yaml(wf_yaml, job_key):
    """Return ordered list of 'shell'/'action' for a job's declared steps."""
    if not wf_yaml:
        return []
    jobs = wf_yaml.get("jobs", {}) or {}
    job_def = jobs.get(job_key)
    if not job_def:
        return []
    out = []
    for step in job_def.get("steps", []) or []:
        if "run" in step:
            out.append("shell")
        elif "uses" in step:
            out.append("action")
        else:
            out.append("action")  # default_fallback
    return out


AUTO_STEP_NAMES = {"Set up job", "Complete job"}


def download_logs_zip(run_id):
    """Returns (total_lines, os hints found in text) or (0, set()) on failure
    (e.g. logs expired past GitHub's retention window)."""
    try:
        r = requests.get(
            f"{API}/repos/{OWNER}/{REPO_NAME}/actions/runs/{run_id}/logs",
            headers=HEADERS, timeout=60,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"  ! could not download logs for run {run_id}: {e}")
        return 0, set()

    total_lines = 0
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in zf.namelist():
            if name.endswith("/") or not name.endswith(".txt"):
                continue
            with zf.open(name) as f:
                total_lines += sum(1 for _ in f)
    return total_lines, r.content  # keep raw bytes around if needed later


OS_LABEL_MAP = {
    "ubuntu": "linux", "linux": "linux",
    "macos": "macos", "mac": "macos",
    "windows": "windows", "win": "windows",
}


def os_from_labels(labels):
    found = set()
    for lbl in labels or []:
        low = lbl.lower()
        for key, osname in OS_LABEL_MAP.items():
            if key in low:
                found.add(osname)
    return found


def extract_features(run, jobs, total_lines):
    all_steps = []  # (job, step_type, duration_sec, conclusion, started_at)
    os_types = set()

    for job in jobs:
        os_types |= os_from_labels(job.get("labels"))
        wf_yaml = get_workflow_yaml(run["path"], run["head_sha"])
        step_types = classify_steps_from_yaml(wf_yaml, job.get("name"))

        real_steps = [s for s in job.get("steps", []) if s.get("name") not in AUTO_STEP_NAMES]
        for idx, step in enumerate(real_steps):
            stype = step_types[idx] if idx < len(step_types) else "action"
            started = step.get("started_at")
            completed = step.get("completed_at")
            dur = 0
            if started and completed:
                t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                dur = max((t1 - t0).total_seconds(), 0)
            all_steps.append({
                "type": stype,
                "dur": dur,
                "failed": step.get("conclusion") == "failure",
                "started_at": started or "",
            })

    all_steps.sort(key=lambda s: s["started_at"])

    total_steps = len(all_steps)
    shell_steps = sum(1 for s in all_steps if s["type"] == "shell")
    action_steps = sum(1 for s in all_steps if s["type"] == "action")
    error_steps = sum(1 for s in all_steps if s["failed"])
    total_dur = sum(s["dur"] for s in all_steps)
    max_dur = max((s["dur"] for s in all_steps), default=0)

    early = all_steps[:3]
    e_total = sum(s["dur"] for s in early)
    e_max = max((s["dur"] for s in early), default=0)
    e_min = min((s["dur"] for s in early), default=0)
    e_shell = sum(1 for s in early if s["type"] == "shell")
    e_action = sum(1 for s in early if s["type"] == "action")
    e_error = sum(1 for s in early if s["failed"])
    e_avg = (e_total / len(early)) if early else 0

    return {
        "log_num_jobs": len(jobs),
        "log_total_steps": total_steps,
        "log_total_duration_sec": total_dur,
        "log_shell_steps": shell_steps,
        "log_action_steps": action_steps,
        "log_error_steps": error_steps,
        "log_total_lines": total_lines,
        "log_has_linux": "linux" in os_types,
        "log_has_macos": "macos" in os_types,
        "log_has_windows": "windows" in os_types,
        "log_num_os_types": len(os_types),
        "log_early3_total_dur": e_total,
        "log_early3_max_dur": e_max,
        "log_early3_min_dur": e_min,
        "log_early3_shell_count": e_shell,
        "log_early3_action_count": e_action,
        "log_early3_error_count": e_error,
        "log_early3_avg_dur": e_avg,
        "log_error_rate": (error_steps / total_steps) if total_steps else 0,
        "log_shell_ratio": (shell_steps / total_steps) if total_steps else 0,
        "log_avg_step_dur": (total_dur / total_steps) if total_steps else 0,
        "log_max_step_dur": max_dur,
    }


def build_metadata_row(run):
    hc = run.get("head_commit") or {}
    repo = run.get("repository") or {}
    head_repo = run.get("head_repository") or {}
    actor = run.get("actor") or {}
    trig = run.get("triggering_actor") or {}
    owner = repo.get("owner") or {}
    head_owner = head_repo.get("owner") or {}

    return {
        "_id": f"{repo.get('full_name')}_{run.get('path')}_{run.get('run_number')}_{run.get('run_attempt')}",
        "repository_name": repo.get("full_name"),
        "workflow_path": run.get("path"),
        "run_number": run.get("run_number"),
        "run_attempt": run.get("run_attempt"),
        "metadata_id": run.get("id"),
        "metadata_name": run.get("name"),
        "metadata_node_id": run.get("node_id"),
        "metadata_head_branch": run.get("head_branch"),
        "metadata_head_sha": run.get("head_sha"),
        "metadata_path": run.get("path"),
        "metadata_display_title": run.get("display_title"),
        "metadata_run_number": run.get("run_number"),
        "metadata_event": run.get("event"),
        "metadata_status": run.get("status"),
        "metadata_conclusion": run.get("conclusion"),
        "metadata_workflow_id": run.get("workflow_id"),
        "metadata_check_suite_id": run.get("check_suite_id"),
        "metadata_check_suite_node_id": run.get("check_suite_node_id"),
        "metadata_pull_requests": json.dumps(run.get("pull_requests") or []),
        "metadata_created_at": run.get("created_at"),
        "metadata_updated_at": run.get("updated_at"),
        "metadata_actor_login": actor.get("login"),
        "metadata_actor_id": actor.get("id"),
        "metadata_actor_node_id": actor.get("node_id"),
        "metadata_actor_gravatar_id": actor.get("gravatar_id"),
        "metadata_actor_type": actor.get("type"),
        "metadata_actor_site_admin": actor.get("site_admin"),
        "metadata_run_attempt": run.get("run_attempt"),
        "metadata_referenced_workflows": json.dumps(run.get("referenced_workflows") or []),
        "metadata_run_started_at": run.get("run_started_at"),
        "metadata_triggering_actor_login": trig.get("login"),
        "metadata_triggering_actor_id": trig.get("id"),
        "metadata_triggering_actor_node_id": trig.get("node_id"),
        "metadata_triggering_actor_gravatar_id": trig.get("gravatar_id"),
        "metadata_triggering_actor_type": trig.get("type"),
        "metadata_triggering_actor_site_admin": trig.get("site_admin"),
        "metadata_head_commit_id": hc.get("id"),
        "metadata_head_commit_tree_id": hc.get("tree_id"),
        "metadata_head_commit_message": hc.get("message"),
        "metadata_head_commit_timestamp": hc.get("timestamp"),
        "metadata_head_commit_author_name": (hc.get("author") or {}).get("name"),
        "metadata_head_commit_author_email": (hc.get("author") or {}).get("email"),
        "metadata_head_commit_committer_name": (hc.get("committer") or {}).get("name"),
        "metadata_head_commit_committer_email": (hc.get("committer") or {}).get("email"),
        "metadata_repository_id": repo.get("id"),
        "metadata_repository_node_id": repo.get("node_id"),
        "metadata_repository_name": repo.get("name"),
        "metadata_repository_full_name": repo.get("full_name"),
        "metadata_repository_private": repo.get("private"),
        "metadata_repository_owner_login": owner.get("login"),
        "metadata_repository_owner_id": owner.get("id"),
        "metadata_repository_owner_node_id": owner.get("node_id"),
        "metadata_repository_owner_gravatar_id": owner.get("gravatar_id"),
        "metadata_repository_owner_type": owner.get("type"),
        "metadata_repository_owner_site_admin": owner.get("site_admin"),
        "metadata_repository_description": repo.get("description"),
        "metadata_repository_fork": repo.get("fork"),
        "metadata_head_repository_id": head_repo.get("id"),
        "metadata_head_repository_node_id": head_repo.get("node_id"),
        "metadata_head_repository_name": head_repo.get("name"),
        "metadata_head_repository_full_name": head_repo.get("full_name"),
        "metadata_head_repository_private": head_repo.get("private"),
        "metadata_head_repository_owner_login": head_owner.get("login"),
        "metadata_head_repository_owner_id": head_owner.get("id"),
        "metadata_head_repository_owner_node_id": head_owner.get("node_id"),
        "metadata_head_repository_owner_gravatar_id": head_owner.get("gravatar_id"),
        "metadata_head_repository_owner_type": head_owner.get("type"),
        "metadata_head_repository_owner_site_admin": head_owner.get("site_admin"),
        "metadata_head_repository_description": head_repo.get("description"),
        "metadata_head_repository_fork": head_repo.get("fork"),
        "logs_archive_path": f"{repo.get('full_name')}/{run.get('id')}/logs.zip",
        "total_logs_size": None,  # filled in after download
    }


def main():
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    since_iso = ""
    if Path(STATE_FILE).exists():
        since_iso = Path(STATE_FILE).read_text().strip()

    runs = list_recent_completed_runs(since_iso)
    if not runs:
        print("No new completed runs to process.")
        return

    file_exists = Path(OUT_CSV).exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        latest_ts = since_iso
        for run in runs:
            run_id = run["id"]
            print(f"Processing run {run_id} ({run.get('name')}, {run.get('conclusion')})...")
            row = build_metadata_row(run)

            jobs = get_jobs(run_id)
            total_lines, raw_zip = download_logs_zip(run_id)
            row["total_logs_size"] = len(raw_zip) if isinstance(raw_zip, (bytes, bytearray)) else 0

            feats = extract_features(run, jobs, total_lines)
            row.update(feats)

            writer.writerow(row)
            latest_ts = max(latest_ts, run["updated_at"]) if latest_ts else run["updated_at"]
            time.sleep(0.5)  # be gentle on API rate limits

    Path(STATE_FILE).write_text(latest_ts)
    print(f"Wrote {len(runs)} new row(s) to {OUT_CSV}. Checkpoint -> {latest_ts}")


if __name__ == "__main__":
    main()
