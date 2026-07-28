#!/usr/bin/env python3
"""
llm_agent.py

Given a failing GitHub Actions run, downloads its logs, isolates the
failing job/step's log text, and asks an LLM (Claude, via the Anthropic
API) to diagnose the likely root cause and recommend a fix.

Design principle (per thesis architecture): this agent NEVER applies a
fix automatically — it only produces a structured recommendation. A
human (or a separate, explicitly-scoped tool) decides whether to act on
it. This keeps the AIOps layer architecturally separate from the CI/CD
pipeline itself.
"""

import os
import io
import json
import zipfile
import requests
from anthropic import Anthropic

API = "https://api.github.com"


def gh_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_jobs(owner, repo, run_id, token):
    r = requests.get(
        f"{API}/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
        headers=gh_headers(token), params={"per_page": 100}, timeout=30,
    )
    r.raise_for_status()
    return r.json().get("jobs", [])


def find_failing_job_and_step(jobs):
    for job in jobs:
        if job.get("conclusion") == "failure":
            for step in job.get("steps", []) or []:
                if step.get("conclusion") == "failure":
                    return job, step
            return job, None
    return None, None


def download_log_text_for_job(owner, repo, run_id, job, token, max_chars=6000):
    """Downloads the full run log zip and extracts the text for the
    failing job, truncated to the last max_chars characters (the error
    is almost always near the end)."""
    r = requests.get(
        f"{API}/repos/{owner}/{repo}/actions/runs/{run_id}/logs",
        headers=gh_headers(token), timeout=60,
    )
    r.raise_for_status()

    job_name = job.get("name", "")
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        # Job logs are usually named like "<job_name>/<step>.txt" or
        # "<n>_<job_name>.txt" at the top level.
        candidates = [
            n for n in zf.namelist()
            if n.endswith(".txt") and job_name.split(" ")[0].lower() in n.lower()
        ]
        if not candidates:
            candidates = [n for n in zf.namelist() if n.endswith(".txt")]
        text_parts = []
        for name in candidates:
            with zf.open(name) as f:
                text_parts.append(f.read().decode("utf-8", errors="ignore"))
    full_text = "\n".join(text_parts)
    return full_text[-max_chars:]


DIAGNOSIS_SYSTEM_PROMPT = """You are a CI/CD failure diagnosis assistant.
You will be given a log excerpt from a failed GitHub Actions job.

Respond with ONLY a JSON object (no markdown fences, no preamble) with
this exact shape:
{
  "root_cause": "one or two sentence plain-language explanation of what went wrong",
  "category": "one of: dependency_error, test_failure, syntax_error, config_error, network_timeout, permission_error, resource_limit, flaky_test, unknown",
  "suggested_fix": "a specific, actionable fix recommendation — what to change and where",
  "confidence": "high, medium, or low"
}

You are recommending a fix for a human to review — you are NOT applying
any change yourself. Be specific and concrete; avoid generic advice like
"check the logs" since the human is already looking at this log."""


def diagnose_failure(log_excerpt, run_metadata, api_key):
    client = Anthropic(api_key=api_key)
    user_content = (
        f"Repository: {run_metadata.get('repository_name')}\n"
        f"Workflow: {run_metadata.get('workflow_path')}\n"
        f"Commit message: {run_metadata.get('metadata_head_commit_message')}\n\n"
        f"Log excerpt (most recent {len(log_excerpt)} chars):\n"
        f"```\n{log_excerpt}\n```"
    )
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=DIAGNOSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    raw_text = resp.content[0].text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "root_cause": "Could not parse LLM response as JSON",
            "category": "unknown",
            "suggested_fix": raw_text[:500],
            "confidence": "low",
        }


def run_diagnosis(owner, repo, run_id, run_metadata, github_token, anthropic_api_key):
    jobs = get_jobs(owner, repo, run_id, github_token)
    job, step = find_failing_job_and_step(jobs)
    if not job:
        return {
            "root_cause": "No failing job found in run data (run may have failed before any job started)",
            "category": "config_error",
            "suggested_fix": "Check the workflow YAML syntax — GitHub may have rejected it before scheduling any job.",
            "confidence": "medium",
        }
    log_text = download_log_text_for_job(owner, repo, run_id, job, github_token)
    diagnosis = diagnose_failure(log_text, run_metadata, anthropic_api_key)
    diagnosis["failing_job"] = job.get("name")
    diagnosis["failing_step"] = step.get("name") if step else None
    return diagnosis
