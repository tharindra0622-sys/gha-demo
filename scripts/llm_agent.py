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
import json
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
    """Downloads the log for ONE SPECIFIC job using GitHub's dedicated
    per-job log endpoint. This is deliberately NOT done by downloading the
    whole run's zip and guessing which file inside it belongs to this job
    — that approach was unreliable for matrix builds (multiple jobs with
    similar names), since the zip's internal file/folder naming doesn't
    always match a job's display name in a predictable way. Using the
    job's numeric ID directly guarantees we get exactly the right job's
    log, every time."""
    job_id = job.get("id")
    r = requests.get(
        f"{API}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs",
        headers=gh_headers(token), timeout=60,
    )
    r.raise_for_status()
    text = r.text
    return text[-max_chars:]


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
    # Claude sometimes wraps its JSON answer in a markdown code fence
    # (```json ... ```); strip that before parsing.
    cleaned = raw_text
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "root_cause": "Could not parse LLM response as JSON",
            "category": "unknown",
            "suggested_fix": raw_text[:500],
            "confidence": "low",
        }


def run_diagnosis(owner, repo, run_id, run_metadata, github_token, anthropic_api_key):
    actual_conclusion = run_metadata.get("metadata_conclusion")
    jobs = get_jobs(owner, repo, run_id, github_token)
    job, step = find_failing_job_and_step(jobs)
    if not job:
        if actual_conclusion == "success":
            # The run actually passed — the model flagged it as risky, but
            # there's no real error to diagnose. This is a false positive,
            # not a broken pipeline.
            return {
                "root_cause": "This run actually succeeded. The ML model predicted a failure risk, but no error occurred — this is a false positive from the model, not a real pipeline problem.",
                "category": "false_positive",
                "suggested_fix": "No fix needed — the run passed. If false positives like this are common, consider reviewing the model's decision threshold or adding more training examples similar to this run.",
                "confidence": "high",
            }
        return {
            "root_cause": "No failing job found in run data (run may have failed before any job started)",
            "category": "config_error",
            "suggested_fix": "Check the workflow YAML syntax — GitHub may have rejected it before scheduling any job.",
            "confidence": "medium",
        }
    log_text = download_log_text_for_job(owner, repo, run_id, job, github_token)
    print(f"  [debug] fetched {len(log_text)} chars of log for job_id={job.get('id')} ({job.get('name')})")
    print(f"  [debug] log preview: {log_text[:300]!r}")
    if len(log_text.strip()) < 50:
        # The log fetch returned little/nothing — don't send this to the
        # LLM, since it would have almost no real information to work
        # with and could produce a plausible-sounding but ungrounded
        # guess. Surface this honestly instead.
        return {
            "root_cause": f"Log retrieval returned only {len(log_text.strip())} characters for job_id={job.get('id')} — too little content to diagnose reliably. This may mean the log hasn't finished uploading yet, or the API token lacks permission to read it.",
            "category": "unknown",
            "suggested_fix": "Check this job's log manually in the GitHub Actions UI. If it looks normal there, this may be a timing issue — try re-running diagnosis a minute later.",
            "confidence": "low",
            "failing_job": job.get("name"),
            "failing_step": step.get("name") if step else None,
        }
    diagnosis = diagnose_failure(log_text, run_metadata, anthropic_api_key)
    diagnosis["failing_job"] = job.get("name")
    diagnosis["failing_step"] = step.get("name") if step else None
    return diagnosis
