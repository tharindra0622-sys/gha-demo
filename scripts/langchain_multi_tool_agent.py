#!/usr/bin/env python3
"""
langchain_multi_tool_agent.py

LangChain-based version of the multi-tool CI/CD diagnosis agent.

This does the same job as multi_tool_llm_agent.py — giving Claude several
tools (failure log, past-issue search, workflow file, GitHub status) and
letting it decide which to use before producing a diagnosis — but the
tool-calling loop itself is built with LangChain's agent framework
(`langchain.agents.create_agent`, LangChain 1.x) instead of a hand-written
loop over the Anthropic SDK.

Design principle unchanged: this agent only ever produces a
recommendation. It never modifies code or applies a fix automatically —
a human reviews the output (posted as a GitHub Issue) and decides.
"""

import os
import base64
import requests
from typing import List, Literal

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from pydantic import BaseModel, Field

API = "https://api.github.com"
STATUS_API = "https://www.githubstatus.com/api/v2/summary.json"


def gh_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# ---------------------------------------------------------------------
# Structured output shape — LangChain fills this in for the final answer
# ---------------------------------------------------------------------

class Diagnosis(BaseModel):
    root_cause: str = Field(description="Plain-language explanation, referencing which evidence supports it")
    category: Literal[
        "dependency_error", "test_failure", "syntax_error", "config_error",
        "network_timeout", "permission_error", "resource_limit",
        "flaky_test", "infrastructure_issue", "unknown",
    ]
    evidence_used: List[str] = Field(description="Which tools/sources actually informed this diagnosis")
    suggested_fix: str = Field(description="A specific, actionable fix recommendation")
    confidence: Literal["high", "medium", "low"]


def build_tools(owner, repo, github_token, job_id, workflow_path, head_sha, run_started_at):
    """Builds the 4 tools as LangChain @tool-decorated functions, bound to
    this specific run's context via closures."""

    @tool
    def get_failure_log() -> str:
        """Get the raw log text for the failing job. Always call this first."""
        r = requests.get(
            f"{API}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs",
            headers=gh_headers(github_token), timeout=60,
        )
        r.raise_for_status()
        return r.text[-6000:]

    @tool
    def search_repo_issues(query: str) -> str:
        """Search this repository's past GitHub Issues for similar error
        text, to check whether this failure (or something similar) has
        happened before and how it was handled."""
        r = requests.get(
            f"{API}/search/issues",
            headers=gh_headers(github_token),
            params={"q": f"{query} repo:{owner}/{repo} type:issue", "per_page": 5},
            timeout=30,
        )
        if r.status_code != 200:
            return f"(search failed: {r.status_code})"
        items = r.json().get("items", [])
        if not items:
            return "No similar past issues found in this repository."
        return "\n".join(f"- #{it['number']} \"{it['title']}\" (state: {it['state']})" for it in items[:5])

    @tool
    def get_workflow_file() -> str:
        """Fetch the actual workflow YAML file that defines this pipeline,
        to understand what the failing step was supposed to do."""
        r = requests.get(
            f"{API}/repos/{owner}/{repo}/contents/{workflow_path}",
            headers=gh_headers(github_token), params={"ref": head_sha}, timeout=30,
        )
        if r.status_code != 200:
            return f"(could not fetch workflow file: {r.status_code})"
        content = base64.b64decode(r.json()["content"]).decode("utf-8", errors="ignore")
        return content[:3000]

    @tool
    def check_github_status() -> str:
        """Check GitHub's own public status page for any platform-wide
        incidents, to rule out (or in) GitHub's infrastructure as the
        real cause rather than the user's code."""
        try:
            r = requests.get(STATUS_API, timeout=15)
            r.raise_for_status()
            data = r.json()
            indicator = data.get("status", {}).get("indicator", "unknown")
            incidents = data.get("incidents", [])
            active = [i["name"] for i in incidents if i.get("status") != "resolved"]
            return (
                f"Current GitHub status indicator: {indicator}. "
                f"Active incidents right now: {active or 'none'}. "
                f"NOTE: this reflects GitHub's status at the time of this check, "
                f"not necessarily at the time the run failed ({run_started_at}); "
                f"treat as a weak, supporting signal only."
            )
        except Exception as e:
            return f"(could not check GitHub status: {e})"

    return [get_failure_log, search_repo_issues, get_workflow_file, check_github_status]


SYSTEM_PROMPT = """You are a CI/CD failure diagnosis assistant with access to several tools.
Do not rely on the failure log alone. Use the tools available to you to
gather multiple sources of evidence before forming a conclusion:
- ALWAYS start with get_failure_log to see the actual error.
- If the error looks like it could be a recurring or known issue, use
  search_repo_issues to check history.
- If you need to understand what the step was supposed to do, use
  get_workflow_file.
- If the failure looks like it could be caused by GitHub's own
  infrastructure rather than the user's code, use check_github_status.

You do not need to call every tool for every failure — use judgement.
You are producing a recommendation for a human to review — you are NOT
applying any change yourself."""


def run_langchain_diagnosis(owner, repo, run_id, job, run_metadata, github_token, anthropic_api_key):
    job_id = job.get("id")
    workflow_path = run_metadata.get("workflow_path")
    head_sha = run_metadata.get("metadata_head_sha")
    run_started_at = run_metadata.get("metadata_run_started_at", "unknown")

    tools = build_tools(owner, repo, github_token, job_id, workflow_path, head_sha, run_started_at)

    model = ChatAnthropic(model="claude-sonnet-4-5", api_key=anthropic_api_key, max_tokens=1500)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=Diagnosis,
    )

    user_message = (
        f"Repository: {run_metadata.get('repository_name')}\n"
        f"Workflow: {workflow_path}\n"
        f"Commit message: {run_metadata.get('metadata_head_commit_message')}\n"
        f"Failing job: {job.get('name')}\n\n"
        f"Diagnose this CI/CD failure. Use your tools to gather evidence first."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    structured = result.get("structured_response")
    if structured is None:
        diagnosis = {
            "root_cause": "LangChain agent did not return a structured diagnosis.",
            "category": "unknown",
            "evidence_used": [],
            "suggested_fix": "n/a",
            "confidence": "low",
        }
    else:
        diagnosis = structured.model_dump()

    diagnosis["failing_job"] = job.get("name")
    return diagnosis
