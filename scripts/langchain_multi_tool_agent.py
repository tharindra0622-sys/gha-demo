#!/usr/bin/env python3
"""
langchain_multi_tool_agent_v2.py

Expanded version of the multi-tool CI/CD diagnosis agent. Adds three new
tools on top of the original four, so the agent can search a much wider
range of possible causes instead of being limited to this repo's own
data:

Original 4 tools:
  1. get_failure_log        — the raw error text
  2. search_repo_issues     — this repo's own past issues
  3. get_workflow_file      — what the step was supposed to do
  4. check_github_status    — is GitHub's own platform having problems

New 3 tools:
  5. search_web_for_error   — searches the open web for the exact error
                               text, the same way a developer would search
                               Stack Overflow or a GitHub Discussions
                               thread. This is what lets the agent
                               recognise failure types it has never seen
                               in this specific repo before.
  6. check_registry_status  — checks whether PyPI (Python) or the npm
                               registry (Node.js) is itself having an
                               outage, which can look identical to a
                               "missing dependency" error but has a
                               completely different real cause.
  7. get_recent_commits     — looks at the last few commits on the branch,
                               so the agent can reason about whether a
                               recent code change is the likely cause,
                               not just the single commit that triggered
                               this run.

Design principle unchanged: this agent only ever produces a
recommendation. It never modifies code or applies a fix automatically.
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
PYPI_STATUS_API = "https://status.python.org/api/v2/summary.json"
NPM_STATUS_API = "https://status.npmjs.org/api/v2/summary.json"


def gh_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


class Diagnosis(BaseModel):
    root_cause: str = Field(description="Plain-language explanation, referencing which evidence supports it")
    category: Literal[
        "dependency_error", "test_failure", "syntax_error", "config_error",
        "network_timeout", "permission_error", "resource_limit",
        "flaky_test", "infrastructure_issue", "registry_outage",
        "recent_code_change", "unknown",
    ]
    evidence_used: List[str] = Field(description="Which tools/sources actually informed this diagnosis")
    suggested_fix: str = Field(description="A specific, actionable fix recommendation")
    confidence: Literal["high", "medium", "low"]


def build_tools(owner, repo, github_token, job_id, workflow_path, head_sha, run_started_at, head_branch):

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
        text, to check whether this failure has happened before and how
        it was handled."""
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
        real cause rather than the user's code. Reflects CURRENT status
        only, not historical status at the time of the failure."""
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
                f"NOTE: reflects status at the time of this check, not "
                f"necessarily at the time the run failed ({run_started_at}); "
                f"treat as a weak, supporting signal only."
            )
        except Exception as e:
            return f"(could not check GitHub status: {e})"

    @tool
    def search_web_for_error(error_text: str) -> str:
        """Search the open web for the exact error message or a short,
        distinctive fragment of it, the same way a developer would search
        Stack Overflow or a forum. Use this for errors that don't look
        specific to this repository — e.g. a library's own error message,
        a known bug in a tool, or a common configuration mistake. Pass a
        short, specific fragment of the error text (a few words to one
        line), not the whole log."""
        try:
            from ddgs import DDGS
            results = list(DDGS().text(error_text, max_results=5))
            if not results:
                return "No web results found for this error text."
            lines = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")[:200]
                lines.append(f"- {title}: {body}")
            return "\n".join(lines)
        except Exception as e:
            return f"(web search failed: {e})"

    @tool
    def check_registry_status() -> str:
        """Check whether PyPI (Python's package registry) or npm (Node's
        package registry) is currently reporting an outage. Use this when
        a failure looks like a dependency installation problem, since an
        outage on the registry side can look identical to a genuine
        missing-dependency bug in the code, but has a completely
        different real cause and fix."""
        results = []
        for name, url in [("PyPI", PYPI_STATUS_API), ("npm", NPM_STATUS_API)]:
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json()
                indicator = data.get("status", {}).get("indicator", "unknown")
                results.append(f"{name}: current status indicator = {indicator}")
            except Exception as e:
                results.append(f"{name}: could not check status ({e})")
        return "\n".join(results) + (
            "\nNOTE: reflects current status only, not historical status at "
            "the time of the failure."
        )

    @tool
    def get_recent_commits() -> str:
        """Get the last 5 commits on the branch this run happened on, with
        their messages. Use this to check whether a recent code change —
        not just the single commit that triggered this run — is the more
        likely underlying cause of the failure."""
        r = requests.get(
            f"{API}/repos/{owner}/{repo}/commits",
            headers=gh_headers(github_token),
            params={"sha": head_branch, "per_page": 5},
            timeout=30,
        )
        if r.status_code != 200:
            return f"(could not fetch recent commits: {r.status_code})"
        commits = r.json()
        lines = []
        for c in commits:
            sha = c["sha"][:7]
            msg = c["commit"]["message"].split("\n")[0]
            author = c["commit"]["author"]["name"]
            date = c["commit"]["author"]["date"]
            lines.append(f"- {sha} ({date}, {author}): {msg}")
        return "\n".join(lines)

    return [
        get_failure_log, search_repo_issues, get_workflow_file, check_github_status,
        search_web_for_error, check_registry_status, get_recent_commits,
    ]


SYSTEM_PROMPT = """You are a CI/CD failure diagnosis assistant with access to seven tools
covering different possible areas a failure could come from: the failure
log itself, this repository's own history, the workflow's definition,
GitHub's platform status, the open web, package registry status, and
recent code changes.

Do not rely on the failure log alone. Think broadly about where the real
cause could be before settling on an answer:
- ALWAYS start with get_failure_log to see the actual error.
- If the error text looks specific to this project, use search_repo_issues.
- If you need to understand what the step was supposed to do, use
  get_workflow_file.
- If the failure could be caused by GitHub's own infrastructure, use
  check_github_status.
- If the error message looks like a general, well-known error (not
  specific to this repo), use search_web_for_error with a short,
  distinctive fragment of the error text.
- If the failure looks like a dependency installation problem, use
  check_registry_status to rule out a registry-side outage.
- If the failure doesn't obviously match the single commit that
  triggered this run, use get_recent_commits to check whether an earlier
  change is the more likely real cause.

You do not need to call every tool for every failure — use judgement
about which of these seven areas are actually relevant to this specific
failure. You are producing a recommendation for a human to review — you
are NOT applying any change yourself."""


SHAP_TOP_FEATURES = [
    "metadata_event_enc", "is_main_branch", "log_num_jobs",
    "log_early3_action_count", "log_shell_steps",
]


def run_langchain_diagnosis_v2(owner, repo, run_id, job, run_metadata, github_token, anthropic_api_key):
    job_id = job.get("id")
    workflow_path = run_metadata.get("workflow_path")
    head_sha = run_metadata.get("metadata_head_sha")
    head_branch = run_metadata.get("metadata_head_branch", "main")
    run_started_at = run_metadata.get("metadata_run_started_at", "unknown")

    tools = build_tools(owner, repo, github_token, job_id, workflow_path, head_sha, run_started_at, head_branch)

    model = ChatAnthropic(model="claude-sonnet-4-5", api_key=anthropic_api_key, max_tokens=1500)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=Diagnosis,
    )

    # Link Phase 1 (ML prediction) into Phase 2 (LLM diagnosis): give the
    # agent the model's own predicted risk score and the values of the
    # features that most influence that model's predictions in general,
    # for THIS specific run. This is context only — the agent still
    # reaches its own conclusion from the actual log/evidence, it isn't
    # told what to conclude.
    predicted_prob = run_metadata.get("predicted_failure_probability", "n/a")
    feature_summary = ", ".join(
        f"{feat}={run_metadata.get(feat, 'n/a')}" for feat in SHAP_TOP_FEATURES
    )

    user_message = (
        f"Repository: {run_metadata.get('repository_name')}\n"
        f"Workflow: {workflow_path}\n"
        f"Commit message: {run_metadata.get('metadata_head_commit_message')}\n"
        f"Failing job: {job.get('name')}\n\n"
        f"ML model context (from the separate early-prediction model — for your "
        f"reference only, do not treat as ground truth): this run was scored with "
        f"a predicted failure probability of {predicted_prob}. Values of this "
        f"model's most influential features for this specific run: {feature_summary}.\n\n"
        f"Diagnose this CI/CD failure using your own tools and reasoning. Use the "
        f"ML model context above only as a hint about what might be unusual about "
        f"this run, not as the answer itself."
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
