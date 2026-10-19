#!/usr/bin/env python3
"""
multi_agent_diagnosis.py

Multi-agent version of the CI/CD failure diagnosis. Instead of ONE agent
that holds all seven tools, a small team of specialist agents each look at
one area, and a coordinator agent combines their reports into the final
diagnosis.

    Stage 1   Log Analyst            reads the failure log
                   |
    Stage 2   (run in parallel, each receives the Log Analyst's report)
              Config & Change Analyst   workflow file + recent commits
              External Analyst          GitHub / PyPI / npm status + web search
              History Analyst           this repo's past issues
                   |
    Stage 3   Coordinator            weighs all four reports, resolves
                                     disagreements, writes the final
                                     Diagnosis

Design principles (same as the single-agent version):
  * Every agent only RECOMMENDS. Nothing changes any code.
  * The output has exactly the same shape as the single-agent version, so
    control_agent.py, the GitHub Issue and the dashboard keep working.
  * One extra key, "agent_reports", records what each specialist found, so
    the reasoning can be inspected afterwards.

The tools are reused from langchain_multi_tool_agent_v2.py (not rewritten).
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

import langchain_multi_tool_agent_v2 as v2
from langchain_multi_tool_agent_v2 import Diagnosis, SHAP_TOP_FEATURES

MODEL_NAME = "claude-sonnet-4-5"


# --------------------------------------------------------------------------
# What each specialist reports back (small, factual, no final verdict)
# --------------------------------------------------------------------------
class LogFindings(BaseModel):
    failing_step: str = Field(description="Name of the step that failed, if visible in the log")
    error_excerpt: str = Field(description="The 1 to 3 most important error lines, copied exactly from the log")
    error_type: str = Field(description="Short label, e.g. 'package not found', 'assertion failed', 'HTTP 500', 'permission denied'")
    plain_summary: str = Field(description="One or two plain sentences: what the log says went wrong")


class ChangeFindings(BaseModel):
    step_purpose: str = Field(description="What the failing step is meant to do, from the workflow file")
    config_issue_found: bool = Field(description="True if the workflow file itself contains something that would cause this error")
    config_issue: str = Field(description="Describe the configuration problem, or 'none'")
    recent_change_suspected: bool = Field(description="True if a recent commit looks like the real cause")
    recent_change_detail: str = Field(description="Which commit and why, or 'none'")


class ExternalFindings(BaseModel):
    github_platform_issue: bool = Field(description="True only if GitHub reports an active incident that could explain this")
    registry_issue: bool = Field(description="True only if PyPI or npm reports an active problem that could explain this")
    known_issue_on_web: str = Field(description="What the web says about this exact error, or 'nothing relevant found'")
    notes: str = Field(description="Anything else relevant, including that status checks only show the current status")


class HistoryFindings(BaseModel):
    seen_before: bool = Field(description="True if this repository has a similar past issue")
    similar_issues: List[str] = Field(description="Issue numbers and titles, or an empty list")
    notes: str = Field(description="How the past issue was handled, or 'none'")


# --------------------------------------------------------------------------
# Prompts: each specialist has a narrow job and must not give the final answer
# --------------------------------------------------------------------------
LOG_PROMPT = """You are the Log Analyst on a CI/CD failure diagnosis team.
Your only job is to read the failing job's log and report exactly what it says.
Call get_failure_log first. Copy the key error lines exactly. Do not guess at
causes outside the log and do not suggest fixes - other team members handle that."""

CHANGE_PROMPT = """You are the Config & Change Analyst on a CI/CD failure diagnosis team.
You examine the workflow definition and recent commits.
- Use get_workflow_file to see what the failing step is supposed to do and whether
  the file itself contains a mistake.
- Use get_recent_commits to see whether a recent change could be the real cause.
You receive the Log Analyst's report. Report facts only. Do not give the final
diagnosis - a coordinator will do that.
Be strict about recent_change_suspected: set it to true only if a commit message
clearly concerns the code that failed. Commits that only add or edit workflow
files, agent scripts or documentation are NOT a suspected cause of a failure
inside application or test code."""

EXTERNAL_PROMPT = """You are the External Analyst on a CI/CD failure diagnosis team.
Your job is to find out whether the cause is OUTSIDE this repository.
- check_github_status: is GitHub itself having an incident?
- check_registry_status: only if the error is about installing packages.
- search_web_for_error: search with a short, distinctive fragment of the error.
You receive the Log Analyst's report. Remember that status checks show CURRENT
status only, not status at the time of the failure, so treat them as weak
evidence. Do not give the final diagnosis."""

HISTORY_PROMPT = """You are the History Analyst on a CI/CD failure diagnosis team.
Use search_repo_issues (one or two searches with short, distinctive terms from the
error) to find whether this repository has seen a similar failure before.
You receive the Log Analyst's report. Report facts only."""

COORDINATOR_PROMPT = """You are the Coordinator of a CI/CD failure diagnosis team.
Four specialists have reported back. Your job is to write the final diagnosis.

Rules:
- Base the root cause mainly on the Log Analyst's exact error lines.
- Use the other reports to confirm, refine or overturn that first reading
  (for example: a config mistake, an outage, a known issue, or a recent commit).
- If two reports disagree, say which one you trusted and why.
- If a specialist reported an error or no result, do not invent its findings.
- Use the category "recent_code_change" ONLY when a recent commit's message or
  changed files clearly relate to the code that failed. Commits that only touch
  workflow files, scripts, docs or tooling do not explain a failure inside
  application or test code. If the log shows a concrete error in the code (for
  example a TypeError in a named function), choose the category that matches
  that error (such as test_failure), even if recent commits exist.
- Do not tell the reader to "review recent commits" unless you have evidence a
  specific commit caused the problem.
- Treat GitHub / registry status as weak evidence (current status only).
- The ML failure probability is only a hint about the run, never the answer.
- Give a specific, actionable fix. You are making a recommendation for a human;
  nothing is changed automatically.
- In evidence_used, list the sources that genuinely informed the diagnosis."""


# --------------------------------------------------------------------------
# Running one specialist
# --------------------------------------------------------------------------
def run_specialist(name, model, tools, system_prompt, schema, task):
    """Runs one tool-using specialist and returns a plain dict. Never raises:
    a failing specialist must not stop the whole team."""
    started = time.perf_counter()
    try:
        agent = create_agent(
            model=model, tools=tools,
            system_prompt=system_prompt, response_format=schema,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": task}]})
        structured = result.get("structured_response")
        tools_used = sorted({
            m.name for m in result.get("messages", [])
            if getattr(m, "type", "") == "tool" and getattr(m, "name", None)
        })
        seconds = round(time.perf_counter() - started, 1)
        if structured is None:
            return {"agent": name, "ok": False, "error": "no structured output",
                    "tools_used": tools_used, "seconds": seconds}
        return {"agent": name, "ok": True, "findings": structured.model_dump(),
                "tools_used": tools_used, "seconds": seconds}
    except Exception as e:  # noqa: BLE001 - report, don't crash the team
        return {"agent": name, "ok": False, "error": f"{type(e).__name__}: {e}",
                "tools_used": [], "seconds": round(time.perf_counter() - started, 1)}


def coordinate(model, run_context, reports):
    """Stage 3: the coordinator reads every report and writes the Diagnosis."""
    structured_model = model.with_structured_output(Diagnosis)
    reports_text = json.dumps(reports, indent=2, default=str)
    result = structured_model.invoke([
        ("system", COORDINATOR_PROMPT),
        ("user", f"{run_context}\n\nSpecialist reports:\n{reports_text}\n\nWrite the final diagnosis."),
    ])
    return result


# --------------------------------------------------------------------------
# Public entry point (same signature as run_langchain_diagnosis_v2)
# --------------------------------------------------------------------------
def run_multi_agent_diagnosis(owner, repo, run_id, job, run_metadata, github_token, anthropic_api_key):
    job_id = job.get("id")
    workflow_path = run_metadata.get("workflow_path")
    head_sha = run_metadata.get("metadata_head_sha")
    head_branch = run_metadata.get("metadata_head_branch", "main")
    run_started_at = run_metadata.get("metadata_run_started_at", "unknown")

    all_tools = v2.build_tools(
        owner, repo, github_token, job_id, workflow_path,
        head_sha, run_started_at, head_branch,
    )
    by_name = {t.name: t for t in all_tools}
    model = ChatAnthropic(model=MODEL_NAME, api_key=anthropic_api_key, max_tokens=1200)

    predicted_prob = run_metadata.get("predicted_failure_probability", "n/a")
    feature_summary = ", ".join(f"{f}={run_metadata.get(f, 'n/a')}" for f in SHAP_TOP_FEATURES)
    run_context = (
        f"Repository: {run_metadata.get('repository_name')}\n"
        f"Workflow: {workflow_path}\n"
        f"Commit message: {run_metadata.get('metadata_head_commit_message')}\n"
        f"Failing job: {job.get('name')}\n"
        f"ML context (hint only, not ground truth): predicted failure probability "
        f"{predicted_prob}; influential feature values: {feature_summary}."
    )

    # ---- Stage 1: Log Analyst ------------------------------------------------
    log_report = run_specialist(
        "Log Analyst", model, [by_name["get_failure_log"]], LOG_PROMPT, LogFindings,
        f"{run_context}\n\nRead the failing job's log and report what it says.",
    )

    # ---- Stage 2: three specialists in parallel ------------------------------
    handoff = (
        f"{run_context}\n\nLog Analyst's report:\n"
        f"{json.dumps(log_report.get('findings', log_report), indent=2, default=str)}"
    )
    stage2 = [
        ("Config & Change Analyst",
         [by_name["get_workflow_file"], by_name["get_recent_commits"]],
         CHANGE_PROMPT, ChangeFindings),
        ("External Analyst",
         [by_name["check_github_status"], by_name["check_registry_status"],
          by_name["search_web_for_error"]],
         EXTERNAL_PROMPT, ExternalFindings),
        ("History Analyst",
         [by_name["search_repo_issues"]],
         HISTORY_PROMPT, HistoryFindings),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(run_specialist, name, model, tools, prompt, schema, handoff)
            for name, tools, prompt, schema in stage2
        ]
        other_reports = [f.result() for f in futures]

    reports = [log_report] + other_reports

    # ---- Stage 3: Coordinator --------------------------------------------------
    try:
        final = coordinate(model, run_context, reports)
    except Exception as e:  # noqa: BLE001
        final = None
        coordinator_error = f"{type(e).__name__}: {e}"
    else:
        coordinator_error = None

    if final is None:
        diagnosis = {
            "root_cause": f"The coordinator agent did not return a diagnosis ({coordinator_error or 'no output'}).",
            "category": "unknown",
            "evidence_used": [],
            "suggested_fix": "n/a",
            "confidence": "low",
        }
    else:
        diagnosis = final.model_dump()

    # Make evidence_used list the real tools used by the team (same style as
    # the single-agent version), and keep the full per-agent reports.
    tools_used = sorted({t for r in reports for t in r.get("tools_used", [])})
    if tools_used:
        diagnosis["evidence_used"] = tools_used
    diagnosis["agent_reports"] = reports
    diagnosis["mode"] = "multi-agent"
    diagnosis["failing_job"] = job.get("name")
    return diagnosis
