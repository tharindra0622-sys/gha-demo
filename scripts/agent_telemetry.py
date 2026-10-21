#!/usr/bin/env python3
"""
agent_telemetry.py

Small helper that turns a LangChain message list into the numbers the
dashboard's "Agent Performance" tab needs: how many LLM calls were made,
how many tokens were used, which tools were called, and how many tool
calls failed.

Nothing here changes how an agent behaves. It only reads the messages the
agent already produced.
"""

# The seven read-only diagnostic tools. Any tool call outside this list and
# outside STRUCTURED_OUTPUT_NAMES is recorded as an "unknown" tool call and
# counted as unsafe by the dashboard.
KNOWN_TOOLS = {
    "get_failure_log", "search_repo_issues", "get_workflow_file",
    "check_github_status", "search_web_for_error", "check_registry_status",
    "get_recent_commits",
}

# LangChain may implement structured output as a "tool call" named after the
# schema. Those are not diagnostic tool calls, so they are ignored.
STRUCTURED_OUTPUT_NAMES = {
    "Diagnosis", "LogFindings", "ChangeFindings", "ExternalFindings", "HistoryFindings",
}

EMPTY = {
    "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
    "tool_calls": 0, "tool_errors": 0, "tools_called": [], "unknown_tool_calls": [],
}


def looks_like_error(text):
    """Our tools report problems as text such as '(search failed: 403)'."""
    t = str(text).strip().lower()
    return t.startswith("(") and ("failed" in t[:90] or "could not" in t[:90])


def message_telemetry(messages):
    """Count LLM calls, tokens and tool use in a list of LangChain messages."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in EMPTY.items()}
    for m in messages or []:
        kind = getattr(m, "type", "")
        if kind == "ai":
            out["llm_calls"] += 1
            usage = getattr(m, "usage_metadata", None) or {}
            out["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            out["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
            for call in getattr(m, "tool_calls", None) or []:
                name = call.get("name")
                if name in KNOWN_TOOLS:
                    out["tool_calls"] += 1
                    out["tools_called"].append(name)
                elif name not in STRUCTURED_OUTPUT_NAMES:
                    out["unknown_tool_calls"].append(name)
        elif kind == "tool":
            name = getattr(m, "name", None)
            if name in KNOWN_TOOLS:
                if getattr(m, "status", "success") == "error" or looks_like_error(getattr(m, "content", "")):
                    out["tool_errors"] += 1
    return out


def usage_only(message):
    """Tokens and one LLM call from a single AI message (used for the
    coordinator, whose structured output is not a diagnostic tool call)."""
    usage = getattr(message, "usage_metadata", None) or {}
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in EMPTY.items()}
    out["llm_calls"] = 1
    out["input_tokens"] = int(usage.get("input_tokens", 0) or 0)
    out["output_tokens"] = int(usage.get("output_tokens", 0) or 0)
    return out


def merge(*parts):
    """Add several telemetry dicts together."""
    total = {k: (list(v) if isinstance(v, list) else v) for k, v in EMPTY.items()}
    for p in parts:
        if not p:
            continue
        for k, v in p.items():
            if isinstance(v, list):
                total[k] = total.get(k, []) + v
            else:
                total[k] = total.get(k, 0) + v
    return total
