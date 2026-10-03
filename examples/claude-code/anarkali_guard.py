#!/usr/bin/env python3
"""Claude Code PreToolUse hook: ask Anarkali whether a tool call breaks the task's rules.

Claude Code runs this before every matching tool call and sends the call as JSON on stdin.
The hook asks Anarkali's coding_agent_step `constraint_violation` question about it and
answers with a permission decision: deny at or above ANARKALI_DENY (default 0.8), ask the
user at or above ANARKALI_ASK (default 0.5), otherwise stay silent so the normal
permission flow applies. If Anarkali is unreachable the hook fails open and says so on stderr.

Configuration (environment):
    ANARKALI_URL          a running `anarkali serve` (fast: the model stays loaded), or
    ANARKALI_MODEL        a model directory or Hugging Face id to load in-process (slow per call)
    ANARKALI_TASK         what the agent was asked to do (optional)
    ANARKALI_CONSTRAINTS  path to a text file, one rule per line (optional; defaults below)
    ANARKALI_DENY, ANARKALI_ASK   probability thresholds

Recent actions are kept in .anarkali/guard-log.jsonl under the project directory, so the
model sees the last few steps as well as this one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import urllib.request

DEFAULT_CONSTRAINTS = [
    "Never push to any remote; never force-push.",
    "Do not delete files outside the repository.",
    "Do not weaken or delete existing tests to make them pass.",
    "Do not read or print credentials, keys or .env files.",
]
QUESTION = {
    "type": "noul",
    "instructions": "This step violated one of the task's stated constraints.",
}


def describe(tool_name: str, tool_input: dict) -> str:
    """One line for the tool call, in the shape the agent-trace importer produces."""
    if tool_name == "Bash":
        return str(tool_input.get("command", ""))
    if tool_name in ("Edit", "MultiEdit", "Write", "NotebookEdit"):
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        body = tool_input.get("new_string") or tool_input.get("content") or tool_input.get("new_source") or ""
        return f"{tool_name.lower()}({path}) {str(body)[:300]}"
    args = ", ".join(f"{k}={str(v)[:120]!r}" for k, v in tool_input.items())
    return f"{tool_name}({args})"


def build_state(event: dict, recent: list[str], constraints: list[str], task: str | None) -> dict:
    return {
        "task": task or f"Work in {event.get('cwd', 'the repository')} as the user asked.",
        "constraints": constraints,
        "acceptance_criteria": "The user's request is done without breaking any constraint.",
        "recent_actions": recent[-3:],
        "last_tool_action": describe(event.get("tool_name", ""), event.get("tool_input") or {})[:400],
        "result": {"observation": "", "looks_like_error": False},
        "step": len(recent) + 1,
    }


def decide(probability: float, deny_at: float, ask_at: float, action: str) -> dict | None:
    """Hook output for a violation probability; None leaves the call to the normal flow."""
    if probability >= deny_at:
        decision = "deny"
    elif probability >= ask_at:
        decision = "ask"
    else:
        return None
    return {"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": (f"Anarkali: {probability:.0%} likely to break a task constraint: "
                                     f"{action[:160]}"),
    }}


def predict_http(url: str, state: dict) -> float:
    body = json.dumps({"state": state, "questions": {"violation": QUESTION}}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("ANARKALI_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['ANARKALI_API_KEY']}"
    request = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return float(json.loads(response.read())["answers"]["violation"]["noul"])


def predict_local(model: str, state: dict) -> float:
    from anarkali import Engine
    return float(Engine.load(model).predict(state, {"violation": QUESTION})["answers"]["violation"]["noul"])


def load_constraints() -> list[str]:
    path = os.environ.get("ANARKALI_CONSTRAINTS")
    if not path:
        return DEFAULT_CONSTRAINTS
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run(event: dict, predict, *, log_dir: Path, deny_at: float, ask_at: float) -> dict | None:
    log = log_dir / "guard-log.jsonl"
    recent = []
    if log.exists():
        with log.open(encoding="utf-8") as stream:
            recent = [json.loads(line)["action"] for line in stream if line.strip()][-10:]
    state = build_state(event, recent, load_constraints(), os.environ.get("ANARKALI_TASK"))
    probability = predict(state)
    log_dir.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"action": state["last_tool_action"], "p_violation": probability}) + "\n")
    return decide(probability, deny_at, ask_at, state["last_tool_action"])


def main() -> int:
    event = json.load(sys.stdin)
    url, model = os.environ.get("ANARKALI_URL"), os.environ.get("ANARKALI_MODEL")
    if not url and not model:
        print("anarkali guard: set ANARKALI_URL or ANARKALI_MODEL", file=sys.stderr)
        return 0
    predict = (lambda state: predict_http(url, state)) if url else (lambda state: predict_local(model, state))
    try:
        output = run(event, predict, log_dir=Path(event.get("cwd") or ".") / ".anarkali",
                     deny_at=float(os.environ.get("ANARKALI_DENY", 0.8)),
                     ask_at=float(os.environ.get("ANARKALI_ASK", 0.5)))
    except Exception as exc:  # fail open: a broken guard must not stop the agent silently
        print(f"anarkali guard unavailable, allowing the call: {exc}", file=sys.stderr)
        return 0
    if output:
        print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
