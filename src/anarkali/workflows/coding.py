"""Canonical coding-workflow schema: fixed question ids, plus seeded
latent-factor samplers and renderers that turn factors into realistic JSON states.

Every workflow has three questions (choice / noul / score) and every state factor
moves at least one gold answer, so no single option can dominate when factors vary
widely. Importing this module does not load torch.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

WORKFLOWS: dict[str, dict[str, dict]] = {
    "coding_ci_failure": {
        "cause": {
            "type": "choice",
            "instructions": "What most likely caused this CI failure?",
            "criteria": {
                "flaky_test": "A test that passes and fails on identical code, due to timing, ordering, or external services.",
                "real_regression": "A recent code change broke behavior; the failure is deterministic and points at the diff.",
                "infra_outage": "The failure comes from CI infrastructure, network, or runners, not from the code under test.",
                "dependency_change": "A new or bumped dependency changed behavior and triggered the failure.",
                "config_error": "A build, workflow, or environment configuration mistake made the job fail.",
            },
        },
        "action": {
            "type": "choice",
            "instructions": "What should the on-call engineer do next?",
            "criteria": {
                "rerun": "Trigger a fresh run of the same job, appropriate when the evidence points to flakiness or infra.",
                "fix_code": "Reproduce, patch the regression, and push a fix rather than retrying.",
                "quarantine_test": "Mark the unstable test as skipped or quarantined so the branch is not blocked while a fix is prepared.",
                "escalate_infra": "Page or file with the platform/infrastructure team because the pipeline itself is broken.",
                "revert_commit": "Undo the offending commit to restore a green mainline before investigating at leisure.",
            },
        },
        "blocks_release": {
            "type": "noul",
            "instructions": "This failure blocks a release until it is resolved.",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is a response? Levels: 0 none, 1 routine, 2 same week, 3 same day.",
            "criteria": [
                "No response needed; nothing is blocked and nobody should act now.",
                "Routine: handle in normal working order within the week.",
                "Same week: the team should resolve it within days, it blocks others.",
                "Same day: mainline or release is broken; act immediately.",
            ],
        },
    },
    "coding_pr_triage": {
        "review_decision": {
            "type": "choice",
            "instructions": "What review decision should a maintainer give this pull request?",
            "criteria": {
                "approve": "The change is safe and complete; approve without further conditions.",
                "request_changes": "Ask the author for specific revisions before it can merge.",
                "needs_senior_review": "Escalate to a senior or domain owner; risk or subtlety exceeds routine review.",
                "split_pr": "The change is too large or mixes concerns; ask the author to split it into smaller PRs.",
            },
        },
        "category": {
            "type": "choice",
            "instructions": "Which category best describes this pull request?",
            "criteria": {
                "feature": "Adds user-visible functionality or a new capability.",
                "bugfix": "Fixes incorrect existing behavior without changing intended behavior.",
                "refactor": "Restructures code with no intended behavior change.",
                "docs": "Documentation only; no runtime code change.",
                "dependency": "Adds, removes, or upgrades dependencies.",
                "test": "Adds or changes tests only.",
            },
        },
        "needs_security_review": {
            "type": "noul",
            "instructions": "This pull request needs a dedicated security review.",
        },
        "risk": {
            "type": "score",
            "instructions": "How risky is merging this as-is? Levels: 0 trivial, 1 low, 2 moderate, 3 high.",
            "criteria": [
                "Trivial: no meaningful chance of harm, e.g. docs or comments.",
                "Low: small surface, easy to revert, no critical paths touched.",
                "Moderate: touches real behavior or shared components; regressions are plausible.",
                "High: security, data integrity, money, or core paths; a bug would be costly.",
            ],
        },
    },
    "coding_agent_step": {
        "next_action": {
            "type": "choice",
            "instructions": "What should the autonomous coding agent do next?",
            "criteria": {
                "continue": "The step succeeded; keep going with the task.",
                "retry_with_fix": "Retry the failed action with a concrete adjustment, such as a corrected command or file.",
                "revert_last_change": "Undo the most recent edit because it caused the failure and a different approach is needed.",
                "ask_human": "Stop and ask the operator; requirements are ambiguous, permissions are missing, or repeated attempts failed.",
                "stop_done": "The task's acceptance criteria are met; finish.",
            },
        },
        "constraint_violation": {
            "type": "noul",
            "instructions": "This step violated one of the task's stated constraints.",
        },
        "progress": {
            "type": "score",
            "instructions": "How much of the task is complete? Levels: 0 none, 1 started, 2 mostly done, 3 done.",
            "criteria": [
                "None: no meaningful progress toward the objective.",
                "Started: some groundwork exists but the objective is far off.",
                "Mostly done: the core work exists; only verification or polish remains.",
                "Done: the objective and its acceptance criteria are met.",
            ],
        },
    },
}

QUESTIONS: dict[str, dict[str, dict]] = {
    workflow: {qid: {"type": spec["type"], "instructions": spec["instructions"]}
               | ({"criteria": spec["criteria"]} if spec["type"] != "noul" else {})
               for qid, spec in questions.items()}
    for workflow, questions in WORKFLOWS.items()
}

# ---------------------------------------------------------------- latent factors

REPOS = ["acme/checkout", "acme/inventory", "globex/web", "globex/api", "initech/billing",
         "initech/mobile", "umbrella/ml-pipeline", "umbrella/docs-site"]
LANGUAGES = ["python", "javascript", "typescript", "go", "rust"]
CI_JOBS = ["test", "lint", "build", "integration"]
PYTEST_ERRORS = [
    "E       assert 200 == 500\nE        +  where 200 = requests.post(url).status_code",
    "FAILED tests/test_cart.py::test_apply_discount - assert Decimal('9.99') == Decimal('10.99')",
    "E       TimeoutError: timed out after 30.0s waiting on future",
    "E       sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not connect to server",
]
JEST_ERRORS = [
    "FAIL src/components/Checkout.test.tsx\n  ● renders total › expects formatted total\n    Expected: \"$42.00\" Received: \"$0.00\"",
    "thrown: \"Exceeded timeout of 5000 ms for a test\"",
    "Cannot find module 'src/lib/pricing' from 'src/components/Receipt.test.tsx'",
]
GO_TEST_ERRORS = [
    "--- FAIL: TestCalculateTotals (0.00s)\n    totals_test.go:42: got 100, want 120",
    "panic: runtime error: index out of range [5] with length 3 [recovered]",
    "--- FAIL: TestFlakyEndpoint (0.31s)\n    endpoint_test.go:88: Get \"http://127.0.0.1:0/health\": dial tcp: connection refused",
]
CARGO_ERRORS = [
    "error[E0308]: mismatched types: expected `u64`, found `usize`",
    "test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured",
    "error: linking with `cc` failed: exit status: 1",
]
DOCKER_ERRORS = [
    "ERROR [build 4/5] RUN go build ./...: exit code 1",
    "error: failed to solve: failed to fetch oauth token for registry.hub.docker.com",
    "E: Unable to locate package libpq-dev (apt-get update failed)",
]
AUTHORS = [("maya-oss", 84), ("dev-chen", 30), ("intern-sam", 2), ("core-ana", 55), ("new-jo", 1)]
TEST_TOUCHING_PATHS = ["tests/", "src/conftest.py", "pytest.ini", "jest.config.ts", "go.mod"]
SECURITY_PATHS = ["auth/", "crypto/", "payments/", ".github/workflows"]

FLAKY_JOBS = {"integration": 0.35, "test": 0.15, "lint": 0.02, "build": 0.05}


def _sample_ci_factors(rng: random.Random) -> dict[str, Any]:
    language = rng.choice(LANGUAGES)
    job = rng.choice(CI_JOBS)
    flake_bias = FLAKY_JOBS[job] + rng.uniform(-0.05, 0.05)
    cause = rng.random()
    if cause < 0.22:
        latent_cause = "flaky_test"
    elif cause < 0.46:
        latent_cause = "real_regression"
    elif cause < 0.62:
        latent_cause = "infra_outage"
    elif cause < 0.82:
        latent_cause = "dependency_change"
    else:
        latent_cause = "config_error"
    retry_history = {"retries": rng.randint(0, 3),
                     "outcomes": [rng.choice(["failed", "failed", "passed"]) for _ in range(rng.randint(0, 3))]}
    # One rerun already passed => strong flake or infra signal; repeated identical
    # failures on the same line => regression; log text carries config/dependency cues.
    same_line = latent_cause in ("real_regression", "config_error", "dependency_change")
    flake_rate = round(min(0.9, max(0.0, flake_bias + (0.3 if latent_cause == "flaky_test" else -0.2)
                                    + rng.uniform(-0.1, 0.1))), 2)
    recent_commits = []
    if latent_cause == "real_regression":
        recent_commits = [{"sha": rng.choice(["f3a19c2", "b84d0e1", "9cc2f7a"]), "message": "Implement requested pricing change",
                           "touches_tests": rng.random() < 0.3}]
    elif latent_cause == "dependency_change":
        recent_commits = [{"sha": rng.choice(["aa11b2c", "de99f01"]), "message": "Bump " + rng.choice(["lodash", "react", "sqlalchemy", "numpy"]) + " to " + rng.choice(["4.17.21", "5.4.3", "2.0.1", "1.26.4"])}]
    else:
        recent_commits = [{"sha": "7d21e0b", "message": rng.choice(["Refactor queue helpers", "Update docs", "Add test fixtures", " chore: lint"]), "touches_tests": rng.random() < 0.5}]
    error_excerpt = rng.choice({"python": PYTEST_ERRORS, "javascript": JEST_ERRORS, "typescript": JEST_ERRORS,
                                "go": GO_TEST_ERRORS, "rust": CARGO_ERRORS}[language] + DOCKER_ERRORS)
    if latent_cause == "config_error" and "exit code" not in error_excerpt and "Unable to locate" not in error_excerpt:
        error_excerpt = rng.choice(DOCKER_ERRORS)
    return {
        "latent_cause": latent_cause,
        "repo": rng.choice(REPOS),
        "language": language,
        "job": job,
        "branch": rng.choice(["main", "release/2.3", "feature/pricing-v2", "hotfix/session-leak"]),
        "failed_step": rng.choice(["unit tests", "build binary", "docker build", "lint", "integration suite"]),
        "error_excerpt": error_excerpt,
        "same_failure_line_across_retries": same_line,
        "retry_history": retry_history,
        "flake_rate_30d": flake_rate,
        "infra_status": rng.choice(["healthy"] * 6 + ["runner pool degraded", "network incidents"] * (3 if latent_cause == "infra_outage" else 1)),
        "recent_commits": recent_commits,
        "queued_releases": rng.randint(0, 4),
        "open_incidents": rng.randint(0, 2) if latent_cause == "infra_outage" else rng.randint(0, 1),
        "main_branch_broken": latent_cause in ("real_regression", "config_error") and rng.random() < 0.7,
    }


def _sample_pr_factors(rng: random.Random) -> dict[str, Any]:
    category = rng.choice(["feature", "bugfix", "refactor", "docs", "dependency", "test"])
    files = []
    touches_security = category in ("feature", "bugfix") and rng.random() < 0.5
    touches_tests = category == "test" or rng.random() < 0.6
    for _ in range(rng.randint(1, 12)):
        if touches_security and rng.random() < 0.6:
            files.append(rng.choice(SECURITY_PATHS) + rng.choice(["login.py", "token.go", "signer.rs", "deploy.yml"]))
        elif touches_tests and rng.random() < 0.5:
            files.append(rng.choice(TEST_TOUCHING_PATHS) + rng.choice(["test_pricing.py", "cart.spec.ts", "totals_test.go"]))
        else:
            files.append(rng.choice(["src/", "lib/", "app/", "services/"]) + rng.choice(["models", "handlers", "views", "routes", "utils"]) + rng.choice([".py", ".ts", ".go"]))
    additions = rng.randint(2, 900)
    deletions = rng.randint(0, min(additions, 400))
    big = additions + deletions > 500
    tenure = rng.choice(AUTHORS)
    return {
        "latent_category": category,
        "title": rng.choice(["Add coupon codes", "Fix rounding on refunds", "Refactor session store", "Bump numpy to 2.0",
                             "Add checkout integration tests", "Document retry policy", "Rotate signing keys", "Migrate config loader"]),
        "description": rng.choice(["", "Adds coverage for the pricing edge cases discussed in standup.",
                                   "Implements the feature requested in #142.",
                                   "No behavior change intended; mechanical rename."]),
        "changed_files": files,
        "additions": additions,
        "deletions": deletions,
        "touches_tests": touches_tests,
        "touches_security_paths": touches_security,
        "security_paths": [f for f in files if any(f.startswith(p) for p in SECURITY_PATHS)],
        "ci_status": rng.choice(["passing"] * 5 + ["failing", "pending"]),
        "author_tenure_days": tenure[1],
        "author": tenure[0],
        "linked_issue": rng.choice([None, None, "#142", "#318", "#77"]),
        "size_flag": "large" if big else "normal",
        "reviews_requested": rng.randint(1, 3),
    }


def _sample_agent_factors(rng: random.Random) -> dict[str, Any]:
    done_ratio = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    step = rng.randint(1, 25)
    max_steps = step + rng.randint(0, 15)
    failed = rng.random() < 0.45
    attempts = rng.randint(1, 4)
    exit_code = 1 if failed else 0
    touched = rng.randint(1, 6)
    diff = rng.randint(5, 400)
    violated = rng.random() < 0.12
    constraints = ["Never push to main directly", "Do not modify files under secrets/",
                   "Stay within a $10 budget for API calls", "Do not run tests longer than 5 minutes"]
    violation = rng.choice(constraints) if violated else None
    near_limit = step >= max_steps - 2
    last_action = rng.choice(["run_tests", "edit_file", "run_command", "read_file", "git_commit", "install_package"])
    return {
        "latent_done_ratio": done_ratio,
        "task": rng.choice(["Add rate limiting to the public API", "Fix flaky checkout test", "Migrate config to pydantic",
                            "Upgrade React and fix breaking changes", "Implement CSV export"]),
        "constraints": constraints + ([f"REQUIRED: {violation}"] if violation else []),
        "last_tool_action": last_action,
        "result": {"exit_code": exit_code,
                   "error": None if not failed else rng.choice([
                       "ModuleNotFoundError: No module named 'requests'", "assert 3 == 4",
                       "ERROR: failed to compile native extension", "port 8080 already in use"]),
                   "tests_passed": 0 if failed and last_action == "run_tests" else rng.randint(0, 40),
                   "tests_failed": rng.randint(1, 5) if failed and last_action == "run_tests" else 0,
                   "stdout_excerpt": None if failed else "OK"},
        "consecutive_failures": attempts if failed else 0,
        "attempt_count": attempts,
        "step": step,
        "max_steps": max_steps,
        "steps_remaining": max(0, max_steps - step),
        "files_touched": touched,
        "diff_lines": diff,
        "constraint_violation": violation,
        "acceptance_criteria": rng.choice([
            "All tests pass and lint is clean",
            "Feature works and docs updated",
            "Config loads and app boots",
        ]),
        "remaining_work": rng.choice(["finish tests", "update docs", "none", "polish edge cases", "verify on staging"]),
        "budget_remaining_usd": round(rng.uniform(0.5, 9.0), 2),
    }


SAMPLERS = {"coding_ci_failure": _sample_ci_factors,
            "coding_pr_triage": _sample_pr_factors,
            "coding_agent_step": _sample_agent_factors}

SECURITY_PREFIXES = tuple(SECURITY_PATHS)


# ---------------------------------------------------------------- gold answers

def gold_answers(workflow: str, factors: dict[str, Any]) -> dict[str, Any]:
    """Derive gold distributions from latent factors. Deterministic in the factors."""
    # sha256, not hash(): str hashing is salted per process, which made gold labels vary between runs.
    digest = hashlib.sha256(repr((workflow, _freeze(factors))).encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    gold: dict[str, Any] = {}
    if workflow == "coding_ci_failure":
        cause = factors["latent_cause"]
        same_line = factors["same_failure_line_across_retries"]
        reruns_passed = "passed" in factors["retry_history"]["outcomes"]
        if reruns_passed and not same_line:
            cause = "flaky_test"
        elif cause == "flaky_test" and same_line and factors["flake_rate_30d"] < 0.1:
            cause = "real_regression"
        cause_probs = {k: 0.08 for k in ("flaky_test", "real_regression", "infra_outage", "dependency_change", "config_error")}
        cause_probs[cause] = 0.6
        gold["cause"] = _sharpen(cause_probs, rng)
        rerun_score = 0.9 if cause in ("flaky_test", "infra_outage") else 0.15
        quarantine_score = 0.8 if cause == "flaky_test" and factors["job"] == "integration" else 0.1
        revert_score = 0.85 if cause == "real_regression" and factors["main_branch_broken"] else 0.1
        action_probs = {"rerun": rerun_score, "fix_code": 0.8 if cause == "real_regression" else 0.15,
                        "quarantine_test": quarantine_score,
                        "escalate_infra": 0.85 if cause == "infra_outage" else 0.05,
                        "revert_commit": revert_score}
        gold["action"] = _sharpen(_softmax_probs(action_probs), rng)
        blocks = (cause in ("real_regression", "config_error") and factors["main_branch_broken"]) \
            or (cause == "dependency_change" and factors["queued_releases"] >= 2)
        gold["blocks_release"] = {"true": 0.9 if blocks else 0.08, "false": 0.9 if not blocks else 0.08}
        gold["blocks_release"] = _sharpen(gold["blocks_release"], rng)
        urgency = 3 if (blocks and factors["queued_releases"] >= 1) else \
            2 if blocks or factors["queued_releases"] >= 3 else 1 if cause != "flaky_test" else 0
        levels = ["none", "routine", "same week", "same day"]
        gold["urgency"] = _ordinal(urgency, 4)
        assert levels[urgency]
    elif workflow == "coding_pr_triage":
        category = factors["latent_category"]
        cat_probs = {k: 0.06 for k in ("feature", "bugfix", "refactor", "docs", "dependency", "test")}
        cat_probs[category] = 0.64
        gold["category"] = _sharpen(cat_probs, rng)
        security = factors["touches_security_paths"] or bool(factors["security_paths"])
        risky = security or factors["ci_status"] == "failing" or factors["author_tenure_days"] < 7
        big = factors["size_flag"] == "large"
        if security:
            review = "needs_senior_review"
        elif big and factors["additions"] > 200:
            review = "split_pr"
        elif factors["ci_status"] == "failing":
            review = "request_changes"
        else:
            review = "approve"
        review_probs = {k: 0.06 for k in ("approve", "request_changes", "needs_senior_review", "split_pr")}
        review_probs[review] = 0.62
        gold["review_decision"] = _sharpen(review_probs, rng)
        gold["needs_security_review"] = _sharpen({"true": 0.9 if security else 0.05,
                                                  "false": 0.9 if not security else 0.05}, rng)
        risk = 3 if (security and risky) else 2 if risky or big else 1 if category not in ("docs",) else 0
        gold["risk"] = _ordinal(risk, 4)
    else:
        done_ratio = factors["latent_done_ratio"]
        failures = factors["consecutive_failures"]
        violation = factors["constraint_violation"]
        violated = violation is not None
        if factors["result"]["exit_code"] == 0 and done_ratio >= 0.99 and not factors["remaining_work"]:
            next_action = "stop_done"
        elif factors["result"]["exit_code"] != 0 and failures >= 3:
            next_action = "ask_human"
        elif factors["result"]["exit_code"] != 0:
            next_action = "retry_with_fix"
        elif done_ratio >= 0.8:
            next_action = "stop_done" if done_ratio >= 0.99 else "continue"
        elif factors["steps_remaining"] == 0:
            next_action = "ask_human"
        else:
            next_action = "continue"
        action_probs = {k: 0.06 for k in ("continue", "retry_with_fix", "revert_last_change", "ask_human", "stop_done")}
        action_probs[next_action] = 0.62
        gold["next_action"] = _sharpen(action_probs, rng)
        gold["constraint_violation"] = _sharpen({"true": 0.9 if violated else 0.05,
                                                 "false": 0.9 if not violated else 0.05}, rng)
        progress = round(done_ratio * 3)
        gold["progress"] = _ordinal(progress, 4)
    return gold


def _softmax_probs(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _sharpen(probs: dict[str, float], rng: random.Random) -> dict[str, float]:
    """Add mild teacher-like noise so targets are not razor sharp."""
    keys = list(probs)
    values = [max(0.0, probs[k] + rng.uniform(-0.04, 0.04)) for k in keys]
    total = sum(values) or 1.0
    values = [v / total for v in values]
    floor = 0.01
    values = [max(floor, v) for v in values]
    total = sum(values)
    return {k: v / total for k, v in zip(keys, values)}


def _ordinal(level: int, n: int) -> dict[str, float]:
    probs = [0.05] * n
    probs[level] = 0.7
    if level + 1 < n:
        probs[level + 1] = 0.15
    if level - 1 >= 0:
        probs[level - 1] += 0.05
    total = sum(probs)
    return {str(i): p / total for i, p in enumerate(probs)}


def _freeze(value: Any):
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


# ---------------------------------------------------------------- renderers

def _render_ci_failure(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo": f["repo"], "language": f["language"], "job": f["job"],
        "branch": f["branch"],
        "failed_step": f["failed_step"],
        "error_excerpt": f["error_excerpt"],
        "same_failure_line_across_retries": f["same_failure_line_across_retries"],
        "retry_history": f["retry_history"],
        "flake_rate_30d": f["flake_rate_30d"],
        "infra_status": f["infra_status"],
        "recent_commits": f["recent_commits"],
        "queued_releases": f["queued_releases"],
        "open_incidents": f["open_incidents"],
        "main_branch_broken": f["main_branch_broken"],
    }


def _render_pr_triage(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": f["title"], "description": f["description"],
        "changed_files": f["changed_files"], "additions": f["additions"], "deletions": f["deletions"],
        "touches_tests": f["touches_tests"],
        "touches_security_paths": bool(f["security_paths"]),
        "security_paths": f["security_paths"],
        "ci_status": f["ci_status"], "author": f["author"],
        "author_tenure_days": f["author_tenure_days"],
        "linked_issue": f["linked_issue"], "size_flag": f["size_flag"],
        "reviews_requested": f["reviews_requested"],
    }


def _render_agent_step(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": f["task"], "constraints": f["constraints"],
        "last_tool_action": f["last_tool_action"], "result": f["result"],
        "consecutive_failures": f["consecutive_failures"], "attempt_count": f["attempt_count"],
        "step": f["step"], "max_steps": f["max_steps"],
        "files_touched": f["files_touched"], "diff_lines": f["diff_lines"],
        "acceptance_criteria": f["acceptance_criteria"], "remaining_work": f["remaining_work"],
        "budget_remaining_usd": f["budget_remaining_usd"],
    }


RENDERERS = {"coding_ci_failure": _render_ci_failure,
             "coding_pr_triage": _render_pr_triage,
             "coding_agent_step": _render_agent_step}


def sample_case(workflow: str, seed: str) -> tuple[str, dict[str, Any], dict[str, dict], dict[str, Any]]:
    """One (case_id, state, questions, gold) tuple; deterministic given `seed`."""
    if workflow not in SAMPLERS:
        raise ValueError(f"unknown workflow {workflow!r}")
    rng = random.Random(seed)
    factors = SAMPLERS[workflow](rng)
    state = RENDERERS[workflow](factors)
    gold = gold_answers(workflow, factors)
    case_id = f"{workflow}::{seed}"
    return case_id, state, QUESTIONS[workflow], gold


__all__ = ["WORKFLOWS", "QUESTIONS", "SAMPLERS", "RENDERERS", "SECURITY_PREFIXES", "sample_case", "gold_answers"]
