# Anarkali as a guardrail for Claude Code

`anarkali_guard.py` is a Claude Code `PreToolUse` hook. Before each `Bash`, `Edit`, `MultiEdit` or `Write` call, it asks Anarkali whether the call breaks one of the task's rules, then acts on the probability:

| Violation probability | What happens |
|---|---|
| ≥ `ANARKALI_DENY` (0.8) | The call is denied, and Claude sees the reason. |
| ≥ `ANARKALI_ASK` (0.5) | You are asked to confirm. |
| lower | Nothing. The normal permission flow applies. |

```bash
pip install "anarkali[serve] @ git+https://github.com/ToufiqQureshi/anarkali"
anarkali serve --model toufiqqureshi651/anarkali --port 8000 &   # keep the model loaded
export ANARKALI_URL=http://localhost:8000
```

Then merge `settings.json` into your project's `.claude/settings.json`. Rules default to "no pushes, no deletes outside the repo, no weakening tests, no reading secrets". To use your own, list one rule per line in a text file and set `ANARKALI_CONSTRAINTS` to its path.

The hook fails open: if the server is down, it prints a warning on stderr and allows the call.

**Status: demo.** The released 0.3.0 model learned `coding_agent_step` from synthetic, rule-labelled cases only. Train on real agent traces (`scripts/import_agent_traces.py`, V4 notebook) and measure it on hand-labelled steps before you rely on it. Until then, keep Claude Code's own permission prompts on.
