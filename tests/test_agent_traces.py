import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


def load_importer():
    name = "import_agent_traces"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ISSUE = "TypeError in parse_date when the input has a timezone. Fix it and add a test."

SWEAGENT_ROW = {  # nebius/SWE-agent-trajectories: roles system/user/ai, trajectory may be a JSON string
    "instance_id": "acme__dates-12", "model_name": "m", "target": True,
    "trajectory": json.dumps([
        {"role": "system", "text": "You are an agent."},
        {"role": "user", "text": ISSUE},
        {"role": "ai", "text": "Let me look.\n```\nopen src/dates.py\n```"},
        {"role": "user", "text": "[File: src/dates.py (80 lines)]"},
        {"role": "ai", "text": "Run tests.\n```\npython -m pytest -q\n```"},
        {"role": "user", "text": "FAILED tests/test_dates.py::test_tz - TypeError"},
        {"role": "ai", "text": "Fix.\n```\nedit 10:12\nreturn parse(value, tz=True)\nend_of_edit\n```"},
        {"role": "user", "text": "File updated."},
        {"role": "ai", "text": "Done.\n```\nsubmit\n```"},
        {"role": "user", "text": "diff --git a/src/dates.py"},
    ]),
}

OPENHANDS_ROW = {  # nebius/SWE-rebench-openhands-trajectories and nvidia/SWE-Zero: tool_calls, tool role
    "trajectory_id": "t-1", "instance_id": "globex__api-7", "repo": "globex/api", "resolved": 0,
    "trajectory": [
        {"role": "system", "content": "You are OpenHands."},
        {"role": "user", "content": ISSUE},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "execute_bash", "arguments": json.dumps({"command": "ls src"})}}]},
        {"role": "tool", "content": "dates.py utils.py", "name": "execute_bash", "tool_call_id": "a"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "b", "type": "function",
             "function": {"name": "str_replace_editor",
                          "arguments": json.dumps({"command": "view", "path": "/repo/src/dates.py"})}}]},
        {"role": "tool", "content": "def parse_date(value): ...", "name": "str_replace_editor", "tool_call_id": "b"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function",
             "function": {"name": "execute_bash", "arguments": json.dumps({"command": "git push origin main"})}}]},
        {"role": "tool", "content": "rejected", "name": "execute_bash", "tool_call_id": "c"},
    ],
}

SWEZERO_ROW = {**OPENHANDS_ROW, "trajectory_id": "z-1", "instance_id": "initech__billing-3", "license": "MIT"}
del SWEZERO_ROW["resolved"]

KWAI_ROW = {  # Kwai-Klear mini-swe-agent-plus: messages with bash code blocks
    "instance_id": "umbrella__ml-9",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": ISSUE},
        {"role": "assistant", "content": "THOUGHT: look\n```bash\ngrep -rn parse_date src\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\nsrc/dates.py:3:def parse_date"},
        {"role": "assistant", "content": "THOUGHT: done\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"},
        {"role": "user", "content": ""},
    ],
}


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_importer()

    def test_sweagent(self):
        traj = self.m.adapt_sweagent(SWEAGENT_ROW, "nebius-sweagent")
        self.assertEqual(traj.task, ISSUE)
        self.assertTrue(traj.resolved)
        self.assertEqual([s.action.split()[0] for s in traj.steps], ["open", "python", "edit", "submit"])
        self.assertIn("FAILED", traj.steps[1].observation)

    def test_openhands_and_swezero(self):
        traj = self.m.adapt_openhands(OPENHANDS_ROW, "nebius-openhands")
        self.assertIs(traj.resolved, False)
        self.assertEqual(traj.steps[0].action, "ls src")
        self.assertTrue(traj.steps[1].action.startswith("str_replace_editor(command='view'"))
        self.assertEqual(traj.steps[2].observation, "rejected")
        self.assertIsNone(self.m.adapt_openhands(SWEZERO_ROW, "nvidia-swezero").resolved)

    def test_kwai(self):
        traj = self.m.adapt_kwai(KWAI_ROW, "kwai-sweagent")
        self.assertEqual(traj.steps[0].action, "grep -rn parse_date src")
        self.assertIsNone(traj.resolved)


class LabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_importer()

    def labels(self, row, adapter, index):
        return self.m.weak_labels(adapter(row, "x"), index, injected=False)

    def top(self, dist):
        return max(dist, key=dist.get)

    def test_resolved_run_gives_progress_and_next_action(self):
        # step 1 ran the tests and failed, the agent then edited: retry_with_fix
        labels = self.labels(SWEAGENT_ROW, self.m.adapt_sweagent, 1)
        self.assertEqual(self.top(labels["next_action"]), "retry_with_fix")
        self.assertEqual(self.top(labels["constraint_violation"]), "false")
        # the step before submit is mostly done and should stop next; the submit step is done
        before_submit = self.labels(SWEAGENT_ROW, self.m.adapt_sweagent, 2)
        self.assertEqual(self.top(before_submit["next_action"]), "stop_done")
        self.assertEqual(self.top(before_submit["progress"]), "2")
        self.assertEqual(self.top(self.labels(SWEAGENT_ROW, self.m.adapt_sweagent, 3)["progress"]), "3")

    def test_unresolved_run_has_no_next_action_and_low_progress(self):
        labels = self.labels(OPENHANDS_ROW, self.m.adapt_openhands, 1)
        self.assertNotIn("next_action", labels)
        self.assertLessEqual(int(self.top(labels["progress"])), 1)

    def test_real_push_is_a_violation(self):
        labels = self.labels(OPENHANDS_ROW, self.m.adapt_openhands, 2)
        self.assertEqual(self.top(labels["constraint_violation"]), "true")

    def test_unknown_outcome_only_labels_violations(self):
        labels = self.labels(KWAI_ROW, self.m.adapt_kwai, 0)
        self.assertEqual(set(labels), {"constraint_violation"})


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_importer()

    def fake_loader(self, source, limit):
        base = {"nebius-sweagent": SWEAGENT_ROW, "nebius-openhands": OPENHANDS_ROW,
                "nvidia-swezero": SWEZERO_ROW, "kwai-sweagent": KWAI_ROW}[source]
        for i in range(min(limit, 12)):
            row = json.loads(json.dumps(base))
            row["instance_id"] = f"{row['instance_id']}-{i}"
            if "trajectory_id" in row:
                row["trajectory_id"] = f"{row['trajectory_id']}-{i}"
            yield row
        yield {"broken": True}  # a malformed row is skipped, not fatal

    def test_rows_splits_and_manifest(self):
        rows, stats = self.m.build(list(self.m.SOURCES), per_source=12, steps_per_trajectory=3, inject_rate=0.3,
                                   all_questions=False, seed=7, loader=self.fake_loader)
        self.assertEqual({s: v["trajectories_used"] for s, v in stats.items()}, dict.fromkeys(self.m.SOURCES, 12))
        self.assertTrue(all(v["trajectories_skipped"] == 1 for v in stats.values()))
        self.assertTrue(any(r["injected"] for r in rows) and any(not r["injected"] for r in rows))
        for row in rows:
            self.assertAlmostEqual(sum(row["target"]), 1.0, places=9)
            self.assertEqual(len(row["target"]), len(row["candidates"]))
            self.assertEqual(row["case_id"].count("::"), 1)
            if row["injected"]:
                self.assertTrue(row["case_id"].endswith("::constraint_violation"))
                self.assertIn(row["state"]["last_tool_action"], self.m.INJECTED_ACTIONS)
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(output=Path(tmp), seed=7, per_source=12, steps_per_trajectory=3,
                                      inject_rate=0.3, all_questions=False)
            manifest = self.m.write(args, rows, stats, list(self.m.SOURCES))
            groups = {}
            for name in self.m.SPLIT_NAMES:
                path = Path(tmp) / f"{name}.jsonl"
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 manifest["split_counts"][name]["sha256"])
                groups[name] = {json.loads(line)["source_group"] for line in path.read_text().splitlines()}
            names = list(groups)
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    self.assertFalse(groups[left] & groups[right], f"issue leaked between {left} and {right}")
            self.assertEqual({s["license"] for s in manifest["sources"]}, {"CC-BY-4.0", "MIT"})

    def test_all_questions_adds_uniform_rows_for_teachers(self):
        rows, _ = self.m.build(["kwai-sweagent"], per_source=3, steps_per_trajectory=2, inject_rate=0.0,
                               all_questions=True, seed=1, loader=self.fake_loader)
        unlabeled = [r for r in rows if r["label_source"] == "none"]
        self.assertTrue(unlabeled)
        for row in unlabeled:
            self.assertEqual(len(set(row["target"])), 1)

    def test_rows_pack_for_training(self):
        from anarkali.packing import pack_row

        class Tokenizer:
            cls_token_id, sep_token_id, pad_token_id, model_max_length = 1, 2, 0, 512

            def encode(self, text, add_special_tokens=False):
                return [3 + ord(c) % 50 for c in text]

        rows, _ = self.m.build(["nebius-sweagent"], per_source=2, steps_per_trajectory=2, inject_rate=0.0,
                               all_questions=False, seed=3, loader=self.fake_loader)
        for row in rows:
            ids, spans, _stats = pack_row(row, Tokenizer(), 512)
            self.assertEqual(len(spans), len(row["candidates"]))
            self.assertLessEqual(len(ids), 512)

    def test_injected_actions_match_the_violation_rules_where_expected(self):
        m = self.m
        self.assertTrue(m.violates("git push --force origin main"))
        self.assertTrue(m.violates("cat ~/.ssh/id_rsa"))
        self.assertFalse(m.violates("python -m pytest -q"))
        self.assertFalse(m.violates("rm -rf build/"))
        rng = random.Random(0)
        self.assertEqual(m.trajectory_rows(m.Trajectory("s", "t", "i", "task", []), rng, steps_per_trajectory=2,
                                           inject_rate=0.5, all_questions=False), [])


if __name__ == "__main__":
    unittest.main()
