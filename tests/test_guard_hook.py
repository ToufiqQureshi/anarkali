from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import threading
from pathlib import Path
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("anarkali_guard", REPO / "examples" / "claude-code" / "anarkali_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def event(tool, tool_input, cwd):
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input, "cwd": cwd}


class GuardTests(unittest.TestCase):
    def test_decisions_follow_thresholds(self):
        self.assertIsNone(guard.decide(0.2, 0.8, 0.5, "ls"))
        ask = guard.decide(0.6, 0.8, 0.5, "git push")["hookSpecificOutput"]
        self.assertEqual((ask["hookEventName"], ask["permissionDecision"]), ("PreToolUse", "ask"))
        deny = guard.decide(0.95, 0.8, 0.5, "git push --force")["hookSpecificOutput"]
        self.assertEqual(deny["permissionDecision"], "deny")
        self.assertIn("95%", deny["permissionDecisionReason"])

    def test_state_matches_the_agent_step_shape_and_keeps_history(self):
        seen = []

        def fake_predict(state):
            seen.append(state)
            return 0.9 if "push" in state["last_tool_action"] else 0.1

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / ".anarkali"
            first = guard.run(event("Bash", {"command": "pytest -q"}, tmp), fake_predict, log_dir=log,
                              deny_at=0.8, ask_at=0.5)
            second = guard.run(event("Bash", {"command": "git push --force origin main"}, tmp), fake_predict,
                               log_dir=log, deny_at=0.8, ask_at=0.5)
            self.assertIsNone(first)
            self.assertEqual(second["hookSpecificOutput"]["permissionDecision"], "deny")
            self.assertEqual(seen[1]["recent_actions"], ["pytest -q"])
            self.assertEqual(seen[1]["step"], 2)
            for key in ("task", "constraints", "last_tool_action", "result"):
                self.assertIn(key, seen[1])
            self.assertEqual(len((log / "guard-log.jsonl").read_text().splitlines()), 2)

    def test_edits_are_described_with_path_and_body(self):
        text = guard.describe("Edit", {"file_path": "tests/test_x.py", "new_string": "assert True"})
        self.assertEqual(text, "edit(tests/test_x.py) assert True")

    def test_http_path_speaks_the_systemone_shape(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                body = json.dumps({"answers": {"violation": {"type": "noul", "noul": 0.87}}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            state = guard.build_state(event("Bash", {"command": "rm -rf ../"}, "/repo"), [], ["No deletes."], None)
            self.assertAlmostEqual(guard.predict_http(url, state), 0.87)
            path, payload = received[0]
            self.assertEqual(path, "/v1/systemone")
            self.assertEqual(payload["questions"]["violation"]["type"], "noul")
            self.assertEqual(payload["state"]["last_tool_action"], "rm -rf ../")
        finally:
            server.shutdown()
            server.server_close()

    def test_settings_snippet_is_valid(self):
        settings = json.loads((REPO / "examples" / "claude-code" / "settings.json").read_text())
        hook = settings["hooks"]["PreToolUse"][0]
        self.assertIn("Bash", hook["matcher"])
        self.assertIn("anarkali_guard.py", hook["hooks"][0]["command"])


if __name__ == "__main__":
    unittest.main()
