from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from test_relabel import FakeTeachers, decision

REPO = Path(__file__).resolve().parents[1]


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_split_dir(root: Path):
    root.mkdir(parents=True)
    splits = {
        "train": [decision("a", ["wrong", "right", "meh"], [0.2, 0.7, 0.1]),
                  decision("b", ["right", "wrong"], [0.9, 0.1]),
                  decision("c", ["meh", "wrong", "right"], [0.1, 0.1, 0.8], state="contested")],
        "development": [decision("d", ["right", "wrong"], [0.5, 0.5])],
        "calibration": [decision("e", ["wrong", "right"], [0.5, 0.5])],
        "test": [decision("f", ["wrong", "right"], [0.3, 0.7])],
    }
    for name, rows in splits.items():
        (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"dataset": "toy", "revision": "r1"}), encoding="utf-8")


class Server:
    @classmethod
    def setUpClass(cls):
        cls.relabel, cls.worker, cls.orchestrator = load("relabel_with_teachers"), load("gpu_worker"), \
            load("gpu_orchestrator")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTeachers)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeTeachers.requests = []
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_split_dir(self.root / "input")

    def tearDown(self):
        self.tmp.cleanup()

    def submit(self, teachers):
        config = self.root / "job.json"
        config.write_text(json.dumps({"splits": "train", "relabel_args": ["--samples", "3"], "teachers": teachers}),
                          encoding="utf-8")
        self.orchestrator.main(["--hub", str(self.root / "hub"), "--job", "toy", "submit",
                                "--config", str(config), "--input", str(self.root / "input")])


class CacheTests(Server, unittest.TestCase):
    def test_torn_lines_are_skipped_and_every_shard_is_read(self):
        cache_dir = self.root / "cache"
        cache_dir.mkdir()
        (cache_dir / "kaggle-1.jsonl").write_text('{"key":"k1","probs":[1.0],"source":"logprobs"}\n{"key":"k2","pro',
                                                  encoding="utf-8")
        (cache_dir / "colab-1.jsonl").write_text('{"key":"k3","probs":[1.0],"source":"sample"}\n', encoding="utf-8")
        cache = self.relabel.Cache(cache_dir, "kaggle-1")
        self.assertEqual(set(cache.entries), {"k1", "k3"})
        cache.put({"key": "k4", "probs": [1.0], "source": "sample"})
        self.assertEqual(set(self.relabel.Cache(cache_dir, "other").entries), {"k1", "k3", "k4"})

    def test_bad_shard_names_are_refused(self):
        with self.assertRaises(ValueError):
            self.relabel.Cache(self.root / "cache", "../escape")

    def test_score_then_offline_matches_a_direct_run(self):
        common = ["--input", str(self.root / "input"), "--samples", "3"]
        teachers = ["--teacher", f"good=good@{self.url}", "--teacher", f"plain=plain@{self.url}"]
        self.relabel.main([*common, *teachers, "--output", str(self.root / "direct")])
        cache = str(self.root / "shared")
        for spec in teachers[1::2]:
            self.relabel.main([*common, "--teacher", spec, "--only-score", "--output", str(self.root / "s"),
                               "--cache-dir", cache])
        calls = len(FakeTeachers.requests)
        self.relabel.main([*common, "--offline", "--cache-dir", cache, "--output", str(self.root / "offline"),
                           "--teacher", "good=good@http://offline.invalid/v1",
                           "--teacher", "plain=plain@http://offline.invalid/v1"])
        self.assertEqual(len(FakeTeachers.requests), calls)
        for name in ("train.jsonl", "dropped-train.jsonl", "test.jsonl"):
            self.assertEqual((self.root / "offline" / name).read_bytes(), (self.root / "direct" / name).read_bytes())

    def test_offline_cache_miss_fails(self):
        with self.assertRaises(self.relabel.CacheMiss):
            self.relabel.main(["--input", str(self.root / "input"), "--output", str(self.root / "o"), "--offline",
                               "--teacher", "good=good@http://offline.invalid/v1", "--workers", "1"])


class WorkerTests(Server, unittest.TestCase):
    def test_worker_finishes_a_job_and_the_next_worker_sees_it_done(self):
        # per-teacher extra_body must survive into the offline build, or its cache keys miss
        self.submit([{"name": "good", "model": "good", "base_url": self.url,
                      "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
                     {"name": "plain", "model": "plain", "base_url": self.url}])
        hub = str(self.root / "hub")
        state = self.worker.main(["--hub", hub, "--job", "toy", "--provider", "kaggle", "--worker", "kaggle-1",
                                  "--workdir", str(self.root / "w1")])
        self.assertEqual(state, "finished")
        job = self.root / "hub" / "jobs" / "toy"
        self.assertEqual(sorted(p.stem for p in (job / "done").glob("*.json")), ["good", "plain"])
        self.assertTrue((job / "cache" / "kaggle-1.jsonl").stat().st_size > 0)
        beat = json.loads((job / "heartbeat" / "kaggle-1.json").read_text(encoding="utf-8"))
        self.assertTrue(beat["final"])
        self.assertEqual(beat["state"], "finished")
        output = json.loads((job / "output" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(output["relabel_stats"]["train"]["relabelled"], 3)
        self.assertEqual((job / "output" / "test.jsonl").read_bytes(), (self.root / "input" / "test.jsonl").read_bytes())
        self.assertFalse((job / "output" / "teacher-cache").exists())
        self.assertTrue(all(b["chat_template_kwargs"] == {"enable_thinking": False}
                            for b in FakeTeachers.requests if b["model"] == "good"))

        calls = len(FakeTeachers.requests)
        state = self.worker.main(["--hub", hub, "--job", "toy", "--provider", "colab", "--workdir", str(self.root / "w2")])
        self.assertEqual(state, "finished")
        self.assertEqual(len(FakeTeachers.requests), calls)

    def test_a_new_worker_reuses_scores_from_an_earlier_shard(self):
        self.submit([{"name": "good", "model": "good", "base_url": self.url}])
        hub = self.root / "hub"
        self.worker.main(["--hub", str(hub), "--job", "toy", "--worker", "colab-1", "--workdir", str(self.root / "w1")])
        # pretend the first worker died before marking done or building the output
        for path in [*(hub / "jobs/toy/done").glob("*"), *(hub / "jobs/toy/output").glob("*")]:
            path.unlink()
        calls = len(FakeTeachers.requests)
        self.worker.main(["--hub", str(hub), "--job", "toy", "--worker", "kaggle-2", "--workdir", str(self.root / "w2")])
        self.assertEqual(len(FakeTeachers.requests), calls)
        self.assertTrue((hub / "jobs/toy/output/manifest.json").exists())

    def test_status_reports_progress(self):
        self.submit([{"name": "good", "model": "good", "base_url": self.url}])
        status = self.orchestrator.main(["--hub", str(self.root / "hub"), "--job", "toy", "status"])
        self.assertEqual((status["teachers"], status["done"], status["finished"]), (["good"], [], False))


class DecideTests(unittest.TestCase):
    NOW = 1_000_000.0

    @classmethod
    def setUpClass(cls):
        cls.decide = staticmethod(load("gpu_orchestrator").decide)

    def status(self, heartbeats=(), launches=None, finished=False):
        return {"finished": finished, "heartbeats": list(heartbeats), "launches": launches or {}}

    def run_decide(self, status):
        return self.decide(status, ["kaggle", "colab"], self.NOW, backoff_hours=24)

    def test_rules(self):
        minute, hour = 60, 3600
        beat = lambda provider, ago, final=False: {"worker": f"{provider}-x", "provider": provider,
                                                   "time": self.NOW - ago, "final": final}
        launch = lambda ago: {"time": self.NOW - ago}
        cases = [
            ("finished job", self.status(finished=True), ("done", None)),
            ("fresh heartbeat", self.status([beat("kaggle", 5 * minute)]), ("running", "kaggle-x")),
            ("nothing launched yet", self.status(), ("launch", "kaggle")),
            ("worker went silent", self.status([beat("kaggle", 40 * minute)], {"kaggle": launch(2 * hour)}),
             ("launch", "kaggle")),
            ("worker ended its session", self.status([beat("kaggle", 1 * minute, final=True)],
                                                     {"kaggle": launch(9 * hour)}), ("launch", "kaggle")),
            ("kaggle still booting", self.status(launches={"kaggle": launch(10 * minute)}), ("waiting", "kaggle")),
            ("kaggle never answered: try colab", self.status(launches={"kaggle": launch(2 * hour)}),
             ("launch", "colab")),
            ("both quiet: stalled", self.status(launches={"kaggle": launch(2 * hour), "colab": launch(3 * hour)}),
             ("stalled", None)),
            ("kaggle launch failed: skip without waiting",
             self.status(launches={"kaggle": {"time": self.NOW - minute, "failed": True}}), ("launch", "colab")),
            ("kaggle backoff over", self.status(launches={"kaggle": launch(25 * hour), "colab": launch(3 * hour)}),
             ("launch", "kaggle")),
        ]
        for name, status, expected in cases:
            with self.subTest(name):
                self.assertEqual(self.run_decide(status), expected)


class TickTests(Server, unittest.TestCase):
    def test_failed_launch_falls_through_to_the_next_provider(self):
        self.submit([{"name": "good", "model": "good", "base_url": self.url}])
        orchestrator = self.orchestrator
        calls = []

        def broken(args):
            calls.append("kaggle")
            raise RuntimeError("Kaggle GPU quota has 0.2 h left this week")

        def colab(args):
            calls.append("colab")
            return "opened issue"

        saved = dict(orchestrator.LAUNCHERS)
        orchestrator.LAUNCHERS.update(kaggle=broken, colab=colab)
        try:
            hub = str(self.root / "hub")
            self.assertEqual(orchestrator.main(["--hub", hub, "--job", "toy", "tick"]), ("launch", "colab"))
            self.assertEqual(calls, ["kaggle", "colab"])
            launches = orchestrator.job_status(orchestrator.open_hub(hub), "toy")["launches"]
            self.assertTrue(launches["kaggle"]["failed"])
            self.assertEqual(orchestrator.main(["--hub", hub, "--job", "toy", "tick"]), ("waiting", "colab"))
            self.assertEqual(calls, ["kaggle", "colab"])
        finally:
            orchestrator.LAUNCHERS.clear()
            orchestrator.LAUNCHERS.update(saved)


class LauncherTests(unittest.TestCase):
    def test_kaggle_kernel_is_valid(self):
        orchestrator = load("gpu_orchestrator")
        with tempfile.TemporaryDirectory() as tmp:
            kernel = orchestrator.kaggle_kernel_dir(Path(tmp), username="me", hub="hf:me/work", job="toy",
                                                    repo_url="https://github.com/o/r", ref="main", hours=8)
            meta = json.loads((kernel / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["id"], meta["enable_gpu"], meta["enable_internet"], meta["is_private"]),
                             ("me/anarkali-gpu-worker", True, True, True))
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            source = (kernel / "run.py").read_text(encoding="utf-8")
            compile(source, "run.py", "exec")
            self.assertIn("'--provider', 'kaggle'", source.replace('"', "'"))

    def test_colab_link_and_tokenless_notice(self):
        orchestrator = load("gpu_orchestrator")
        link = orchestrator.colab_link("https://github.com/o/r.git", "main")
        self.assertEqual(link, "https://colab.research.google.com/github/o/r/blob/main/notebooks/gpu_worker_colab.ipynb")
        self.assertTrue((REPO / "notebooks" / "gpu_worker_colab.ipynb").exists())


if __name__ == "__main__":
    unittest.main()
