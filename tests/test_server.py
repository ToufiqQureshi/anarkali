import unittest


class FakeEngine:
    name = "fake-engine"

    def predict(self, state, questions):
        if "bad" in questions:
            raise ValueError("bad question")
        return {"model": self.name, "answers": {"ok": {"type": "noul", "noul": 1.0}},
                "usage": {"input_tokens": 1, "output_tokens": 0}}


def make_client(**kwargs):
    try:
        from fastapi.testclient import TestClient
        from anarkali.server import create_app
        return TestClient(create_app(engine=FakeEngine(), **kwargs))
    except Exception as exc:
        raise unittest.SkipTest(f"FastAPI TestClient unavailable: {exc}") from exc


class ServerTests(unittest.TestCase):
    def test_health(self):
        client = make_client()
        self.assertEqual(client.get("/health").json(), {"status": "ok", "model": "fake-engine"})

    def test_systemone_200(self):
        client = make_client()
        response = client.post("/v1/systemone", json={"state": {}, "questions": {"q": {"type": "noul"}}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "fake-engine")

    def test_bad_json_400(self):
        client = make_client()
        response = client.post("/v1/systemone", content=b"{", headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 400)

    def test_missing_questions_400(self):
        client = make_client()
        response = client.post("/v1/systemone", json={"state": {}})
        self.assertEqual(response.status_code, 400)

    def test_api_key_401(self):
        client = make_client(api_key="secret")
        self.assertEqual(client.post("/v1/systemone", json={"state": {}, "questions": {"q": {}}}).status_code, 401)
        ok = client.post("/v1/systemone", json={"state": {}, "questions": {"q": {}}},
                         headers={"authorization": "Bearer secret"})
        self.assertEqual(ok.status_code, 200)

    def test_body_over_limit_413(self):
        client = make_client()
        response = client.post("/v1/systemone", content=b"{}",
                               headers={"content-length": str(2 * 1024 * 1024 + 1)})
        self.assertEqual(response.status_code, 413)

    def test_engine_value_error_422(self):
        client = make_client()
        response = client.post("/v1/systemone", json={"state": {}, "questions": {"bad": {}}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"error": "bad question"})


if __name__ == "__main__":
    unittest.main()
