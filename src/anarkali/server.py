"""Jev-compatible HTTP server for released Anarkali engines.

FastAPI, uvicorn and Engine are imported inside functions so importing this
module stays lightweight.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hmac
import json
import os
from typing import Any

MAX_BODY_BYTES = 2 * 1024 * 1024


def _engine_from_env(model_path: str | None, threads: int | None):
    from .engine import Engine

    selected = model_path or os.getenv("ANARKALI_MODEL")
    if not selected:
        raise RuntimeError("ANARKALI_MODEL or --model is required")
    return Engine.load(selected, threads=threads)


def create_app(*, engine: Any | None = None, model_path: str | None = None,
               threads: int | None = None, api_key: str | None = None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    selected_threads = threads
    if selected_threads is None and os.getenv("ANARKALI_THREADS"):
        selected_threads = int(os.environ["ANARKALI_THREADS"])
    app = FastAPI(title="Anarkali", version="0.3.0")
    app.state.engine = engine or _engine_from_env(model_path, selected_threads)
    app.state.api_key = api_key if api_key is not None else os.getenv("ANARKALI_API_KEY")
    app.state.inference_lock = asyncio.Lock()
    app.state.executor = ThreadPoolExecutor(max_workers=1)

    def json_error(status_code: int, message: str) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status_code)

    async def read_json_body(request: Request) -> dict[str, Any] | JSONResponse:
        length = request.headers.get("content-length")
        if length is not None:
            try:
                if int(length) > MAX_BODY_BYTES:
                    return json_error(413, "request body exceeds 2 MB")
            except ValueError:
                return json_error(400, "bad JSON")
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return json_error(413, "request body exceeds 2 MB")
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return json_error(400, "bad JSON")
        if not isinstance(parsed, dict):
            return json_error(400, "request body must be an object")
        return parsed

    def authorized(request: Request) -> bool:
        required = app.state.api_key
        if not required:
            return True
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token, required)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": app.state.engine.name}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        if not authorized(request):
            return json_error(401, "unauthorized")
        payload = await read_json_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        questions = payload.get("questions")
        if questions is None:
            return json_error(400, "missing questions")
        state = payload.get("state")
        loop = asyncio.get_running_loop()
        async with app.state.inference_lock:
            try:
                return await loop.run_in_executor(
                    app.state.executor,
                    lambda: app.state.engine.predict(state, questions),
                )
            except ValueError as exc:
                return json_error(422, str(exc))
            except Exception:
                return json_error(500, "internal server error")

    return app


def serve(*, model_path: str | None = None, host: str | None = None,
          port: int | None = None, threads: int | None = None) -> None:
    import uvicorn

    app = create_app(model_path=model_path, threads=threads)
    uvicorn.run(app, host=host or os.getenv("ANARKALI_HOST", "0.0.0.0"),
                port=port or int(os.getenv("ANARKALI_PORT", "8000")))


__all__ = ["MAX_BODY_BYTES", "create_app", "serve"]
