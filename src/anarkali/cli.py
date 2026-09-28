import argparse
import json
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="anarkali")
    sub = parser.add_subparsers(dest="command", required=True)
    decide = sub.add_parser("decide", help="run a Jev-shaped request against a released model")
    decide.add_argument("--model", required=True)
    decide.add_argument("--request", required=True)
    decide.add_argument("--orders", type=int, help="average over this many option orders (slower, less position bias)")
    serve = sub.add_parser("serve", help="serve the Jev-compatible HTTP API")
    serve.add_argument("--model", required=True)
    serve.add_argument("--port", type=int)
    serve.add_argument("--threads", type=int)
    serve.add_argument("--orders", type=int, help="average over this many option orders; env ANARKALI_ORDERS")
    smoke = sub.add_parser("smoke", help="test neural mechanics without downloading a model")
    smoke.add_argument("--out")
    validate = sub.add_parser("validate-request", help="validate a choice request JSON file")
    validate.add_argument("request")
    compare = sub.add_parser("compare", help="compare paired JSONL prediction ledgers")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        if args.command == "decide":
            from .engine import Engine
            payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            result = Engine.load(args.model, orders=args.orders).predict(payload.get("state"), payload.get("questions"))
        elif args.command == "serve":
            from .server import serve as run_server
            run_server(model_path=args.model, port=args.port, threads=args.threads, orders=args.orders)
            return 0
        elif args.command == "smoke":
            from .smoke import run_smoke
            result = run_smoke()
        elif args.command == "validate-request":
            from .schema import ChoiceRequest
            request = ChoiceRequest.from_dict(json.loads(Path(args.request).read_text(encoding="utf-8")))
            result = {"valid": True, "request": request.to_dict(), "prediction_produced": False}
        else:
            from .evaluation import compare_predictions, load_predictions
            result = compare_predictions(load_predictions(args.baseline), load_predictions(args.candidate))
        rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if getattr(args, "out", None):
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"anarkali: {exc}", file=sys.stderr)
        return 2
