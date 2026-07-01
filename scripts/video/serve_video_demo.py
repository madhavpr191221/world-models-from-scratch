from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import asdict
from pathlib import Path

from jepa_world_models.analysis.video_dynamics import build_video_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the video latent-dynamics demo.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained VicReg checkpoint.")
    parser.add_argument("--data-root", type=str, default="data", help="Data root with Something-Something V2 files.")
    parser.add_argument("--index", type=int, default=0, help="Default dataset index to open.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of neighbors to expose.")
    parser.add_argument("--bank-size", type=int, default=512, help="Number of clips to sample into the bank.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind.")
    return parser.parse_args()


def _json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _serve_file(handler: SimpleHTTPRequestHandler, path: Path) -> None:
    if not path.exists() or not path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND, "File not found")
        return

    content_type, _ = mimetypes.guess_type(path.name)
    data = path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type or "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    frontend_root = repo_root / "frontend"
    data_root = repo_root / args.data_root
    video_root = data_root / "something_v2" / "20bn-something-something-v2"
    engine = build_video_engine(
        args.checkpoint,
        data_root=data_root,
        bank_size=args.bank_size,
    )
    default_payload = engine.analyze_index(args.index, top_k=args.top_k).to_json()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(frontend_root), **handler_kwargs)

        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            route = parsed.path

            if route in {"/", "/index.html"}:
                self.path = "/video.html"
                return super().do_GET()

            if route == "/api/video/examples":
                examples = [asdict(example) for example in engine.list_examples(limit=12)]
                return _json_response(self, {"examples": examples})

            if route == "/api/video/analyze":
                params = urllib.parse.parse_qs(parsed.query)
                index = int(params.get("index", [args.index])[0])
                top_k = int(params.get("top_k", [args.top_k])[0])
                reverse = params.get("reverse", ["0"])[0] == "1"
                if reverse:
                    payload = engine.analyze_reverse_index(index=index, top_k=top_k).to_json()
                else:
                    payload = engine.analyze_index(index=index, top_k=top_k).to_json()
                return _json_response(self, payload)

            if route.startswith("/api/video/file/"):
                filename = route.removeprefix("/api/video/file/")
                filename = filename.split("?", 1)[0]
                safe_name = Path(filename).name
                return _serve_file(self, video_root / safe_name)

            if route == "/api/video/default":
                return _json_response(self, default_payload)

            return super().do_GET()

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving video demo on http://{args.host}:{args.port}")
    print(f"Frontend root: {frontend_root}")
    print(f"Video root: {video_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
