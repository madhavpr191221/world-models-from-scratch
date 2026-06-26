from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.parse
from dataclasses import asdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from jepa_world_models.analysis.video_latent_projection import build_latent_projection_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the latent projection browser demo.")
    parser.add_argument("--world-model-checkpoint", type=str, required=True, help="Path to the trained latent world-model checkpoint.")
    parser.add_argument("--data-root", type=str, default="data", help="Data root with Something-Something V2 files.")
    parser.add_argument("--source-split", type=str, default="train", help="Dataset split used for the latent bank.")
    parser.add_argument("--subset-size", type=int, default=256, help="Number of videos to sample into the bank.")
    parser.add_argument("--image-size", type=int, default=224, help="Frame resize target used for latent extraction.")
    parser.add_argument("--context-seconds", type=float, default=5.0, help="Observed context duration in seconds.")
    parser.add_argument("--future-seconds", type=float, default=1.5, help="Prediction horizon in seconds.")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Sampling rate in frames per second.")
    parser.add_argument("--feature-batch-size", type=int, default=1, help="Batch size used while building the latent cache.")
    parser.add_argument("--index", type=int, default=0, help="Default dataset index to open.")
    parser.add_argument("--projection-method", type=str, default="pca", choices=["pca", "tsne"], help="Default 2D projection method.")
    parser.add_argument("--background-sample-size", type=int, default=512, help="Background cloud size for the rendered manifold.")
    parser.add_argument("--examples-limit", type=int, default=12, help="Number of example clips to expose in the dropdown.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override, for example cuda or cpu.")
    parser.add_argument("--cache-dir", type=str, default="logs/video_world_model/cache", help="Directory used for cached latent sequences.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind.")
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


def _seconds_to_frame_count(seconds: float, sample_fps: float) -> int:
    frames = max(2, int(round(seconds * sample_fps)))
    return frames + (frames % 2)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    frontend_root = repo_root / "frontend"
    data_root = repo_root / args.data_root
    video_root = data_root / "something_v2" / "20bn-something-something-v2"

    context_frames = _seconds_to_frame_count(args.context_seconds, args.sample_fps)
    future_frames = _seconds_to_frame_count(args.future_seconds, args.sample_fps)
    total_frames = context_frames + future_frames

    engine = build_latent_projection_engine(
        world_model_checkpoint=args.world_model_checkpoint,
        data_root=data_root,
        source_split=args.source_split,
        subset_size=args.subset_size,
        image_size=args.image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        feature_batch_size=args.feature_batch_size,
        cache_dir=args.cache_dir,
        seed=args.seed,
        device=args.device,
    )
    default_payload = engine.analyze_index(
        args.index,
        projection_method=args.projection_method,
        background_sample_size=args.background_sample_size,
        seed=args.seed,
    ).to_json()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(frontend_root), **handler_kwargs)

        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            route = parsed.path

            if route in {"/", "/index.html"}:
                return _serve_file(self, frontend_root / "latent.html")

            if route == "/latent.html":
                return _serve_file(self, frontend_root / "latent.html")

            if route == "/latent.css":
                return _serve_file(self, frontend_root / "latent.css")

            if route == "/latent.js":
                return _serve_file(self, frontend_root / "latent.js")

            if route == "/video.css":
                return _serve_file(self, frontend_root / "video.css")

            if route in {"/video.html", "/method.html", "/results.html", "/styles.css", "/app.js", "/demo.html", "/reconstruction.html", "/reconstruction.css", "/reconstruction.js"}:
                return _serve_file(self, frontend_root / route.lstrip("/"))

            if route == "/api/latent/examples":
                examples = [asdict(example) for example in engine.list_examples(limit=args.examples_limit)]
                return _json_response(self, {"examples": examples})

            if route == "/api/latent/analyze":
                params = urllib.parse.parse_qs(parsed.query)
                index = int(params.get("index", [args.index])[0])
                method = params.get("method", [args.projection_method])[0]
                background_sample_size = int(params.get("background_sample_size", [args.background_sample_size])[0])
                seed = int(params.get("seed", [args.seed])[0])
                payload = engine.analyze_index(
                    index,
                    projection_method=method,
                    background_sample_size=background_sample_size,
                    seed=seed,
                ).to_json()
                return _json_response(self, payload)

            if route == "/api/latent/default":
                return _json_response(self, default_payload)

            if route.startswith("/api/latent/file/"):
                filename = route.removeprefix("/api/latent/file/")
                filename = filename.split("?", 1)[0]
                safe_name = Path(filename).name
                return _serve_file(self, video_root / safe_name)

            return super().do_GET()

        def log_message(self, format: str, *args):  # noqa: A003
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving latent projection demo on http://{args.host}:{args.port}")
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


