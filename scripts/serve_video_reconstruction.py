from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import os
import shutil
import uuid
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

from jepa_world_models.analysis.video_reconstruction import (
    build_reconstruction_head,
    build_tubelet_bank,
    load_clip_from_video_path,
    reconstruct_clip_with_decoder,
    reconstruct_clip_with_bank,
    save_reconstruction_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the VideoMAE reconstruction demo.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--source-split", type=str, default="train")
    parser.add_argument("--subset-size", type=int, default=128)
    parser.add_argument("--bank-batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--reconstruction-mode", type=str, default="decoder", choices=["decoder", "retrieval"])
    parser.add_argument("--cache-dir", type=str, default="logs/video_reconstruction/cache")
    parser.add_argument("--output-dir", type=str, default="logs/video_reconstruction")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser


class ReconstructionServer(BaseHTTPRequestHandler):
    server_version = "VideoReconstructionHTTP/1.0"

    def _send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start_str, end_str = range_header.removeprefix("bytes=").split("-", 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else len(data) - 1
            end = min(end, len(data) - 1)
            if start > end or start >= len(data):
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid range")
                return
            chunk = data[start : end + 1]
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", mime)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        if path == "/" or path == "/index.html":
            self._serve_file(self.server.frontend_dir / "reconstruction.html")
            return
        if path in {"/reconstruction.js", "/reconstruction.css"}:
            self._serve_file(self.server.frontend_dir / path.lstrip("/"))
            return
        if path.startswith("/artifacts/"):
            rel = path.removeprefix("/artifacts/")
            self._serve_file(self.server.output_dir / rel)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown path")

    def do_POST(self) -> None:
        if self.path != "/api/reconstruct":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown path")
            return
        ctype, pdict = cgi.parse_header(self.headers.get("content-type", ""))
        if ctype != "multipart/form-data":
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected multipart/form-data")
            return
        pdict["boundary"] = pdict["boundary"].encode("utf-8")
        content_length = int(self.headers.get("content-length", "0"))
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers["content-type"],
                "CONTENT_LENGTH": str(content_length),
            },
            keep_blank_values=True,
        )
        if "file" not in form:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing file upload")
            return
        mask_ratio = float(form.getfirst("mask_ratio", "0.5"))
        mask_mode = form.getfirst("mask_mode", "middle")
        upload = form["file"]
        filename = Path(upload.filename or "upload.webm").name
        run_id = uuid.uuid4().hex[:12]
        upload_dir = self.server.output_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved_upload = upload_dir / f"{run_id}_{filename}"
        with saved_upload.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

        clip = load_clip_from_video_path(saved_upload, num_frames=self.server.num_frames, image_size=self.server.image_size)
        if self.server.reconstruction_mode == "retrieval":
            result = reconstruct_clip_with_bank(
                checkpoint_path=self.server.checkpoint_path,
                bank=self.server.bank,
                clip=clip,
                mask_ratio=mask_ratio,
                mask_mode=mask_mode,
            )
        else:
            result = reconstruct_clip_with_decoder(
                checkpoint_path=self.server.checkpoint_path,
                head=self.server.head,
                clip=clip,
                mask_ratio=mask_ratio,
                mask_mode=mask_mode,
        )
        run_dir = self.server.output_dir / "runs" / run_id
        payload = save_reconstruction_artifacts(result, run_dir)
        artifact_keys = {
            "original_video",
            "masked_video",
            "reconstructed_video",
            "original_gif",
            "masked_gif",
            "reconstructed_gif",
            "original_sheet",
            "masked_sheet",
            "reconstructed_sheet",
            "mask_sheet",
        }
        response = {
            "run_id": run_id,
            "source_file": str(saved_upload.relative_to(self.server.output_dir)),
            "mask_ratio": mask_ratio,
            "mask_mode": mask_mode,
            "reconstruction_mode": self.server.reconstruction_mode,
            "artifacts": {
                k: f"/artifacts/{Path(v).relative_to(self.server.output_dir).as_posix()}"
                for k, v in payload.items()
                if k in artifact_keys
            },
            "metrics": {
                k: v
                for k, v in payload.items()
                if k not in artifact_keys
            },
        }
        body = json.dumps(response, indent=2).encode("utf-8")
        self._send_bytes(body, "application/json")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frontend_dir = Path("frontend")
    bank = None
    head = None
    if args.reconstruction_mode == "retrieval":
        print("Building or loading reconstruction tubelet bank...")
        bank = build_tubelet_bank(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            source_split=args.source_split,
            subset_size=args.subset_size,
            image_size=args.image_size,
            num_frames=args.num_frames,
            batch_size=args.bank_batch_size,
            cache_dir=args.cache_dir,
        )
    else:
        print("Building or loading reconstruction decoder head...")
        head, head_bundle = build_reconstruction_head(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            source_split=args.source_split,
            subset_size=args.subset_size,
            image_size=args.image_size,
            num_frames=args.num_frames,
            batch_size=args.bank_batch_size,
            cache_dir=args.cache_dir,
        )
    server = ThreadingHTTPServer((args.host, args.port), ReconstructionServer)
    server.frontend_dir = frontend_dir
    server.output_dir = output_dir
    server.bank = bank
    server.head = head
    server.checkpoint_path = args.checkpoint
    server.num_frames = args.num_frames
    server.image_size = args.image_size
    server.reconstruction_mode = args.reconstruction_mode
    print(f"Serving reconstruction demo at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
