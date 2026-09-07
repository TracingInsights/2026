#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=1.10"]
# ///
"""Sync this telemetry tree to a Hugging Face bucket.

Auth (never commit a token; this repo is public):
  export HF_TOKEN=hf_...     # write token from https://huggingface.co/settings/tokens
  # or:  hf auth login

  uv run sync_hf_bucket.py
  uv run sync_hf_bucket.py --dry-run
  uv run sync_hf_bucket.py --delete   # also remove remote files not present locally
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BUCKET = "tracinginsights/2026"
SKIP_TOP = {
    ".git",
    ".venv",
    ".cursor",
    ".freebuff",
    "cache",
    "cache_preseason",
    "cache_joblib",
    "fastf1_cache",
    "_bench_out",
    ".bench_metrics_tmp",
}
SKIP_PARTS = {"__pycache__", ".venv"}
SKIP_SUFFIXES = (".sqlite", ".ff1pkl", ".pyc")
BATCH = 64
WORKERS = 10


def local_files(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_TOP and d not in SKIP_PARTS]
        rel_dir = os.path.relpath(dirpath, root)
        top = "." if rel_dir == "." else rel_dir.split(os.sep)[0]
        if top in SKIP_TOP:
            continue
        for name in filenames:
            if name.endswith(SKIP_SUFFIXES) or name == ".env":
                continue
            full = Path(dirpath) / name
            dest = name if rel_dir == "." else f"{rel_dir}/{name}".replace("\\", "/")
            out[dest] = full
    return out


def upload_batch(payload: tuple[str, list[tuple[str, str]]]) -> tuple[int, str | None]:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    bucket, batch = payload
    from huggingface_hub import batch_bucket_files

    try:
        batch_bucket_files(bucket, add=batch)
        return len(batch), None
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if "hf_" in msg:
            msg = f"{type(exc).__name__}: <redacted>"
        return 0, msg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help="namespace/bucket id")
    p.add_argument("--root", type=Path, default=ROOT, help="local directory to upload")
    p.add_argument("--dry-run", action="store_true", help="print counts only; do not upload")
    p.add_argument(
        "--delete",
        action="store_true",
        help="delete remote files that are not present locally (after excludes)",
    )
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--batch-size", type=int, default=BATCH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ.setdefault("HF_XET_FIXED_UPLOAD_CONCURRENCY", "8")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token is not None and not token.strip():
        token = None

    from huggingface_hub import HfApi, batch_bucket_files

    api = HfApi(token=token)
    root = args.root.resolve()
    bucket = args.bucket
    print(f"root={root}", flush=True)
    print(f"bucket=hf://buckets/{bucket}", flush=True)
    print("auth=HF_TOKEN env" if token else "auth=huggingface_hub cached login", flush=True)

    print("listing local files...", flush=True)
    local = local_files(root)
    print(f"local={len(local)}", flush=True)

    print("listing remote...", flush=True)
    remote_size: dict[str, int] = {}
    for item in api.list_bucket_tree(bucket, recursive=True):
        if getattr(item, "type", None) == "directory":
            continue
        remote_size[item.path] = int(getattr(item, "size", 0) or 0)
    print(f"remote={len(remote_size)}", flush=True)

    to_upload: list[tuple[str, str]] = []
    for dest, path in local.items():
        size = path.stat().st_size
        if dest not in remote_size or remote_size[dest] != size:
            to_upload.append((str(path), dest))

    to_delete = sorted(set(remote_size) - set(local)) if args.delete else []
    print(f"upload={len(to_upload)} delete={len(to_delete)}", flush=True)
    if args.dry_run:
        print("dry-run: no changes", flush=True)
        return 0
    if not to_upload and not to_delete:
        print("already in sync", flush=True)
        return 0

    if to_upload:
        batches = [
            (bucket, to_upload[i : i + args.batch_size])
            for i in range(0, len(to_upload), args.batch_size)
        ]
        print(
            f"batches={len(batches)} workers={args.workers} batch={args.batch_size}",
            flush=True,
        )
        done = 0
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(upload_batch, b) for b in batches]
            for fut in as_completed(futs):
                n, err = fut.result()
                done += n
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                if err:
                    print(f"ERROR {err}", flush=True)
                else:
                    print(
                        f"uploaded {done}/{len(to_upload)} "
                        f"({100 * done / len(to_upload):.1f}%) {rate:.0f} files/s",
                        flush=True,
                    )

    if to_delete:
        print(f"deleting {len(to_delete)} remote-only files...", flush=True)
        for i in range(0, len(to_delete), 200):
            batch_bucket_files(bucket, delete=to_delete[i : i + 200], token=token)
            print(f"deleted {min(i + 200, len(to_delete))}/{len(to_delete)}", flush=True)

    print("DONE", flush=True)
    info = api.bucket_info(bucket)
    print(f"remote_files={info.total_files} remote_bytes={info.size}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
