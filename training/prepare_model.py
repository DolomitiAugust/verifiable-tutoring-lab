"""Download the pinned official Qwen Base weights and verify every byte."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import requests


ROOT = Path(__file__).resolve().parent
CHUNK = 12 * 1024 * 1024


def digest(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def fetch_span(url, part, start, end, size):
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    for attempt in range(4):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(12, 40)) as res:
                res.raise_for_status()
                cr = res.headers.get("Content-Range")
                full = start == 0 and end + 1 == size
                if cr != f"bytes {start}-{end}/{size}" and not (full and not cr and res.status_code == 200):
                    raise OSError(f"unexpected response range {cr} for {start}-{end}")
                remaining = end - start + 1
                with part.open("r+b") as dst:
                    dst.seek(start)
                    for data in res.iter_content(256 * 1024):
                        if len(data) > remaining:
                            raise OSError("response longer than requested segment")
                        dst.write(data)
                        remaining -= len(data)
                if remaining:
                    raise OSError(f"short response: missing {remaining} bytes")
            return
        except (requests.RequestException, OSError) as err:
            if attempt == 3:
                raise
            print(f"retry {part.name} at {start}: {err}", flush=True)
            time.sleep(1 + attempt)


def obtain(name, size, expected, repo, revision, dest, workers):
    path = dest / name
    if path.is_file() and path.stat().st_size == size and digest(path) == expected:
        print(f"verified existing {name}", flush=True)
        return
    url = f"https://modelscope.cn/api/v1/models/{repo}/repo?Revision={revision}&FilePath={name}"
    part = path.with_name(path.name + ".part")
    with part.open("wb") as dst:
        dst.truncate(size)
    spans = [(i, min(size, i + CHUNK) - 1) for i in range(0, size, CHUNK)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_span, url, part, i, j, size) for i, j in spans]
        for count, pending in enumerate(as_completed(futures), 1):
            pending.result()
            if len(spans) > 1 and (count % 8 == 0 or count == len(spans)):
                print(f"{name}: {count}/{len(spans)} segments", flush=True)
    if part.stat().st_size != size or digest(part) != expected:
        raise OSError(f"size or SHA-256 mismatch: {name}")
    os.replace(part, path)
    print(f"verified {name} ({size:,} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=ROOT / "model" / "base")
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=6)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "base_source.json").read_text(encoding="utf-8"))
    for name, (size, sha) in manifest["files"].items():
        obtain(name, size, sha, manifest["model_id"], manifest["revision"], args.dest, args.workers)


if __name__ == "__main__":
    main()
