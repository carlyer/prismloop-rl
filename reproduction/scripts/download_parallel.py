#!/usr/bin/env python3
"""Resume pinned Hugging Face assets with segmented HTTPS and LFS checksums."""
import concurrent.futures
import argparse
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import quote

from huggingface_hub import HfApi, hf_hub_download
from download_assets import ASSETS, ROOT

ARIA = Path("/opt/pyvision-tools/usr/bin/aria2c")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["image", "all"], default="image")
    args = parser.parse_args()
    names = (["image-rl-data", "image-sft-model", "judge"] if args.scope == "image" else
             ["image-rl-data", "video-rl-data", "image-sft-model", "judge",
              "image-sft-data", "video-sft-model", "video-sft-data"])
    policy_path = ROOT / "download-policy.json"
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    if policy.get("allow_judge_download") is False:
        names = [name for name in names if name != "judge"]
        print("Remote judge download disabled by user; downloading training assets only", flush=True)
    records = []
    for name in names:
        manifest = ROOT / "manifests" / f"{name}.json"
        if manifest.exists():
            record = json.loads(manifest.read_text())
        else:
            kind, repo, category = ASSETS[name]
            info = HfApi().repo_info(repo, repo_type=kind, files_metadata=True)
            record = {"repo_id": repo, "repo_type": kind, "revision": info.sha,
                      "local_dir": str(ROOT / category / repo.split("/")[-1]),
                      "files": [{"path": f.rfilename, "size": f.size,
                                 "sha256": getattr(f.lfs, "sha256", None) if f.lfs else None}
                                for f in info.siblings]}
        record.update(status="downloading", transport="aria2-segmented-https")
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        records.append((name, manifest, record))
    small, large = [], []
    for name, _, record in records:
        for item in record["files"]:
            (large if (item["size"] or 0) >= 64 * 1024**2 else small).append((record, item))

    def download_small(pair):
        record, item = pair
        path = hf_hub_download(record["repo_id"], item["path"],
                               repo_type=record["repo_type"], revision=record["revision"],
                               local_dir=record["local_dir"])
        if item.get("sha256"):
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual != item["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {item['path']}")

    # Small config/tokenizer/JSON files remain managed by huggingface_hub.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    metadata_future = pool.submit(lambda: list(pool.map(download_small, small)))
    input_lines = []
    for record, item in large:
        target = Path(record["local_dir"]) / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        # Recover the sequential prefix from our earlier HF transfer.
        if not target.exists() and item.get("sha256"):
            cache = Path(record["local_dir"]) / ".cache/huggingface/download"
            partials = [p for p in cache.glob("*.incomplete") if item["sha256"] in p.name]
            if partials:
                partial = max(partials, key=lambda p: p.stat().st_size)
                partial.rename(target)
        prefix = "datasets/" if record["repo_type"] == "dataset" else ""
        url = (f"https://huggingface.co/{prefix}{record['repo_id']}/resolve/"
               f"{record['revision']}/{quote(item['path'])}")
        input_lines += [url, f"  dir={target.parent}", f"  out={target.name}"]
        if item.get("sha256"):
            input_lines.append(f"  checksum=sha-256={item['sha256']}")
    uri_file = ROOT / "manifests/download-uris.txt"
    uri_file.write_text("\n".join(input_lines) + "\n")
    env = os.environ.copy()
    lib = "/opt/pyvision-tools/usr/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"]:
        if name in env:
            env[name.lower()] = env[name]
    command = [str(ARIA), "--input-file=" + str(uri_file), "--continue=true",
               "--max-concurrent-downloads=16", "--split=4", "--max-connection-per-server=4",
               "--min-split-size=16M", "--file-allocation=none", "--auto-file-renaming=false",
               "--allow-overwrite=true", "--check-integrity=true", "--max-tries=8",
               "--retry-wait=5", "--timeout=60", "--connect-timeout=15",
               "--summary-interval=30", "--enable-color=false", "--show-console-readout=false",
               "--console-log-level=warn", "--log-level=warn",
               "--log=" + str(ROOT / "logs/aria2.log")]
    print(f"Downloading {len(large)} large files with SHA-256 checks, pinned revisions", flush=True)
    rc = subprocess.call(command, env=env)
    metadata_future.result()
    pool.shutdown()
    if rc:
        raise RuntimeError(f"aria2 exited {rc}; rerun this script to resume")
    for name, manifest, record in records:
        for item in record["files"]:
            path = Path(record["local_dir"]) / item["path"]
            if not path.is_file() or (item["size"] is not None and path.stat().st_size != item["size"]):
                raise RuntimeError(f"Missing or incomplete file: {path}")
        record.update(status="complete", lfs_checksums_verified=True,
                      completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        if name == "video-sft-data":
            record["data_available"] = False
            record["note"] = "Upstream repository contains metadata only; no training samples or media."
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        print(f"COMPLETE {name}", flush=True)


if __name__ == "__main__":
    with (ROOT / "download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
