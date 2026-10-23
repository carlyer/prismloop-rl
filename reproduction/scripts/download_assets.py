#!/usr/bin/env python3
"""Download official, revision-pinned assets; retain manifests and retry safely."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "image-rl-data": ("dataset", "Agents-X/PyVision-Image-RL-Data", "data"),
    "video-rl-data": ("dataset", "Agents-X/PyVision-Video-RL-Data", "data"),
    "image-sft-data": ("dataset", "Agents-X/PyVision-Image-SFT-Data", "data"),
    "video-sft-data": ("dataset", "Agents-X/PyVision-Video-SFT-Data", "data"),
    "image-sft-model": ("model", "Agents-X/PyVision-Image-7B-SFT", "models"),
    "video-sft-model": ("model", "Agents-X/PyVision-Video-7B-SFT", "models"),
    "judge": ("model", "Qwen/Qwen2.5-72B-Instruct", "models"),
}


def download(name):
    policy_path = ROOT / "download-policy.json"
    if name == "judge" and policy_path.exists():
        if json.loads(policy_path.read_text()).get("allow_judge_download") is False:
            print("SKIP judge: remote download disabled by user", flush=True)
            return name
    kind, repo, category = ASSETS[name]
    target = ROOT / category / repo.split("/")[-1]
    manifest_path = ROOT / "manifests" / f"{name}.json"
    api = HfApi()
    # Keep the same revision when restarting a partially completed download.
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    info = api.repo_info(repo, repo_type=kind, revision=previous.get("revision"), files_metadata=True)
    files = [{"path": f.rfilename, "size": f.size,
              "sha256": getattr(f.lfs, "sha256", None) if f.lfs else None}
             for f in info.siblings]
    record = {"repo_id": repo, "repo_type": kind, "revision": info.sha,
              "local_dir": str(target), "files": files, "status": "downloading"}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"START {name}: {sum(f['size'] or 0 for f in files)/1e9:.2f} GB, revision {info.sha}", flush=True)
    for attempt in range(1, 5):
        try:
            snapshot_download(repo, repo_type=kind, revision=info.sha,
                              local_dir=target, max_workers=4)
            missing = [f["path"] for f in files if not (target / f["path"]).is_file()
                       or (f["size"] is not None and (target / f["path"]).stat().st_size != f["size"])]
            if missing:
                raise RuntimeError(f"Missing or incomplete files: {missing}")
            record["status"] = "complete"
            record["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            manifest_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"COMPLETE {name}: {target}", flush=True)
            return name
        except Exception as exc:
            print(f"RETRY {name} attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
            if attempt == 4:
                record["status"] = "failed"
                record["error"] = str(exc)
                manifest_path.write_text(json.dumps(record, indent=2) + "\n")
                raise
            time.sleep(min(5 * attempt, 15))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", nargs="*", choices=list(ASSETS), default=list(ASSETS))
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name in pool.map(download, args.assets):
            print(f"VERIFIED {name}", flush=True)
