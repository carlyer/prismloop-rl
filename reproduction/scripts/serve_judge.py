#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess

from run_training import assert_idle, ROOT, PYTHON
from runtime_env import configure_runtime

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpus")
    p.add_argument("--port", type=int)
    p.add_argument("--model")
    args = p.parse_args()
    selection_path = ROOT / "judge-selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    args.gpus = args.gpus or selection.get("gpus", "5,6")
    args.port = args.port or selection.get("port", 18961)
    model = Path(args.model or selection.get("model_path", ROOT / "models/Qwen2.5-72B-Instruct"))
    index = json.loads((model / "model.safetensors.index.json").read_text())
    files = {model / name for name in index["weight_map"].values()}
    if any(not f.is_file() or Path(str(f) + ".aria2").exists() for f in files):
        raise RuntimeError("Judge checkpoint is not completely downloaded")
    assert_idle(args.gpus.split(","))
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", args.port))
    env = configure_runtime(os.environ.copy())
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, VLLM_USE_V1="0", OMP_NUM_THREADS="8",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cmd = [str(PYTHON), "-m", "vllm.entrypoints.openai.api_server",
           "--model", str(model), "--host", "127.0.0.1", "--port", str(args.port),
           "--tensor-parallel-size", str(len(args.gpus.split(","))),
           "--served-model-name", "judge", "--dtype", "bfloat16",
           "--gpu-memory-utilization", "0.8", "--max-model-len", "32768",
           "--disable-log-requests", "--max-num-seqs", "32", "--enforce-eager"]
    if selection.get("text_only_judging"):
        cmd += ["--limit-mm-per-prompt", '{"image":0,"video":0}']
    raise SystemExit(subprocess.call(cmd, env=env))
