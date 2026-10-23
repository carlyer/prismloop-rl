#!/usr/bin/env python3
"""Apply a narrow, recorded fix for the upstream recommended dependency pair."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
versions = {p: importlib.metadata.version(p) for p in ["vllm", "transformers"]}
if versions != {"vllm": "0.9.1", "transformers": "4.54.0"}:
    raise RuntimeError(f"Review compatibility patch for different versions: {versions}")
path = Path(sysconfig.get_paths()["purelib"]) / "vllm/transformers_utils/configs/ovis.py"
text = path.read_text()
old = 'AutoConfig.register("aimv2", AIMv2Config)'
# Ovis is unused by Qwen2.5-VL. Keep the built-in Transformers registration.
new = ('# PyVision reproduction: transformers 4.54 already registers aimv2.\n'
       'if "aimv2" not in __import__("transformers").models.auto.configuration_auto.CONFIG_MAPPING:\n'
       '    AutoConfig.register("aimv2", AIMv2Config)')
if new in text:
    print("vLLM AIMv2 registration compatibility fix already applied")
elif text.count(old) == 1:
    before = hashlib.sha256(text.encode()).hexdigest()
    patched = text.replace(old, new)
    path.with_suffix(".py.pyvision-original").write_text(text)
    path.write_text(patched)
    record = {"versions": versions, "file": str(path), "original_sha256": before,
              "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
              "change": "Skip duplicate AIMv2 registration; Qwen model code and RL algorithm unchanged."}
    (ROOT / "manifests/dependency-patches.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
else:
    raise RuntimeError("Unexpected vLLM Ovis source; refusing an unreviewed replacement")
