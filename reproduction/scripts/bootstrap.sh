#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname -- "$REPRO_ROOT")"
REPRO_PYTHON=/opt/pyvision-rl-venv/bin/python
cd "$REPO_ROOT"
uv venv --python 3.10 --allow-existing /opt/pyvision-rl-venv
# The host defaults to a CUDA 13 package index, which has no torch 2.7 wheels.
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u UV_TORCH_BACKEND \
  uv --no-config pip install --python "$REPRO_PYTHON" --index-url https://pypi.org/simple \
  -r "$REPRO_ROOT/requirements-runtime.txt" -r "$REPO_ROOT/pv_requirements.txt"
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u UV_TORCH_BACKEND \
  uv --no-config pip install --python "$REPRO_PYTHON" --no-deps \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u UV_TORCH_BACKEND \
  uv --no-config pip install --python "$REPRO_PYTHON" --no-deps --editable "$REPO_ROOT/verl_agents"
uv pip freeze --python "$REPRO_PYTHON" > "$REPRO_ROOT/manifests/environment-freeze.txt"
"$REPRO_PYTHON" "$REPRO_ROOT/scripts/patch_dependencies.py"
"$REPRO_PYTHON" "$REPRO_ROOT/scripts/patch_rollout_configuration.py"
"$REPRO_PYTHON" "$REPRO_ROOT/scripts/patch_tensor_padding.py"
"$REPRO_PYTHON" "$REPRO_ROOT/scripts/patch_training_runtime.py"
"$REPRO_PYTHON" "$REPRO_ROOT/scripts/patch_tool_cleanup.py"
"$REPRO_PYTHON" "$REPRO_ROOT/scripts/patch_numeric_reward_guard.py"
"$REPRO_PYTHON" "$REPRO_ROOT/scripts/patch_optimizer_configuration.py"
"$REPRO_PYTHON" "$REPRO_ROOT/scripts/patch_checkpoint_merger.py"
python "$REPRO_ROOT/scripts/install_python_headers.py"
export CPATH="/opt/pyvision-tools/usr/include/python3.10:/opt/pyvision-tools/usr/include${CPATH:+:$CPATH}"
PYTHONPATH="$REPO_ROOT/verl_agents" \
  LLM_AS_A_JUDGE_CONFIG_PATH="$REPRO_ROOT/judge-config.json" \
  CUDA_VISIBLE_DEVICES="${PREFLIGHT_GPU:-7}" VLLM_USE_V1=0 \
  "$REPRO_PYTHON" "$REPRO_ROOT/scripts/preflight.py"
"$REPRO_PYTHON" "$REPRO_ROOT/scripts/run_training.py" --config-only \
  > "$REPRO_ROOT/manifests/smoke-resolved-config.yaml"
printf '%s\n' '{"status":"complete","runtime_preflight_passed":true}' \
  > "$REPRO_ROOT/manifests/environment-ready.json"
