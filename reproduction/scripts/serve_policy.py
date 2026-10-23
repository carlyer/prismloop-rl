"""Serve an isolated local policy for the released evaluation engine."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from run_training import assert_idle, PYTHON
from runtime_env import configure_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--port', type=int, default=18962)
    parser.add_argument('--config-only', action='store_true')
    args = parser.parse_args()
    model = Path(args.model).resolve()
    command = [str(PYTHON), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(model),
               '--host', '127.0.0.1', '--port', str(args.port), '--served-model-name', 'policy',
               '--tensor-parallel-size', str(len(args.gpus.split(','))), '--dtype', 'bfloat16',
               '--max-model-len', '32768', '--max-num-seqs', '4', '--gpu-memory-utilization', '0.8',
               '--limit-mm-per-prompt', '{"image":50,"video":0}',
               '--mm-processor-kwargs', '{"min_pixels":3136,"max_pixels":2000000}',
               '--enforce-eager', '--disable-log-requests']
    if args.config_only:
        print(json.dumps(command, indent=2))
        return 0
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    for filename in set(index['weight_map'].values()):
        path = model / filename
        if not path.is_file() or Path(str(path) + '.aria2').exists():
            raise RuntimeError('Incomplete policy checkpoint: ' + filename)
    assert_idle(args.gpus.split(','))
    env = configure_runtime(os.environ.copy())
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, VLLM_USE_V1='0', OMP_NUM_THREADS='8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return subprocess.call(command, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
