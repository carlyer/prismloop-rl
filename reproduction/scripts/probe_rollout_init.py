"""Isolate vLLM vision profiling from Ray/FSDP without updating any weights."""
import argparse
import json
import os
from pathlib import Path

from run_training import ROOT, assert_idle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['smoke', 'full'], default='smoke')
    args = parser.parse_args()
    assert_idle(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    from vllm import LLM
    smoke = args.profile == 'smoke'
    config = dict(model=str(ROOT / 'models/PyVision-Image-7B-SFT'), load_format='dummy',
                  dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
                  max_model_len=6144 if smoke else 52480, max_num_batched_tokens=6144 if smoke else 52480,
                  max_num_seqs=8, gpu_memory_utilization=0.5,
                  limit_mm_per_prompt={'image': 32, 'video': 0},
                  mm_processor_kwargs={'min_pixels': 3136, 'max_pixels': 262144 if smoke else 2000000})
    print(json.dumps(config, indent=2), flush=True)
    model = LLM(**config)
    import torch
    torch.cuda.synchronize()
    result = {'status': 'passed', 'profile': args.profile, 'config': config,
              'training_completed': False, 'uses_dummy_weights': True,
              'allocated_GiB': torch.cuda.memory_allocated() / 1024**3}
    (ROOT / f'manifests/rollout-init-{args.profile}-probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
