"""Forward existing processor limits into vLLM; leave RL objectives unchanged."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT.parent / 'verl_agents/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'


def main():
    source = PATH.read_text()
    old = '            max_num_batched_tokens=max_num_batched_tokens,\n'
    new = (old + '            # Local reproduction: align vision profiling with the data processor.\n'
           '            max_num_seqs=config.get("max_num_seqs", 256),\n'
           '            mm_processor_kwargs=dict(config.get("mm_processor_kwargs", {})),\n')
    manifest = ROOT / 'manifests/rollout-configuration-patch.json'
    if new in source:
        print('Rollout configuration forwarding already applied')
        return
    if source.count(old) != 1:
        raise RuntimeError('Unexpected rollout source; review patch before applying')
    backup = ROOT / 'references/vllm_rollout_spmd.original.py'
    if not backup.exists():
        backup.write_text(source)
    patched = source.replace(old, new)
    PATH.write_text(patched)
    manifest.write_text(json.dumps({'file': str(PATH), 'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'patched_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'reason': 'Forward image processor size limits and scheduler concurrency to vLLM; avoid oversize visual dummy profiling',
        'rl_reward_or_filtering_modified': False}, indent=2) + '\n')
    print('Recorded rollout processor-configuration compatibility patch')


if __name__ == '__main__':
    main()
