"""Archive the small source/configuration artifacts needed to replay a launch."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
EVAL = REPO.parent / 'PyVision-RL-Eval'
TRAINING_FILES = [
    'verl_agents/verl/trainer/main_ppo.py',
    'verl_agents/verl/utils/reward_score/research_answer.py',
    'verl_agents/verl/trainer/ppo/research_selection.py',
    'verl_agents/verl/workers/agent/envs/agents_x/research_tool_quality.py',
    'verl_agents/verl/workers/agent/envs/agents_x/research_worker_cleanup.py',
    'verl_agents/verl/workers/reward_manager/naive.py',
    'verl_agents/scripts/model_merger.py',
    'verl_agents/verl/trainer/ppo/ray_trainer.py',
    'verl_agents/verl/utils/torch_functional.py',
    'verl_agents/verl/workers/agent/parallel_env.py',
    'verl_agents/verl/workers/agent/envs/agents_x/safe_persis_python_exe_tool_w_image_hint.py',
    'verl_agents/verl/workers/agent/envs/agents_x/safe_persis_python_exe_tool_wo_image_hint.py',
    'verl_agents/verl/workers/agent/envs/agents_x/safe_persis_python_exe_tool_wo_video_hint.py',
    'verl_agents/verl/utils/reward_score/rule_match_verify.py',
    'verl_agents/verl/workers/fsdp_workers.py',
    'verl_agents/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
]
EVALUATION_FILES = [
    'inference_engine/vis_inference_demo_gpt.py',
    'evaluation/eval_gpt_pl.py',
    'cal_score/answer_verify.py',
]


def git(repo, *arguments):
    return subprocess.check_output(['git', *arguments], cwd=repo)


def snapshot(run, phase):
    if phase not in ['prelaunch', 'observed_during_run']:
        raise ValueError('Unknown source observation phase')
    run = Path(run).resolve(strict=True)
    observed = datetime.now(timezone.utc)
    folder = run / 'source-snapshots' / (observed.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    training_revision = git(REPO, 'rev-parse', 'HEAD').decode().strip()
    evaluation_revision = git(EVAL, 'rev-parse', 'HEAD').decode().strip()
    sources = sorted(f for f in (ROOT / 'scripts').iterdir() if f.suffix in ['.py', '.sh'])
    sources += [REPO / name for name in TRAINING_FILES]
    sources += [EVAL / name for name in EVALUATION_FILES]
    sources += sorted((ROOT / 'manifests').glob('*patch*.json'))
    sources += [path for path in [
        ROOT / 'references/evaluation-cache-compatibility.json',
        ROOT / 'references/evaluate_official.before-context-budget-fix.py',
        ROOT / 'references/evaluate_official.context-budget-fixed.py',
        ROOT / 'manifests/context-budget-classification-validation.json',
        ROOT / 'manifests/sft-context-recovery-validation.json',
        ROOT / 'manifests/tool-cleanup-validation.json',
        ROOT / 'manifests/numeric-reward-guard-validation.json',
        ROOT / 'manifests/training-validation-archive-checks.json',
        ROOT / 'manifests/training-validation-archive-native-step20.json',
        ROOT / 'manifests/rl-training-validation-media-id-audit.json',
        ROOT / 'manifests/video-training-validation400.json',
        ROOT / 'data/prepared/video_val_dataset_vsibench400.json',
        ROOT / 'data/prepared/video_val_dataset_vsibench400.raw.json',
    ] if path.exists()]
    sources += sorted((ROOT / 'references/context-budget-original-case').glob('*.json'))
    contents, records = {}, []
    originals = {}
    for source in sources:
        data = source.read_bytes()
        originals[source] = data
        member = str(source.relative_to(REPO.parent))
        contents[member] = data
        records.append(dict(member=member, source=str(source), bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest()))
    contents['metadata/training-working-tree.patch'] = git(REPO, 'diff', '--binary', '--', *TRAINING_FILES)
    contents['metadata/evaluation-working-tree.patch'] = git(EVAL, 'diff', '--binary', '--', *EVALUATION_FILES)
    packages = sorted([(dist.metadata.get('Name', ''), dist.version) for dist in metadata.distributions()])
    contents['metadata/python-package-versions.json'] = json.dumps(dict(
        python=sys.version, executable=sys.executable, distributions=packages), indent=2).encode()
    for name, data in contents.items():
        if name.startswith('metadata/'):
            records.append(dict(member=name, source='generated_at_observation', bytes=len(data),
                                sha256=hashlib.sha256(data).hexdigest()))
    if any(source.read_bytes() != data for source, data in originals.items()):
        raise RuntimeError('Source changed during snapshot; refusing a mixed observation')
    if (git(REPO, 'rev-parse', 'HEAD').decode().strip() != training_revision or
            git(EVAL, 'rev-parse', 'HEAD').decode().strip() != evaluation_revision):
        raise RuntimeError('Repository revision changed during snapshot')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, data in sorted(contents.items()):
            entry = tarfile.TarInfo(name)
            entry.size, entry.mtime, entry.mode = len(data), int(observed.timestamp()), 0o644
            archive.addfile(entry, io.BytesIO(data))
    data = buffer.getvalue()
    folder.mkdir(parents=True, exist_ok=False)
    archive_path = folder / 'sources.tar.gz'
    archive_path.write_bytes(data)
    report = dict(status='sources_archived', observed_utc=observed.isoformat(), phase=phase,
                  captured_before_launch=phase == 'prelaunch',
                  training_revision=training_revision, evaluation_revision=evaluation_revision,
                  archive=str(archive_path), archive_sha256=hashlib.sha256(data).hexdigest(),
                  archive_bytes=len(data), files=records,
                  scope='Local reproduction scripts, the explicitly patched training files, selected '
                        'released evaluator sources, patch receipts, restricted Git diffs, and '
                        'installed package versions. Unmodified upstream source is identified by commit. '
                        'An observed_during_run snapshot does not prove the historical launch source version.')
    receipt = folder / 'manifest.json'
    receipt.write_text(json.dumps(report, indent=2) + '\n')
    return dict(receipt=str(receipt), receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                archive=str(archive_path), archive_sha256=report['archive_sha256'], phase=phase)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=['prelaunch', 'observed_during_run'], required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run, args.phase)), flush=True)


if __name__ == '__main__':
    main()
