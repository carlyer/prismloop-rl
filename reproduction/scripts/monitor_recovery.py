"""Observe recovery and mirror small metadata/logs to the persistent workspace."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import time

from run_training import ROOT
from archive_training_logs import archive as archive_workers
from pipeline import update_state_fields

SEEN_ARTIFACTS = {}
CREATED_ARTIFACT_DIRECTORIES = set()
BOUND_TRAINING_SESSIONS = {}


def atomic(path, value):
    data = (json.dumps(value, indent=2) + '\n').encode()
    if path.exists() and path.read_bytes() == data:
        return
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_bytes(data)
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def live(record):
    if not record:
        return False
    try:
        fields = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == str(record['start_time_ticks'])
    except FileNotFoundError:
        return False


def append_log(source, target):
    """Append only new bytes; refuse to silently replace a different log."""
    offset = target.stat().st_size if target.exists() else 0
    if source.stat().st_size < offset:
        raise ValueError('Source log shrank: ' + str(source))
    with source.open('rb') as stream:
        if offset:
            start = max(0, offset - 4096)
            stream.seek(start)
            with target.open('rb') as previous:
                previous.seek(start)
                if stream.read(offset - start) != previous.read(offset - start):
                    raise ValueError('Persistent log tail differs: ' + str(source))
        stream.seek(offset)
        with target.open('ab') as destination:
            while block := stream.read(1024 * 1024):
                destination.write(block)


def mirror_small_artifacts(archive, time_budget=30):
    """Preserve score records, selected rollouts, source snapshots and worker logs."""
    deadline = time.monotonic() + time_budget
    copied, reused = 0, 0
    def result(pending):
        return dict(status='backfill_pending' if pending else 'caught_up_for_observed_files',
                    files_copied_this_pass=copied, files_reused_this_pass=reused,
                    cached_artifacts=len(SEEN_ARTIFACTS), time_budget_seconds=time_budget)
    destination = archive / 'ongoing-local-runtime-results'
    for folder in (ROOT / 'outputs').iterdir():
        if not folder.is_dir() or folder.is_symlink():
            continue
        for parent, directories, files in os.walk(folder, followlinks=False):
            directories[:] = [name for name in directories if name != 'checkpoints'
                              and not (Path(parent) / name).is_symlink()]
            for name in files:
                if name.endswith('messages.json'):
                    continue
                source = Path(parent) / name
                is_json = source.suffix == '.json'
                is_snapshot = name == 'sources.tar.gz'
                is_worker_log = 'runtime-logs' in source.parts and source.suffix in ['.out', '.err']
                if not (is_json or is_snapshot or is_worker_log):
                    continue
                if time.monotonic() >= deadline:
                    return result(True)
                stat = source.stat()
                identity = (stat.st_size, stat.st_mtime_ns)
                if SEEN_ARTIFACTS.get(str(source)) == identity:
                    continue
                target = destination / source.relative_to(ROOT / 'outputs')
                if target.parent not in CREATED_ARTIFACT_DIRECTORIES:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    CREATED_ARTIFACT_DIRECTORIES.add(target.parent)
                if is_worker_log:
                    append_log(source, target)
                else:
                    data = source.read_bytes()
                    if is_json:
                        try:
                            json.loads(data)
                        except json.JSONDecodeError:
                            continue  # A producer may still be writing this record.
                    after = source.stat()
                    if (after.st_size, after.st_mtime_ns) != identity:
                        continue
                    if target.exists() and target.stat().st_size == len(data) and target.read_bytes() == data:
                        SEEN_ARTIFACTS[str(source)] = identity
                        reused += 1
                        continue
                    temporary = target.with_name(target.name + '.tmp-' + str(os.getpid()))
                    temporary.write_bytes(data)
                    temporary.replace(target)
                SEEN_ARTIFACTS[str(source)] = identity
                copied += 1
    return result(False)


def observe():
    coordinator = read(ROOT / 'manifests/recovery-coordinator-process.json')
    prefix = coordinator['prefix']
    training = read(ROOT / 'manifests/recovery-image-training-process.json')
    for kind in ['image', 'video']:
        audit = read(ROOT / 'manifests' / f'recovered-{kind}-weight-update-validation.json')
        flag = kind + '_nonzero_weight_update_verified'
        if (audit.get('status') == 'passed_nonzero_weight_update'
                and audit.get('nonzero_weight_update_verified')
                and not read(ROOT / 'status.json').get(flag)):
            update_state_fields(**{flag: True, kind + '_optimizer_audit': str(
                ROOT / 'manifests' / f'recovered-{kind}-weight-update-validation.json')})
    for owner, run_name in [(training, prefix + '-full')]:
        if live(owner):
            run = ROOT / 'outputs' / run_name
            invocation = read(run / 'invocation.json')
            ray_root = invocation.get('args', {}).get('ray_temp_dir')
            session = Path(ray_root) / 'ray/session_latest' if ray_root else None
            if session and session.exists() and list((session / 'logs').glob('worker-*.out')):
                key = (owner['pid'], owner['start_time_ticks'])
                bound = BOUND_TRAINING_SESSIONS.setdefault(key, session.resolve())
                archive_workers(bound, run)
    log = ROOT / 'logs' / (prefix + '-full.log')
    text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text(errors='replace')) if log.exists() else ''
    steps = []
    for match in re.finditer(r'\bstep:(\d+) - ([^\n]+)', text):
        fields = dict((key, float(value)) for key, value in re.findall(
            r'(?:^| - )([^: ]+):(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', match[2]))
        steps.append(dict(step=int(match[1]), metrics=fields))
    evaluations = []
    for name, expected in [('official-image-sft-control-vstar', 6112),
                           ('official-image-sft-control-mathvision-mini', 304),
                           ('official-video-sft-control-vsibench', 5130),
                           ('official-image-rl-control-vstar', 6112),
                           ('official-image-rl-control-mathvision-mini', 304)]:
        output = ROOT / 'outputs' / name
        count = sum(bool(re.fullmatch(r'.+-repeat\d+\.json', item.name))
                    for item in output.iterdir()) if output.exists() else 0
        evaluations.append(dict(run=name, record_files=count, expected=expected,
                                summary=read(output / 'summary.json'),
                                partial_record_count_is_not_an_accepted_score=True))
    report = dict(observed_utc=datetime.now(timezone.utc).isoformat(), prefix=prefix,
                  execution_root=str(ROOT), status=read(ROOT / 'status.json'),
                  coordinator_alive=live(coordinator), training_alive=live(training),
                  baseline_continuation=read(ROOT / 'manifests/baseline-continuation-status.json'),
                  pipeline_owner_alive=live(read(ROOT / 'manifests/pipeline-process.json')),
                  parallel_video_training=read(ROOT / 'manifests/parallel-video-training-status.json'),
                  parallel_video_training_alive=live(read(ROOT / 'manifests/parallel-video-training-process.json')),
                  resume_checkpoint_step=1, completed_updates_after_restore=steps,
                  latest_completed_step=steps[-1]['step'] if steps else None,
                  console_precision_limited=True, source_log=str(log),
                  engineering_weight_verification=read(ROOT / 'outputs' / (prefix + '-smoke') /
                                                      'optimizer-update-validation.json'),
                  formal_weight_verification=read(ROOT / 'manifests/recovered-image-weight-update-validation.json'),
                  evaluations=evaluations, checkpoints_on_temporary_local_storage=True)
    atomic(ROOT / 'manifests/recovery-progress.json', report)
    original = (ROOT / 'origin-reproduction').resolve(strict=True)
    archive = Path(coordinator['archive'])
    try:
        atomic(original / 'manifests/recovery-progress.json', report)
        for name in ['recovery-coordinator-process.json', 'recovery-image-training-process.json',
                     'recovery-image-weight-audit-process.json', 'recovery-launch.json',
                     'image-sft-control-process.json', 'video-sft-control-process.json',
                     'image-sft-control-status.json', 'video-sft-control-status.json',
                     'official-image-sft-control-mathvision-mini-process.json',
                     'image-rl-early-control-process.json', 'image-rl-early-control-status.json',
                     'image-rl-control-process.json', 'image-rl-control-status.json',
                     'video-rl-control-process.json', 'video-rl-control-status.json',
                     'expanded-evaluation-process.json', 'expanded-evaluation-status.json',
                     'parallel-video-controller-process.json', 'parallel-video-training-process.json',
                     'parallel-video-training-status.json', 'parallel-video-weight-audit-process.json',
                     'recovered-video-weight-update-validation.json',
                     'recovery-monitor-process.json', 'baseline-continuation-process.json',
                     'baseline-continuation-status.json', 'recovery-judge-process.json']:
            source = ROOT / 'manifests' / name
            if not source.exists():
                continue
            target = original / 'manifests' / name
            backup = archive / ('before-local-runtime-' + name)
            if target.exists() and not backup.exists():
                atomic(backup, read(target))
            atomic(target, read(source))
        logs = archive / 'ongoing-local-runtime-logs'
        logs.mkdir(exist_ok=True)
        for source in (ROOT / 'logs').glob('*.log'):
            video_name = report['status'].get('video_full_run', '')
            if (source.name.startswith(prefix) or source.name.startswith('official-')
                    or (video_name and source.name.startswith(video_name))):
                append_log(source, logs / source.name)
        report['persistent_artifact_mirror'] = mirror_small_artifacts(archive)
        atomic(ROOT / 'manifests/recovery-progress.json', report)
        atomic(original / 'manifests/recovery-progress.json', report)
        atomic(archive / 'latest-recovery-progress.json', report)
    except (OSError, ValueError) as exc:
        print('Persistent mirror needs retry:', str(exc), flush=True)
    owners = [coordinator, training,
              read(ROOT / 'manifests/image-sft-control-process.json'),
              read(ROOT / 'manifests/video-sft-control-process.json'),
              read(ROOT / 'manifests/image-rl-early-control-process.json'),
              read(ROOT / 'manifests/image-rl-control-process.json'),
              read(ROOT / 'manifests/video-rl-control-process.json'),
              read(ROOT / 'manifests/expanded-evaluation-process.json'),
              read(ROOT / 'manifests/parallel-video-training-process.json'),
              read(ROOT / 'manifests/parallel-video-controller-process.json'),
              read(ROOT / 'manifests/baseline-continuation-process.json')]
    return report, any(live(owner) for owner in owners)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=120)
    args = parser.parse_args()
    with (ROOT / 'recovery-monitor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                report, running = observe()
            except (OSError, json.JSONDecodeError) as exc:
                if args.once:
                    raise
                print('Observation needs retry:', str(exc), flush=True)
                time.sleep(args.interval)
                continue
            print(json.dumps(dict(observed_utc=report['observed_utc'],
                                  latest_step=report['latest_completed_step'],
                                  training_alive=report['training_alive'],
                                  evaluation_records=[e['record_files'] for e in report['evaluations']])), flush=True)
            if args.once or not running:
                break
            time.sleep(args.interval)
