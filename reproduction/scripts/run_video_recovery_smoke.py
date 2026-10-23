"""Verify two Video RL updates on idle GPUs while the Image baseline continues."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from archive_training_logs import archive, process_identity
from monitor_recovery import append_log, atomic
from run_training import ROOT, REPO, PYTHON, assert_idle

PREFIX = 'video-recovery-20261019T1622Z'
RUN = ROOT / 'outputs' / (PREFIX + '-smoke')
STATUS = ROOT / 'manifests/video-recovery-smoke-status.json'
# Ray adds a timestamped session and socket filename under this prefix.
# Keep the final AF_UNIX socket path below Linux's 108-byte limit.
RAY = Path('/tmp/pv-video-ray-1019') / 'ray/session_latest'


def main():
    original = (ROOT / 'origin-reproduction').resolve(strict=True)
    recovery = original / 'recovery/20261019T142145Z'
    log_path = ROOT / 'logs' / (PREFIX + '-smoke.log')
    record = dict(status='running', stage='checking_idle_video_smoke_gpus', run=str(RUN),
                  gpus='0,2', steps=2, engineering_only=True,
                  formal_700_step_video_training_completed=False)

    def progress(stage, **fields):
        record.update(stage=stage, observed_utc=datetime.now(timezone.utc).isoformat(), **fields)
        atomic(STATUS, record)
        try:
            atomic(original / 'manifests' / STATUS.name, record)
            if log_path.exists():
                append_log(log_path, recovery / 'ongoing-local-runtime-logs' / log_path.name)
        except OSError as exc:
            print('Persistent mirror needs retry:', str(exc), flush=True)
        print(stage, flush=True)

    try:
        progress(record['stage'])
        assert_idle(['0', '2'])
        state = json.loads((ROOT / 'status.json').read_text())
        if state.get('video_full_run') not in [None, PREFIX + '-full']:
            raise RuntimeError('A different Video run is already selected; inspect it before launching')
        data = ROOT / 'data/prepared/video_smoke.json'
        rows = json.loads(data.read_text())
        if not rows or not all(Path(row['mm_hint']['hint_path']).is_file() for row in rows):
            raise RuntimeError('Video smoke inputs are missing media')
        if RUN.exists() and (RUN / 'invocation.json').exists():
            raise RuntimeError('This engineering run already exists; preserve and inspect it before retrying')
        record['training_input_sha256'] = hashlib.sha256(data.read_bytes()).hexdigest()
        command = [str(PYTHON), '-u', str(ROOT / 'scripts/run_training.py'),
                   '--kind', 'video', '--mode', 'smoke', '--gpus', '0,2',
                   '--name', PREFIX + '-smoke', '--ray-temp-dir', str(RAY.parent.parent),
                   '--checkpoint-dir', '/tmp/pyvision-rl-storage/checkpoints/' + PREFIX + '-smoke']
        environment = dict(os.environ, NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
        with log_path.open('a') as log:
            child = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        metadata = dict(pid=child.pid, start_time_ticks=process_identity(child.pid), command=command)
        atomic(ROOT / 'manifests/video-recovery-smoke-training-process.json', metadata)
        atomic(original / 'manifests/video-recovery-smoke-training-process.json', metadata)
        progress('running_two_step_video_training', child=metadata)
        while child.poll() is None:
            if RAY.exists() and (RUN / 'invocation.json').exists():
                archive(RAY, RUN)
            progress('running_two_step_video_training')
            time.sleep(60)
        if child.returncode:
            raise RuntimeError('Video engineering run failed; inspect its preserved log and exit receipt')
        proof = json.loads((RUN / 'optimizer-update-validation.json').read_text())
        receipt = json.loads((RUN / 'exit-status.json').read_text())
        if (receipt.get('returncode') != 0 or proof.get('nonzero_weight_update_verified') is not True
                or proof.get('checkpoint_step') != 2):
            raise RuntimeError('Video engineering run lacks a successful two-step weight-update proof')
        state = json.loads((ROOT / 'status.json').read_text())
        if state.get('video_full_run') not in [None, PREFIX + '-full']:
            raise RuntimeError('Video training selection changed while the engineering check ran')
        state.update(video_full_run=PREFIX + '-full', video_recovery_smoke_run=PREFIX + '-smoke',
                     video_two_step_weight_update_verified=True)
        atomic(ROOT / 'status.json', state)
        atomic(recovery / 'video-two-step-optimizer-update-validation.json', proof)
        destination = original / 'outputs' / RUN.name
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(RUN, target_is_directory=True)
        progress('passed_two_step_video_weight_update', status='complete', proof=proof)
    except BaseException as exc:
        progress('needs_attention', status='failed', error=str(exc))
        raise


if __name__ == '__main__':
    with (ROOT / 'video-recovery-smoke.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(ROOT / 'manifests/video-recovery-smoke-process.json', dict(
            pid=os.getpid(), start_time_ticks=process_identity(os.getpid()),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        main()
