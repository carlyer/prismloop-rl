"""Run the authorized Video baseline on idle GPUs beside the Image baseline."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from archive_training_logs import archive, process_identity
from monitor_recovery import append_log, atomic
from pipeline import completed_training, update_state_fields
from run_training import ROOT, REPO, PYTHON, assert_idle

RAY_ROOT = Path('/tmp/pv-video-full')


def launch(command, log, metadata, **fields):
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    record = dict(pid=process.pid, start_time_ticks=process_identity(process.pid),
                  command=command, **fields)
    atomic(ROOT / 'manifests' / metadata, record)
    return process, record


def main():
    original = (ROOT / 'origin-reproduction').resolve(strict=True)
    initial = json.loads((ROOT / 'status.json').read_text())
    name = initial['video_full_run']
    output = ROOT / 'outputs' / name
    log = ROOT / 'logs' / (name + '.log')
    status_path = ROOT / 'manifests/parallel-video-training-status.json'
    report = dict(status='running', stage='checking_video_baseline_prerequisites',
                  run_name=name, gpus='0,2', target_steps=700,
                  checkpoint_storage_is_temporary=True, training_method_changed=False)

    def progress(stage, **fields):
        report.update(stage=stage, observed_utc=datetime.now(timezone.utc).isoformat(), **fields)
        if log.exists():
            text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text(errors='replace'))
            steps = re.findall(r'\bstep:(\d+) - ', text)
            report['latest_completed_step'] = int(steps[-1]) if steps else None
        atomic(status_path, report)
        print(stage, report.get('latest_completed_step'), flush=True)
        try:
            atomic(original / 'manifests' / status_path.name, report)
            if log.exists():
                append_log(log, original / 'recovery/20261019T142145Z/ongoing-local-runtime-logs' / log.name)
        except OSError as exc:
            print('Persistent mirror needs retry:', str(exc), flush=True)

    try:
        progress(report['stage'])
        engineering = json.loads((ROOT / 'manifests/video-recovery-engineering-validation.json').read_text())
        if not (engineering['nonzero_weight_update_verified'] and engineering['corrected_export_tensor_bits_unchanged']
                and engineering['real_video_probe_completed']['status'] == 'passed_real_video_tool_probe'):
            raise RuntimeError('Video weight-update and export preflight have not passed')
        if (output / 'invocation.json').exists():
            raise RuntimeError('This full Video run already exists; inspect and resume its checkpoint explicitly')
        storage = os.statvfs('/tmp')
        if storage.f_bavail * storage.f_frsize < 1_250_000_000_000:
            raise RuntimeError('Insufficient temporary space for both baselines and checkpoint rotation')
        assert_idle(['0', '2'])
        command = [str(PYTHON), '-u', str(ROOT / 'scripts/run_training.py'), '--kind', 'video',
                   '--mode', 'full', '--gpus', '0,2', '--name', name, '--save-freq', '20',
                   '--checkpoint-dir', '/tmp/pyvision-rl-storage/checkpoints/' + name,
                   '--ray-temp-dir', str(RAY_ROOT)]
        training, metadata = launch(command, log, 'parallel-video-training-process.json', run_name=name)
        report['training_process'] = metadata
        update_state_fields(parallel_video_training_started=True, parallel_video_training_gpus='0,2',
                            parallel_video_training_completed=False, shared_judge_lifetime='independent_of_trainers')
        audit, audit_metadata = launch([
            str(PYTHON), '-u', str(ROOT / 'scripts/validate_optimizer_update.py'),
            '--run', str(output), '--source-model', str(ROOT / 'models/PyVision-Video-7B-SFT'),
            '--minimum-step', '20', '--report', str(ROOT / 'manifests/recovered-video-weight-update-validation.json')],
            ROOT / 'logs' / (name + '-weight-audit.log'), 'parallel-video-weight-audit-process.json', run_name=name)
        report['weight_audit_process'] = audit_metadata
        bound_session = None
        while training.poll() is None:
            session = RAY_ROOT / 'ray/session_latest'
            if (output / 'invocation.json').exists() and session.exists():
                if bound_session is None:
                    bound_session = session.resolve()
                if list((bound_session / 'logs').glob('worker-*.out')):
                    archive(bound_session, output)
            progress('running_700_step_video_baseline',
                     weight_audit_returncode=audit.poll())
            time.sleep(60)
        if training.returncode or not completed_training(name, 700):
            raise RuntimeError('Video baseline stopped before a successful step-700 checkpoint; inspect its preserved log')
        update_state_fields(parallel_video_training_completed=True, video_rl_700_steps_completed=True)
        progress('video_training_complete_pending_final_model_evaluation', status='complete')
    except BaseException as exc:
        progress('needs_attention', status='failed', error=str(exc))
        raise


if __name__ == '__main__':
    for variable in ['NO_PROXY', 'no_proxy']:
        os.environ[variable] = '127.0.0.1,localhost'
    with (ROOT / 'parallel-video-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(ROOT / 'manifests/parallel-video-controller-process.json', dict(
            pid=os.getpid(), start_time_ticks=process_identity(os.getpid()),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        main()
