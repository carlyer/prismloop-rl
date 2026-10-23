"""Gate Image checkpoint recovery and SFT continuation on a real two-step update."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from pipeline import completed_training, state, stop_owned
from run_training import ROOT, REPO, PYTHON, assert_idle


def write(path, record):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def identity(process, command):
    ticks = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    return dict(pid=process.pid, start_time_ticks=ticks, command=command,
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


def launch(command, log_name, metadata):
    with (ROOT / 'logs' / log_name).open('ab') as stream:
        stream.write(('\nRECOVERY COMMAND: ' + json.dumps(command) + '\n').encode())
        process = subprocess.Popen(command, cwd=REPO, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    write(ROOT / 'manifests' / metadata, identity(process, command))
    return process


def wait_service(port, owner, timeout=1200):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if owner.poll() is not None:
            raise RuntimeError(f'Service owner exited before port {port} became healthy')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise RuntimeError(f'Service {port} startup timed out')


class ExistingJudge:
    def __init__(self, record):
        self.pid = record['pid']
        self.start_ticks = str(record['start_time_ticks'])
        if not any(value in [str(ROOT / 'scripts/serve_judge.py'), str((ROOT / 'origin-reproduction').resolve() / 'scripts/serve_judge.py')] for value in record['command']):
            raise RuntimeError('Recorded owner is not the reproduction judge')
        if self.poll() is not None:
            raise RuntimeError('Recorded judge owner is no longer live')

    def poll(self):
        try:
            fields = Path(f'/proc/{self.pid}/stat').read_text().rsplit(')', 1)[1].split()
            return None if fields[0] != 'Z' and fields[19] == self.start_ticks else 1
        except FileNotFoundError:
            return 1

    def wait(self, timeout=None):
        deadline = time.monotonic() + (timeout or 30)
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired('Existing judge', timeout)
            time.sleep(.2)
        return 0


def main(args):
    # The restored host has a global HTTP proxy; inference endpoints are local.
    for variable in ['NO_PROXY', 'no_proxy']:
        previous = os.environ.get(variable, '')
        os.environ[variable] = '127.0.0.1,localhost' + (',' + previous if previous else '')
    backup = Path(args.archive).resolve(strict=True)
    if not (backup / 'prior-sft-result-records.json').is_file():
        raise RuntimeError('Original SFT result backup has not finished')
    ticks = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
    write(ROOT / 'manifests/pipeline-process.json', dict(
        pid=os.getpid(), start_time_ticks=ticks, command=sys.argv,
        scope='Two-step verification, Image RL recovery, and SFT controls only'))
    judge = None
    children = {}
    training = None
    try:
        state('starting_recovery_judge', status='running', recovery_archive=str(backup), error=None)
        if args.reuse_judge:
            judge = ExistingJudge(json.loads((ROOT / 'manifests/recovery-judge-process.json').read_text()))
            print('Reusing verified judge owner', judge.pid, flush=True)
        else:
            judge = launch([str(PYTHON), '-u', str(ROOT / 'scripts/serve_judge.py')],
                           args.prefix + '-judge.log', 'recovery-judge-process.json')
        wait_service(18961, judge)
        state('running_two_step_weight_verification', status='running', recovery_smoke_run=args.prefix + '-smoke')
        smoke = launch([str(PYTHON), '-u', str(ROOT / 'scripts/run_training.py'), '--mode', 'smoke',
                        '--steps', '2', '--gpus', '1,4', '--name', args.prefix + '-smoke',
                        '--checkpoint-dir', str(Path(args.checkpoint_root) / (args.prefix + '-smoke'))],
                       args.prefix + '-smoke.log', 'recovery-smoke-process.json')
        if smoke.wait() != 0:
            raise RuntimeError('Two-step engineering training or weight comparison failed')
        report_path = ROOT / 'outputs' / (args.prefix + '-smoke') / 'optimizer-update-validation.json'
        report = json.loads(report_path.read_text())
        if report.get('status') != 'passed_nonzero_weight_update' or not report.get('nonzero_weight_update_verified'):
            raise RuntimeError('Two-step run did not prove a nonzero parameter update')
        state('two_step_weight_verification_passed', two_step_weight_update_verified=True,
              recovery_weight_report=str(report_path))
        deadline = time.monotonic() + 300
        while True:
            try:
                assert_idle(['1', '4'])
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Smoke workers did not release training GPUs')
                time.sleep(5)
        checkpoint = ROOT / 'outputs/image-reproduction-20261016T100702Z-full/checkpoints/global_step_1'
        name = args.prefix + '-full'
        ray_link = Path('/tmp/pyvision-rl-ray/ray/session_latest')
        preceding_session = ray_link.resolve() if ray_link.exists() else None
        state('resuming_700_step_image_rl', full_run=name, status='running',
              resumed_from=str(checkpoint), checkpoint_save_frequency=20,
              completed_updates_current_run=1, previous_observed_updates=42,
              previous_step42_weights_available=False, image_nonzero_weight_update_verified=False)
        training = launch([str(PYTHON), '-u', str(ROOT / 'scripts/run_training.py'), '--mode', 'full',
                           '--name', name, '--gpus', '1,4', '--resume-from', str(checkpoint),
                           '--save-freq', '20', '--validation', str(ROOT / 'data/prepared/vstar.json'),
                           '--checkpoint-dir', str(Path(args.checkpoint_root) / name)],
                          args.prefix + '-full.log', 'recovery-image-training-process.json')
        for kind, gpu, port in [('image', '7', '18963'), ('video', '6', '18964')]:
            children[kind] = launch([str(PYTHON), '-u', str(ROOT / 'scripts/run_image_sft_control.py'),
                                     '--kind', kind, '--gpu', gpu, '--port', port],
                                    args.prefix + '-' + kind + '-sft.log', kind + '-sft-control-process.json')
        wait_service(18963, children['image'])
        children['mathvision-mini'] = launch([
            str(PYTHON), '-u', str(ROOT / 'scripts/evaluate_official.py'), '--benchmark', 'mathvision-mini',
            '--output', str(ROOT / 'outputs/official-image-sft-control-mathvision-mini'),
            '--endpoint', 'http://127.0.0.1:18963/v1', '--workers', '2'],
            args.prefix + '-mathvision-mini.log', 'official-image-sft-control-mathvision-mini-process.json')
        write(ROOT / 'manifests/recovery-launch.json', dict(
            status='launched_after_verified_weight_update', prefix=args.prefix,
            archive=str(backup), smoke_report=str(report_path), training_pid=training.pid,
            controls={kind: process.pid for kind, process in children.items()},
            original_training_output_unchanged=True, checkpoint_interval=20,
            original_result_backup=str(backup / 'prior-sft-result-records.tar.gz')))
        # Verify the resumed model itself at its first new saved checkpoint.
        children['weight-audit'] = launch([
            str(PYTHON), '-u', str(ROOT / 'scripts/validate_optimizer_update.py'),
            '--run', str(ROOT / 'outputs' / name),
            '--source-model', str(ROOT / 'models/PyVision-Image-7B-SFT'), '--minimum-step', '20',
            '--report', str(ROOT / 'manifests/recovered-image-weight-update-validation.json')],
            args.prefix + '-weight-audit.log', 'recovery-image-weight-audit-process.json')
        recorded = set()
        bound_session = None
        last_archive = 0.0
        while training.poll() is None or any(p.poll() is None for k, p in children.items() if k != 'weight-audit'):
            if judge.poll() is not None:
                raise RuntimeError('Judge service exited during recovery workloads')
            for kind, process in [('training', training), *children.items()]:
                if process.poll() is not None and kind not in recorded:
                    recorded.add(kind)
                    write(ROOT / 'manifests' / ('recovery-exit-' + kind + '.json'), dict(
                        returncode=process.returncode, prefix=args.prefix,
                        observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
                    print(kind, 'exited', process.returncode, flush=True)
                    if kind == 'training' and process.returncode:
                        state('recovered_image_training_failed', status='failed',
                              error='See ' + args.prefix + '-full.log')
                    if kind == 'weight-audit' and process.returncode == 0:
                        state('resuming_700_step_image_rl', image_nonzero_weight_update_verified=True,
                              image_optimizer_audit=str(ROOT / 'manifests/recovered-image-weight-update-validation.json'))
            if training.poll() is None and time.monotonic() - last_archive >= 120:
                if bound_session is None and ray_link.exists() and ray_link.resolve() != preceding_session:
                    bound_session = ray_link.resolve()
                if bound_session and list((bound_session / 'logs').glob('worker-*.out')):
                    from archive_training_logs import archive
                    try:
                        receipt = archive(bound_session, ROOT / 'outputs' / name)
                        print('Periodic worker-log archive:', receipt['status'], flush=True)
                    except OSError as exc:
                        print('Worker-log archive needs retry:', str(exc), flush=True)
                    last_archive = time.monotonic()
            time.sleep(20)
        failures = {k: p.returncode for k, p in [('training', training), *children.items()]
                    if k != 'weight-audit' and p.returncode}
        if training.returncode == 0 and not completed_training(name, 700):
            failures['training_checkpoint'] = 'Missing successful step-700 checkpoint'
        state('image_training_and_sft_controls_finished' if not failures else 'recovery_workloads_need_attention',
              status='completed' if not failures else 'failed', recovery_failures=failures,
              image_rl_700_steps_completed=completed_training(name, 700),
              image_main_evaluation_completed=False, paper_results_reproduced=False)
    except BaseException as exc:
        state('recovery_needs_attention', status='failed', error=str(exc))
        # Keep launched workloads and their judge available for inspection.
        # Before formal launch there are no independent workloads to preserve.
        if training is None:
            stop_owned(judge)
        raise
    finally:
        if training is not None and training.poll() is not None and all(
                p.poll() is not None for k, p in children.items() if k != 'weight-audit'):
            stop_owned(judge)
            if 'weight-audit' in children:
                stop_owned(children['weight-audit'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--checkpoint-root', required=True)
    parser.add_argument('--reuse-judge', action='store_true')
    arguments = parser.parse_args()
    with (ROOT / 'recovery-launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(arguments)
