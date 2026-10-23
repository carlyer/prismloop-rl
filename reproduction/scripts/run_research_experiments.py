"""Run two controlled research variants from frozen baseline checkpoints."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from archive_training_logs import archive, process_identity
from download_preparation_queue import write
from run_training import ROOT, REPO, PYTHON, assert_idle

SPEC = {
    'video': dict(gpus='0,2', initial_step=120, checkpoint='/tmp/pyvision-rl-storage/checkpoints/video-recovery-20261019T1622Z-full/global_step_120'),
    'image': dict(gpus='1,3', initial_step=20, checkpoint='/tmp/pyvision-rl-storage/checkpoints/image-recovery-local-20261019T150230Z-full/global_step_20'),
}


def command_for(kind, variant):
    spec = SPEC[kind]
    name = f'{kind}-research-{variant}-20261021'
    return [str(PYTHON), '-u', str(ROOT/'scripts/run_training.py'), '--kind', kind,
            '--mode', 'full', '--gpus', spec['gpus'], '--name', name,
            '--resume-from', spec['checkpoint'], '--steps', str(spec['initial_step']+20),
            '--save-freq', '5', '--optimizer-foreach', 'false', '--research-variant', variant,
            '--checkpoint-dir', '/tmp/pyvision-rl-storage/research-checkpoints/'+name,
            '--ray-temp-dir', f'/tmp/pvr-{kind}-{variant}'], name


def main():
    states, children = {}, {}
    for kind in SPEC:
        states[kind] = dict(status='pending', variant='core', initial_checkpoint=SPEC[kind]['checkpoint'],
                            core_updates=20, ppo2_updates=20, ppo2_starts_from_same_initial_checkpoint=True)
    status = dict(status='running', states=states, baseline_training_stopped=True,
                  existing_evaluators_unchanged=True, checkpoints_to_keep_per_research_run=1)
    path = ROOT/'manifests/research-training-status.json'
    write(ROOT/'manifests/research-controller-process.json', dict(pid=os.getpid(), start_time_ticks=process_identity(os.getpid())))
    while True:
        for kind, state in states.items():
            try:
                if state['status'] == 'pending':
                    if os.statvfs('/tmp').f_bavail*os.statvfs('/tmp').f_frsize < 220*1024**3:
                        state['waiting_reason'] = 'checkpoint storage headroom below 220 GiB'
                        continue
                    assert_idle(SPEC[kind]['gpus'].split(','))
                    command, name = command_for(kind, state['variant'])
                    out = ROOT/'outputs'/name
                    if (out/'invocation.json').exists():
                        raise RuntimeError('Run already exists; refusing to overwrite '+name)
                    with (ROOT/'logs'/f'{name}.log').open('ab') as log:
                        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                                 stdin=subprocess.DEVNULL, start_new_session=True)
                    children[kind] = child
                    record = dict(pid=child.pid, start_time_ticks=process_identity(child.pid), command=command)
                    write(ROOT/'manifests'/f'{name}-process.json', record)
                    state.update(status='running', name=name, process=record, started_utc=datetime.now(timezone.utc).isoformat())
                if state['status'] == 'running':
                    child = children[kind]
                    log = ROOT/'logs'/f"{state['name']}.log"
                    text = log.read_text(errors='replace') if log.exists() else ''
                    steps = re.findall(r'\bstep:(\d+) - ', text)
                    state['latest_completed_global_step'] = int(steps[-1]) if steps else SPEC[kind]['initial_step']
                    state['new_updates_completed'] = state['latest_completed_global_step'] - SPEC[kind]['initial_step']
                    ray = Path(f"/tmp/pvr-{kind}-{state['variant']}/ray/session_latest")
                    out = ROOT/'outputs'/state['name']
                    if ray.exists() and (out/'invocation.json').exists() and time.time()-state.get('last_archive_unix', 0) > 120:
                        archive(ray, out)
                        state['last_archive_unix'] = time.time()
                    tracker = out/'checkpoints/latest_checkpointed_iteration.txt'
                    if tracker.exists():
                        saved = int(tracker.read_text())
                        state['latest_saved_step'] = saved
                        audit = ROOT/'manifests'/f"{state['name']}-update-audit.json"
                        if not audit.exists() and saved > SPEC[kind]['initial_step']:
                            audit_command = [str(PYTHON), str(ROOT/'scripts/validate_research_update.py'), '--kind', kind,
                                             '--checkpoint', str(out/'checkpoints'/f'global_step_{saved}'),
                                             '--report', str(audit)]
                            result = subprocess.run(audit_command, cwd=REPO, capture_output=True, text=True)
                            state['update_audit_returncode'] = result.returncode
                            if result.returncode:
                                raise RuntimeError('Research update audit failed: '+result.stderr[-1000:])
                    if child.poll() is not None:
                        receipt = json.loads((out/'exit-status.json').read_text()) if (out/'exit-status.json').exists() else {}
                        if child.returncode or receipt.get('returncode') != 0 or state.get('latest_saved_step') != SPEC[kind]['initial_step']+20:
                            raise RuntimeError('Training did not complete: '+str(receipt))
                        # Reap only the finished run's Ray process group.
                        try:
                            os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        if state['variant'] == 'core':
                            state['core_result'] = dict(name=state['name'], checkpoint_step=state['latest_saved_step'])
                            state.update(status='pending', variant='ppo2')
                        else:
                            state['status'] = 'complete'
            except Exception as exc:
                state.update(status='failed', error=str(exc))
        status['observed_utc'] = datetime.now(timezone.utc).isoformat()
        if all(s['status'] in ['complete', 'failed'] for s in states.values()):
            status['status'] = 'complete' if all(s['status']=='complete' for s in states.values()) else 'needs_attention'
            write(path, status)
            return
        write(path, status)
        time.sleep(20)


if __name__ == '__main__':
    for variable in ['NO_PROXY', 'no_proxy']:
        os.environ[variable] = '127.0.0.1,localhost'
    with (ROOT/'research-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
