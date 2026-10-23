"""Reclaim only proven closed-rollout tools of the pinned, already running Image job.

No process-tree/group signals. Each signal uses a Linux pidfd after identity, age,
completed-rollout and nonblocking Python stack checks. Defaults to observation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/image-reproduction-20261016T100702Z-full'
PARENTS = {2219515: 3465468544, 2226544: 3465469388}
DRIVER = (2203684, 3465466979)
TOOLS = ROOT.parent / 'verl_agents/verl/workers/agent/envs/agents_x'
TOOL = TOOLS / 'safe_persis_python_exe_tool_w_image_hint.py'
TICKS = os.sysconf('SC_CLK_TCK')
BOOT = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines()
                if line.startswith('btime ')))


def identity(pid):
    try:
        base = Path('/proc') / str(pid)
        fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]))
    except FileNotFoundError:
        return None


def owners_alive():
    for pid, ticks in [DRIVER, *PARENTS.items()]:
        actual = identity(pid)
        if not actual or actual['start_ticks'] != ticks or actual['state'] == 'Z':
            return False
    return True


def eligible(child, boundary, now):
    if not child or child['state'] == 'Z' or child['ppid'] not in PARENTS:
        return False
    birth = BOOT + child['start_ticks'] / TICKS
    return birth < boundary - 120 and now - birth >= 600


def blocked_tool(stack):
    tool_frames = all(fragment in stack for fragment in (
        f'_worker_loop ({TOOL}:', f'exec_code ({TOOL}:'))
    observed_stalls = (
        'write (/opt/pyvision-rl-venv/lib/python3.10/site-packages/ray/_private/utils.py:424)',
        '_kmeans_single_lloyd (/opt/pyvision-rl-venv/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:700)',
    )
    return tool_frames and any(fragment in stack for fragment in observed_stalls)


def memory():
    wanted = {'MemAvailable', 'MemFree', 'AnonPages', 'Shmem', 'Cached'}
    return {parts[0].rstrip(':'): int(parts[1]) for line in Path('/proc/meminfo').read_text().splitlines()
            if (parts := line.split())[0].rstrip(':') in wanted}


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sweep(apply=False):
    if not owners_alive():
        raise RuntimeError('Pinned Image owners exited or changed identity; refusing cleanup')
    metrics = json.loads((ROOT / 'manifests/image-training-metrics.json').read_text())
    if metrics['run'] != RUN.name:
        raise RuntimeError('Unexpected training run')
    expected_log = Path('/tmp/pyvision-rl-ray/ray/session_2026-10-16_10-58-04_282371_2192181/logs/'
                        'worker-b63fdba05d8a06a6064e91921120c860199edf1da192fb41d436129e-01000000-2203684.out')
    if Path(metrics['source']) != expected_log:
        raise RuntimeError('Unexpected training log source')
    steps = re.findall(r'^step:(\d+) - .*actor/grad_norm:', expected_log.read_text(), re.M)
    if not steps:
        raise RuntimeError('No completed training steps in pinned driver log')
    completed = int(steps[-1])
    if completed < max(row['step'] for row in metrics['completed_updates']):
        raise RuntimeError('Training source regressed')
    rollout = RUN / 'rollouts' / f'{completed}.json'
    data = rollout.read_bytes()
    rows = json.loads(data)
    if len(rows) != 128 or any(row['step'] != completed for row in rows):
        raise RuntimeError('Invalid completed rollout boundary')
    boundary = rollout.stat().st_mtime
    now = time.time()
    folder = RUN / 'expired-tool-cleanup' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(observed_utc=datetime.now(timezone.utc).isoformat(), apply=apply,
                  cleanup_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  pinned_parents=PARENTS, pinned_driver=DRIVER, completed_step=completed,
                  rollout=str(rollout), rollout_sha256=hashlib.sha256(data).hexdigest(),
                  rollout_mtime=boundary, memory_before_kib=memory(), children=[],
                  policy='Only direct pinned-actor children born >120s before a verified completed '
                         'rollout, aged >=600s and in exact old tool execution frames with an '
                         'observed Ray I/O or sklearn KMeans stall stack. '
                         'Identity checked again immediately before pidfd SIGKILL. '
                         'No active rollout tools, parents, scores, or trajectories modified.')
    output = folder / 'report.json'
    write(output, report)
    children = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            child = identity(int(entry.name))
            if child and child['ppid'] in PARENTS:
                children.append(child)
    report['direct_children_observed'] = len(children)
    for child in children:
        if not eligible(child, boundary, now):
            continue
        pid = child['pid']
        record = dict(identity=child, age_seconds=now-(BOOT+child['start_ticks']/TICKS),
                      action='observed_only')
        report['children'].append(record)
        try:
            capture = subprocess.run(['/opt/pyvision-rl-venv/bin/py-spy', 'dump', '--pid', str(pid),
                                      '--nonblocking', '--full-filenames'],
                                     capture_output=True, text=True, timeout=10)
            stack = capture.stdout
            (folder / f'{pid}.stack.txt').write_text(stack + capture.stderr)
            record['stack_returncode'] = capture.returncode
            record['blocked_tool_stack_confirmed'] = capture.returncode == 0 and blocked_tool(stack)
            if not record['blocked_tool_stack_confirmed']:
                record['action'] = 'skipped_stack_not_confirmed'
            elif apply:
                fd = os.pidfd_open(pid)
                try:
                    if (not owners_alive() or identity(pid) != child or
                            not eligible(identity(pid), boundary, time.time())):
                        record['action'] = 'skipped_identity_changed'
                    else:
                        # Persist the exact target and stack evidence before sending any signal.
                        record['action'] = 'pidfd_signal_prepared'
                        write(output, report)
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                        record['action'] = 'pidfd_sigkill_sent'
                finally:
                    os.close(fd)
        except (OSError, subprocess.TimeoutExpired) as error:
            record['action'] = 'skipped_observation_error'
            record['error'] = str(error)
        write(output, report)
    if apply:
        time.sleep(2)
    for record in report['children']:
        record['identity_after'] = identity(record['identity']['pid'])
    report['owners_alive_after'] = owners_alive()
    report['memory_after_kib'] = memory()
    report['signals_sent'] = sum(row['action'] == 'pidfd_sigkill_sent' for row in report['children'])
    write(output, report)
    print(json.dumps(dict(receipt=str(output), signals_sent=report['signals_sent'],
                          eligible_observed=len(report['children']),
                          owners_alive_after=report['owners_alive_after'],
                          memory_before_kib=report['memory_before_kib'],
                          memory_after_kib=report['memory_after_kib'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=180)
    arguments = parser.parse_args()
    if arguments.interval < 60:
        parser.error('Watch interval must be at least 60 seconds')
    while True:
        if not owners_alive():
            print(json.dumps(dict(status='stopped', reason='pinned_image_owners_exited_or_changed')),
                  flush=True)
            break
        sweep(arguments.apply)
        if not arguments.watch:
            break
        time.sleep(arguments.interval)
