"""Release only three identified pre-fix evaluators stuck after verified completion."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from download_preparation_queue import ROOT, write
from evaluation_evidence import audit_records


def fields(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()


if __name__ == '__main__':
    manifest = ROOT / 'manifests/legacy-evaluator-drain-plan.json'
    jobs = json.loads(manifest.read_text())['jobs']
    observed = set()
    while jobs:
        pending = []
        for job in jobs:
            pid = job['pid']
            try:
                state = fields(pid)
                if state[0] == 'Z' or state[19] != job['start_time_ticks']:
                    continue
                cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                args = [part.decode() for part in cmd if part]
                if (str(ROOT / 'scripts/evaluate_official.py') not in args
                        or args[args.index('--output') + 1] != job['output']):
                    raise RuntimeError('Legacy evaluator identity changed')
                pending.append(job)
                folder = Path(job['output'])
                summary, invocation = folder / 'summary.json', folder / 'invocation.json'
                if not summary.exists() or summary.stat().st_mtime < invocation.stat().st_mtime:
                    continue
                report, config = json.loads(summary.read_text()), json.loads(invocation.read_text())
                if report.get('status') != 'complete' or report.get('infrastructure_failures') != 0:
                    continue
                evidence = audit_records(folder, config, report,
                                         ROOT / 'data/evaluation' / (config['benchmark'] + '.json'))
                stack = subprocess.check_output(['/opt/venv/bin/py-spy', 'dump', '--pid', str(pid)],
                                                text=True, timeout=15)
                if '_finalize_join' not in stack or '_exit_function' not in stack:
                    continue
                if pid not in observed:
                    observed.add(pid)
                    continue
                fd = os.pidfd_open(pid)
                try:
                    if fields(pid)[19] != job['start_time_ticks']:
                        continue
                    (folder / 'legacy-shutdown-stack.txt').write_text(stack)
                    write(folder / 'legacy-process-exit-validation.json', dict(
                        status='passed', pid=pid, start_time_ticks=job['start_time_ticks'],
                        evidence=evidence, action='SIGINT to main process blocked in post-completion queue cleanup',
                        inference_and_scoring_unchanged=True))
                    signal.pidfd_send_signal(fd, signal.SIGINT)
                    print('Released verified completed evaluator', pid, flush=True)
                finally:
                    os.close(fd)
            except FileNotFoundError:
                continue
            except Exception as exc:
                print('Inspection deferred:', pid, str(exc), flush=True)
        jobs = pending
        if jobs:
            time.sleep(60)
