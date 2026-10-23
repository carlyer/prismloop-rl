"""Wait for an existing SFT control; resume only a confirmed budget-classification failure."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from evaluate_official import context_budget_details, invocation_fingerprint
from run_training import ROOT, REPO, PYTHON


def write_atomic(path, record):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def live(record):
    try:
        fields = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == str(record['start_time_ticks'])
    except FileNotFoundError:
        return False


def recoverable_failure(kind, status):
    if status.get('status') != 'failed':
        raise RuntimeError('Owner did not report a known failed control')
    prefix = 'official-' + kind + '-sft-control-'
    benchmarks = status.get('benchmarks', [])
    matches = [b for b in benchmarks if status.get('error') ==
               'Command failed (2); see ' + prefix + b + '.log']
    if len(matches) != 1:
        raise RuntimeError('Failure is not a completed evaluator with infrastructure records')
    benchmark = matches[0]
    output = ROOT / 'outputs' / (prefix + benchmark)
    summary = json.loads((output / 'summary.json').read_text())
    invocation = json.loads((output / 'invocation.json').read_text())
    expected = invocation['samples'] * invocation['repeat']
    if (not summary.get('full_dataset_evaluated') or summary.get('self_test_only') or
            summary.get('samples') != expected):
        raise RuntimeError('Refusing to resume from an incomplete or simulated sweep')
    records = [json.loads(p.read_text()) for p in output.glob('*.json')
               if re.fullmatch(r'.+-repeat\d+\.json', p.name)]
    if len(records) != expected or any(r.get('invocation_sha256') != invocation_fingerprint(invocation) for r in records):
        raise RuntimeError('Result coverage/configuration does not match the original invocation')
    failures = [r for r in records if r.get('failure_type') == 'infrastructure']
    if not failures or any(r.get('score') != 0.0 or not context_budget_details(
            r.get('error', ''), r.get('error_class'), r.get('failure_phase'), invocation['context_length'])
            for r in failures):
        raise RuntimeError('A genuine or unrecognized infrastructure failure still needs attention')
    return dict(benchmark=benchmark, records=expected,
                confirmed_context_budget_question_ids=[r['question_id'] for r in failures],
                original_invocation_sha256=invocation_fingerprint(invocation))


def main(args):
    metadata = ROOT / 'manifests' / (args.kind + '-sft-control-process.json')
    status_path = ROOT / 'manifests' / (args.kind + '-sft-control-status.json')
    recovery_path = ROOT / 'manifests' / (args.kind + '-sft-context-recovery-status.json')
    fields = dict(status='running', kind=args.kind, no_live_evaluator_interruption=True)
    def progress(stage, **extra):
        fields.update(stage=stage, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
        write_atomic(recovery_path, fields)
        print(json.dumps(fields), flush=True)
    try:
        owner = json.loads(metadata.read_text())
        if owner.get('start_time_ticks') is None:
            raise RuntimeError('Original control identity lacks its start time')
        progress('waiting_for_current_owner', original_owner=owner)
        while live(owner):
            current = json.loads(metadata.read_text())
            if (current['pid'], str(current.get('start_time_ticks'))) != (owner['pid'], str(owner['start_time_ticks'])):
                progress('superseded_by_another_owner', status='complete')
                return
            time.sleep(20)
        current = json.loads(metadata.read_text())
        if (current['pid'], str(current.get('start_time_ticks'))) != (owner['pid'], str(owner['start_time_ticks'])):
            progress('superseded_by_another_owner', status='complete')
            return
        status = json.loads(status_path.read_text())
        if status.get('status') == 'complete':
            progress('original_control_completed', status='complete')
            return
        evidence = recoverable_failure(args.kind, status)
        progress('restarting_control_after_confirmed_budget_error', evidence=evidence)
        # Preserve the previous owner's records and metadata. The new evaluator
        # performs the reviewed, fingerprint-checked cache adoption itself.
        write_atomic(ROOT / 'manifests' / (args.kind + '-sft-control-before-context-recovery.json'), owner)
        gpu, port = ('7', '18963') if args.kind == 'image' else ('6', '18964')
        command = [str(PYTHON), '-u', str(ROOT / 'scripts/run_image_sft_control.py'),
                   '--kind', args.kind, '--gpu', gpu, '--port', port]
        if args.kind == 'video':
            command.append('--preflight-training')
        with (ROOT / 'logs' / (args.kind + '-sft-context-recovery.log')).open('ab') as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        ticks = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write_atomic(metadata, dict(pid=process.pid, start_time_ticks=ticks, command=command,
                                   reason='Resume only after the completed original sweep hit confirmed context-budget classification'))
        progress('waiting_for_resumed_control', resumed_pid=process.pid, resumed_start_time_ticks=ticks)
        while process.poll() is None:
            time.sleep(20)
        final = json.loads(status_path.read_text())
        if process.returncode or final.get('status') != 'complete':
            raise RuntimeError('Resumed control stopped; no further automatic retries were attempted')
        progress('resumed_control_completed', status='complete')
    except Exception as exc:
        progress('needs_attention', status='failed', error_class=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['image', 'video'], required=True)
    args = parser.parse_args()
    with (ROOT / (args.kind + '-sft-context-recovery.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ticks = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write_atomic(ROOT / 'manifests' / (args.kind + '-sft-context-recovery-process.json'),
                     dict(pid=os.getpid(), start_time_ticks=ticks,
                          script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        main(args)
