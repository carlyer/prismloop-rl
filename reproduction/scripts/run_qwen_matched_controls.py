"""Run the requested Qwen/SFT/RL comparison without altering prior evaluations."""
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
import urllib.request

from download_preparation_queue import ROOT, write
from evaluation_evidence import audit_records
from pipeline import evaluate_suite

BENCHMARKS = ['vstar', 'hrbench-4k']
PROTOCOL_KEYS = ['benchmark', 'data_sha256', 'samples', 'full_dataset_samples',
                 'limit', 'question_ids', 'repeat', 'config', 'max_generation_turns',
                 'context_length', 'judge', 'adaptations', 'sampling_source',
                 'self_test_only', 'supplementary_metric_protocol', 'multi_image_prompt',
                 'template_placeholder_fix', 'vsi_choices']
RUNS = {
    'qwen': {b: 'official-qwen-instruct-control-' + b for b in BENCHMARKS},
    'sft': {b: 'official-image-sft-control-' + b for b in BENCHMARKS},
    'rl': {'vstar': 'official-image-rl-control-vstar',
           'hrbench-4k': 'official-image-rl-matched-control-hrbench-4k'},
}


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def identity():
    return dict(pid=os.getpid(), start_time_ticks=Path(f'/proc/{os.getpid()}/stat').read_text()
                .rsplit(')', 1)[1].split()[19], started_utc=now(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def compare_protocol(invocation, reference):
    differences = [k for k in PROTOCOL_KEYS if invocation.get(k) != reference.get(k)]
    if differences:
        raise ValueError('Evaluation conditions differ from SFT control: ' + ', '.join(differences))


def read(path):
    return json.loads(path.read_text())


def build_comparison():
    record = dict(observed_utc=now(), status='pending', results={},
                  comparison_scope='Same local agent/tool evaluation, not the paper Qwen direct-inference baseline',
                  pending=[], errors=[], paper_numbers_reproduced=False)
    lines = ['Qwen / official PyVision SFT / official PyVision RL: matched local evaluation',
             'VStar avg@32; HRBench-4K once per question; identical prompts, tools, sampling and local judge.',
             'Qwen is the official Instruct checkpoint before additional PyVision training.',
             'Partial records are never accepted as scores.', '',
             'Benchmark | Qwen (%) | SFT (%) | RL (%) | SFT-Qwen (pp) | RL-Qwen (pp) | RL-SFT (pp)']
    for benchmark in BENCHMARKS:
        scores = {}
        reference = read(ROOT / 'outputs' / RUNS['sft'][benchmark] / 'invocation.json')
        rows = {}
        for kind in ['qwen', 'sft', 'rl']:
            folder = ROOT / 'outputs' / RUNS[kind][benchmark]
            entry = dict(run=folder.name, status='pending')
            if (folder / 'summary.json').exists():
                try:
                    invocation, summary = read(folder / 'invocation.json'), read(folder / 'summary.json')
                    compare_protocol(invocation, reference)
                    expected_model = (ROOT / 'models' / {'qwen': 'Qwen2.5-VL-7B-Instruct',
                                      'sft': 'PyVision-Image-7B-SFT', 'rl': 'PyVision-Image-7B-RL'}[kind]).resolve()
                    actual = {Path(m['root']).resolve() for m in invocation.get('policy_models', [])}
                    if actual != {expected_model}:
                        raise ValueError('Evaluated checkpoint does not match comparison role')
                    evidence = audit_records(folder, invocation, summary,
                                             ROOT / 'data/evaluation' / (benchmark + '.json'))
                    score = 100 * summary['score_mean']
                    entry.update(status='complete', accuracy_percent=score, evidence=evidence,
                                 model_root=str(expected_model), model_failures=summary['failures'])
                    scores[kind] = score
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    entry.update(status='failed_audit', error=str(exc))
                    record['errors'].append(dict(run=folder.name, error=str(exc)))
            else:
                record['pending'].append(folder.name)
            rows[kind] = entry
        row = dict(models=rows)
        if len(scores) == 3:
            row['deltas_percentage_points'] = dict(sft_minus_qwen=scores['sft']-scores['qwen'],
                rl_minus_qwen=scores['rl']-scores['qwen'], rl_minus_sft=scores['rl']-scores['sft'])
        values = [f'{scores[k]:.3f}' if k in scores else 'pending' for k in ['qwen', 'sft', 'rl']]
        deltas = row.get('deltas_percentage_points', {})
        values += [f'{deltas[k]:+.3f}' if k in deltas else 'pending'
                   for k in ['sft_minus_qwen', 'rl_minus_qwen', 'rl_minus_sft']]
        lines.append(benchmark + ' | ' + ' | '.join(values))
        record['results'][benchmark] = row
    if not record['pending'] and not record['errors']:
        record['status'] = 'complete'
    elif record['errors']:
        record['status'] = 'needs_attention'
    write(ROOT / 'manifests/qwen-sft-rl-matched-comparison.json', record)
    target = ROOT / 'QWEN_SFT_RL_MATCHED_COMPARISON.txt'
    temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
    temporary.write_text('\n'.join(lines) + '\n\nObserved UTC: ' + record['observed_utc'] + '\n')
    temporary.replace(target)
    return record


def preflight(benchmarks):
    selection = read(ROOT / 'judge-selection.json')
    with urllib.request.urlopen('http://127.0.0.1:18961/v1/models', timeout=15) as response:
        models = json.load(response)['data']
    if not any(m.get('id') == 'judge' and m.get('root') == selection['model_path'] for m in models):
        raise RuntimeError('Shared judge differs from frozen local selection')
    for benchmark in benchmarks:
        command = [sys.executable, str(ROOT / 'scripts/evaluate_official.py'), '--benchmark', benchmark,
                   '--output', str(ROOT / 'outputs' / RUNS['qwen'][benchmark]), '--config-only']
        if benchmark == 'vstar':
            command += ['--repeat', '32']
        current = json.loads(subprocess.check_output(command, text=True))
        compare_protocol(current, read(ROOT / 'outputs' / RUNS['sft'][benchmark] / 'invocation.json'))


def worker(role):
    qwen = role == 'qwen'
    key = 'qwen-instruct-control' if qwen else 'image-rl-matched-control'
    benchmarks = BENCHMARKS if qwen else ['hrbench-4k']
    gpu, port = ('1', 18968) if qwen else ('4', 18969)
    model = ROOT / 'models' / ('Qwen2.5-VL-7B-Instruct' if qwen else 'PyVision-Image-7B-RL')
    with (ROOT / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(ROOT / 'manifests' / (key + '-process.json'), identity())
        fields = dict(status='running', stage='preflight', role=role, model=str(model.resolve()),
                      gpu=gpu, port=port, benchmarks=benchmarks, started_utc=now(), weights_updated=False)
        status = ROOT / 'manifests' / (key + '-status.json')
        try:
            write(status, fields)
            preflight(benchmarks)
            fields.update(stage='evaluating', updated_utc=now())
            write(status, fields)
            evaluate_suite(model, benchmarks,
                           'official-qwen-instruct-control' if qwen else 'official-image-rl-matched-control',
                           gpu=gpu, port=port, update_state=False)
            for benchmark in benchmarks:
                folder = ROOT / 'outputs' / RUNS[role if qwen else 'rl'][benchmark]
                invocation, summary = read(folder / 'invocation.json'), read(folder / 'summary.json')
                compare_protocol(invocation, read(ROOT / 'outputs' / RUNS['sft'][benchmark] / 'invocation.json'))
                audit_records(folder, invocation, summary, ROOT / 'data/evaluation' / (benchmark + '.json'))
            fields.update(status='complete', stage='full_evaluation_verified')
        except BaseException as exc:
            fields.update(status='failed', stage='needs_attention', error=str(exc))
            raise
        finally:
            fields['updated_utc'] = now()
            write(status, fields)


def main():
    with (ROOT / 'qwen-matched-comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(ROOT / 'manifests/qwen-matched-comparison-process.json', identity())
        build_comparison()
        children = []
        for role in ['qwen', 'rl-hrbench']:
            log = (ROOT / 'logs' / ('official-qwen-matched-' + role + '-controller.log')).open('a')
            command = [sys.executable, '-u', str(Path(__file__)), '--role', role]
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close()
            children.append(child)
        while any(p.poll() is None for p in children):
            build_comparison()
            time.sleep(30)
        result = build_comparison()
        if any(p.returncode != 0 for p in children) or result['status'] != 'complete':
            raise RuntimeError('Matched comparison incomplete; inspect worker statuses and comparison report')
        subprocess.run([sys.executable, str(ROOT / 'scripts/collect_results.py')], check=True)


if __name__ == '__main__':
    for variable in ['NO_PROXY', 'no_proxy']:
        os.environ[variable] = '127.0.0.1,localhost'
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['qwen', 'rl-hrbench'])
    args = parser.parse_args()
    worker(args.role) if args.role else main()
