"""Observe the unchanged training judge on frozen cases with native request parameters."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace
import urllib.request
from unittest.mock import patch

from download_preparation_queue import ROOT, write


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(ROOT.parent / 'verl_agents'))
    from openai import OpenAI
    from verl.utils.reward_score import multi_task_verifier as verifier

    case_path = ROOT / 'manifests/judge-calibration-cases.json'
    calibration_path = ROOT / 'manifests/judge-calibration.json'
    original_calibration = calibration_path.read_bytes()
    calibration = json.loads(original_calibration)
    cases_bytes = case_path.read_bytes()
    cases = json.loads(cases_bytes)
    tasks = [(case, repeat) for case in cases for repeat in range(case['repeats'])]
    source = Path(verifier.__file__)
    source_sha = digest(source.read_bytes())
    assert source_sha == calibration['upstream_verifier_sha256']
    assert digest(cases_bytes) == calibration['case_file_sha256']
    assert len(cases) == calibration['unique_cases'] and len(tasks) == calibration['trials']
    config = json.loads((ROOT / 'judge-config.json').read_text())
    selection = json.loads((ROOT / 'judge-selection.json').read_text())
    with urllib.request.urlopen(config['base_url'] + '/models', timeout=15) as response:
        models = json.load(response)['data']
    assert any(model['id'] == config['model_name'] and model.get('root') == selection['model_path']
               for model in models)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = ROOT / 'outputs' / ('native-training-judge-audit-' + stamp)
    folder.mkdir(exist_ok=False)
    report_path = ROOT / 'manifests/native-training-judge-audit.json'
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  purpose='Native-request diagnostic; does not replace the historical controlled calibration or alter training',
                  source=str(source), source_sha256=source_sha,
                  script_sha256=digest(Path(__file__).read_bytes()),
                  case_file_sha256=digest(cases_bytes),
                  historical_calibration_sha256=digest(original_calibration),
                  unique_cases=len(cases), trials=len(tasks), workers=args.workers,
                  judge_selection_sha256=digest((ROOT / 'judge-selection.json').read_bytes()),
                  advertised_models=[dict(id=model['id'], root=model.get('root')) for model in models],
                  sdk_version=version('openai'), results_path=str(folder / 'results.jsonl'),
                  request_policy='Forward native constructor and create kwargs unchanged; no seed, token cap, prompt or decoding override',
                  limitation='Controlled answer pairs only; not paper-72B agreement, real-trajectory accuracy or a full Bench')
    write(report_path, report)
    write(folder / 'audit-plan.json', report)
    local = threading.local()

    def factory(**constructor):
        assert set(constructor) == {'api_key', 'base_url'}
        client = OpenAI(**constructor)

        def create(**request):
            assert set(request) == {'model', 'messages', 'seed', 'temperature'}
            assert request['model'] == config['model_name'] and request['temperature'] == 0.3
            local.request_parameters = {key: value for key, value in request.items() if key != 'messages'}
            local.prompt_sha256 = digest(json.dumps(request['messages'], sort_keys=True).encode())
            local.calls += 1
            try:
                response = client.chat.completions.create(**request)
                local.raw = response.choices[0].message.content or ''
                local.finish_reason = response.choices[0].finish_reason
                local.usage = response.usage.model_dump() if response.usage else {}
                return response
            finally:
                client.close()

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def run(task):
        case, repeat = task
        for key, value in dict(calls=0, raw=None, usage={}, request_parameters={}, prompt_sha256=None,
                               finish_reason=None).items():
            setattr(local, key, value)
        start = time.monotonic()
        result = dict(case, repeat=repeat)
        try:
            answer = '<answer>\n\\boxed{' + case['answer'] + '}\n</answer>'
            native = verifier.compute_score(answer, case['gold'], {'question': case['question']}, config,
                                            case['source'])
            result.update(prediction=int(native['score']), native_reward=native)
        except Exception as exc:
            result.update(prediction=None, error_class=type(exc).__name__, error=str(exc))
        result.update(matches_expected=result['prediction'] == case['expected'],
                      valid_output=bool(local.raw is not None and re.fullmatch(
                          r'(?:Judgement:\s*)?[01]', local.raw.strip())),
                      raw_judge_output=local.raw, request_parameters=local.request_parameters,
                      prompt_sha256=local.prompt_sha256, api_calls=local.calls,
                      finish_reason=local.finish_reason, usage=local.usage,
                      elapsed_seconds=time.monotonic() - start)
        return result

    rows = []
    with patch.object(verifier, 'OpenAI', factory), ThreadPoolExecutor(max_workers=args.workers) as pool, \
            (folder / 'results.jsonl').open('w') as log:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            if len(rows) % 16 == 0 or len(rows) == len(tasks):
                print(f'Native request audit {len(rows)}/{len(tasks)}', flush=True)
    assert calibration_path.read_bytes() == original_calibration
    assert digest(source.read_bytes()) == source_sha and case_path.read_bytes() == cases_bytes
    total = len(rows)
    negatives = [row for row in rows if row['expected'] == 0]
    metrics = dict(accuracy=sum(row['matches_expected'] for row in rows) / total,
                   false_positive_rate=sum(row['prediction'] == 1 for row in negatives) / len(negatives),
                   valid_output_rate=sum(row['valid_output'] for row in rows) / total,
                   critical_errors=sum(row['critical'] and not row['matches_expected'] for row in rows),
                   request_errors=sum('error' in row for row in rows))
    report.update(status='diagnostic_completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                  completed_trials=total, metrics=metrics,
                  native_request_contract_preserved=all(row['api_calls'] == 1 and
                      set(row['request_parameters']) == {'model', 'seed', 'temperature'} for row in rows),
                  request_failures=[dict(id=row['id'], repeat=row['repeat'], error_class=row['error_class'],
                                         error=row['error']) for row in rows if 'error' in row],
                  mismatches=[dict(id=row['id'], repeat=row['repeat'], prediction=row['prediction'],
                                   expected=row['expected']) for row in rows if not row['matches_expected']],
                  historical_calibration_unchanged=True, training_configuration_unchanged=True,
                  results_sha256=digest((folder / 'results.jsonl').read_bytes()))
    write(report_path, report)
    write(folder / 'report.json', report)
    print(json.dumps(dict(status=report['status'], completed_trials=total, metrics=metrics,
                          native_request_contract_preserved=report['native_request_contract_preserved'])), flush=True)
    return 0 if report['native_request_contract_preserved'] and not metrics['request_errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
