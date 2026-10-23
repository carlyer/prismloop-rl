"""Check the records underlying a claimed full benchmark result."""
from collections import defaultdict
import hashlib
import json
import math
import re


def audit_records(folder, config, report, data_path):
    if (config.get('self_test_only') or config.get('limit') is not None
            or config.get('question_ids') is not None):
        raise ValueError('The invocation is a probe or a selected subset')
    data = data_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != config['data_sha256']:
        raise ValueError('Benchmark input differs from the evaluated input')
    rows = json.loads(data)
    identifiers = [str(row.get('question_id', row.get('id'))) for row in rows]
    if len(set(identifiers)) != len(rows):
        raise ValueError('Benchmark question IDs are not unique')
    repeat = config['repeat']
    if not isinstance(repeat, int) or repeat < 1:
        raise ValueError('Invalid repeat count')
    if config['benchmark'] == 'vstar' and repeat != 32:
        raise ValueError('V* paper comparison requires avg@32')
    if config['samples'] != len(rows) or config['full_dataset_samples'] != len(rows):
        raise ValueError('Invocation does not cover the full benchmark')
    expected = {(identifier, rep) for identifier in identifiers for rep in range(repeat)}
    invocation_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    seen, scores, categories, failures = set(), [], defaultdict(list), 0
    for path in folder.iterdir():
        if not re.fullmatch(r'.+-repeat\d+\.json', path.name):
            continue
        record = json.loads(path.read_text())
        key = (str(record['question_id']), record['repeat'])
        if key not in expected or key in seen or path.name != f'{key[0]}-repeat{key[1]}.json':
            raise ValueError('Unexpected, duplicate or misnamed benchmark record: ' + path.name)
        if record.get('invocation_sha256') != invocation_hash:
            raise ValueError('Record belongs to a different invocation: ' + path.name)
        if record.get('failure_type') == 'infrastructure':
            raise ValueError('Infrastructure failure remains in the benchmark: ' + path.name)
        score = record['score']
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid benchmark score: ' + path.name)
        scores.append(score)
        categories[str(record['category'])].append(score)
        failures += 'error' in record
        seen.add(key)
    if seen != expected:
        raise ValueError(f'Incomplete record coverage: {len(seen)}/{len(expected)}')
    def same(actual, expected_value):
        return math.isclose(actual, expected_value, rel_tol=1e-10, abs_tol=1e-12)
    if (report.get('status') != 'complete' or not report.get('full_dataset_evaluated')
            or report.get('self_test_only') or not report.get('real_model_inference_completed')
            or report.get('infrastructure_failures') != 0 or report.get('score_is_provisional')
            or report['samples'] != len(scores) or report['failures'] != failures
            or not same(report['score_mean'], sum(scores) / len(scores))):
        raise ValueError('Summary does not match its complete real-inference records')
    if set(report['categories']) != set(categories):
        raise ValueError('Summary category coverage differs from the records')
    for category, values in categories.items():
        entry = report['categories'][category]
        if entry['count'] != len(values) or not same(entry['score_mean'], sum(values) / len(values)):
            raise ValueError('Summary category score differs from the records: ' + category)
    macro = sum(sum(values) / len(values) for values in categories.values()) / len(categories)
    if not same(report['category_macro_mean'], macro):
        raise ValueError('Summary macro score differs from the records')
    return dict(status='passed', records_checked=len(scores), unique_questions=len(rows),
                repeat=repeat, invocation_sha256=invocation_hash,
                scope='Full question/repeat coverage, finite scores, invocation provenance and summary agreement')
