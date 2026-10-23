"""Verify that automatic continuation rejects genuine/incomplete failures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import resume_sft_after_context_budget as recovery

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'references/context-budget-original-case'
    source_record = (source / '904-repeat0.json').read_bytes()
    actual = json.loads(source_record)
    invocation = json.loads((source / 'invocation.json').read_text())
    invocation.update(samples=2, repeat=1)
    fingerprint = recovery.invocation_fingerprint(invocation)
    actual['invocation_sha256'] = fingerprint
    good = dict(actual, question_id='905', score=1.0)
    for key in ['failure_type', 'error', 'error_class', 'failure_phase']:
        good.pop(key, None)
    baseline = [actual, good]
    summary = dict(full_dataset_evaluated=True, self_test_only=False, samples=2)
    owner = dict(status='failed', benchmarks=['vsibench'],
                 error='Command failed (2); see official-video-sft-control-vsibench.log')
    checks = []
    with tempfile.TemporaryDirectory(prefix='pyvision-context-recovery-') as directory:
        root = Path(directory)
        output = root / 'outputs/official-video-sft-control-vsibench'
        output.mkdir(parents=True)
        (output / 'invocation.json').write_text(json.dumps(invocation))
        def reset(rows=baseline, report=summary):
            (output / 'summary.json').write_text(json.dumps(report))
            for r in rows:
                (output / (r['question_id'] + '-repeat0.json')).write_text(json.dumps(r))
        def rejected(name, status=owner):
            try:
                recovery.recoverable_failure('video', status)
            except RuntimeError:
                checks.append(name)
            else:
                raise AssertionError(name)
        original_root = recovery.ROOT
        try:
            recovery.ROOT = root
            reset()
            evidence = recovery.recoverable_failure('video', owner)
            assert evidence['confirmed_context_budget_question_ids'] == ['904']
            checks.append('only_complete_known_budget_failure_can_resume')
            rejected('reject_running_owner', dict(owner, status='running'))
            rejected('reject_other_command_failure', dict(owner, error='Command failed (1); see train-video.log'))
            reset(report=dict(summary, full_dataset_evaluated=False))
            rejected('reject_partial_dataset')
            reset(report=dict(summary, self_test_only=True))
            rejected('reject_simulated_run')
            reset(report=dict(summary, samples=1))
            rejected('reject_missing_result_coverage')
            for name, update in [
                ('reject_genuine_infrastructure', dict(error_class='ConnectionError', error='Connection refused')),
                ('reject_scoring_service_failure', dict(failure_phase='scoring')),
                ('reject_nonzero_error_score', dict(score=1.0)),
                ('reject_mixed_result_configuration', dict(invocation_sha256='another-configuration')),
                ('reject_unrecognized_http_400', dict(error='Error code: 400 - invalid role')),
            ]:
                rows = copy.deepcopy(baseline)
                rows[0].update(update)
                reset(rows=rows)
                rejected(name)
            reset(rows=[dict(good, question_id='904'), good])
            rejected('reject_failure_without_infrastructure_records')
        finally:
            recovery.ROOT = original_root
    assert (source / '904-repeat0.json').read_bytes() == source_record
    checks.append('frozen_original_fixture_unchanged')
    report = dict(status='passed', checks=len(checks), check_names=checks,
                  recovery_script_sha256=hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest(),
                  automatic_restart_performed=False, live_outputs_modified=False)
    (ROOT / 'manifests/sft-context-recovery-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
