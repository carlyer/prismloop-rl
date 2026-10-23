"""Offline regression checks for a real overlong-prompt error and cache adoption."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import evaluate_official as evaluator

ROOT = Path(__file__).resolve().parents[1]


def main():
    checked = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checked.append(name)

    original = ROOT / 'references/evaluate_official.before-context-budget-fix.py'
    current = ROOT / 'scripts/evaluate_official.py'
    old_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    new_sha = hashlib.sha256(current.read_bytes()).hexdigest()
    result_path = ROOT / 'references/context-budget-original-case/904-repeat0.json'
    original_result = result_path.read_bytes()
    result = json.loads(original_result)
    details = evaluator.context_budget_details(result['error'], result['error_class'],
                                               result['failure_phase'], 32768)
    check('actual_33130_token_error_is_budget', details is not None and details['decoder_prompt_tokens'] == 33130)
    for name, error, cls, phase, limit in [
        ('equal_limit', result['error'].replace('33130', '32768'), 'BadRequestError', 'generation', 32768),
        ('below_limit', result['error'].replace('33130', '32767'), 'BadRequestError', 'generation', 32768),
        ('different_configured_limit', result['error'], 'BadRequestError', 'generation', 65536),
        ('judge_error', result['error'], 'BadRequestError', 'scoring', 32768),
        ('unknown_error_class', result['error'], 'HTTPError', 'generation', 32768),
        ('server_500', result['error'].replace('Error code: 400', 'Error code: 500'), 'BadRequestError', 'generation', 32768),
        ('unrelated_400', 'Error code: 400 - invalid role', 'BadRequestError', 'generation', 32768),
        ('oversized_requested_completion', 'Error code: 400 - requested 40000 tokens with 28000 prompt tokens', 'BadRequestError', 'generation', 32768),
    ]:
        check('reject_' + name, evaluator.context_budget_details(error, cls, phase, limit) is None)
    changed = copy.deepcopy(result)
    check('cached_budget_reclassified', evaluator.reclassify_cached_context_budget(changed, 32768))
    check('zero_score_and_raw_error_preserved', changed['score'] == result['score'] == 0.0 and
          changed['error'] == result['error'] and changed['error_class'] == result['error_class'])
    changed = dict(result, score=1.0)
    check('never_turn_a_positive_score_into_budget_zero', not evaluator.reclassify_cached_context_budget(changed, 32768))
    changed = dict(result, failure_type='model_budget')
    check('existing_budget_result_not_reclassified_twice', not evaluator.reclassify_cached_context_budget(changed, 32768))

    old_inv = json.loads((result_path.parent / 'invocation.json').read_text())
    new_inv = copy.deepcopy(old_inv)
    key = str(current.resolve())
    check('fixture_source_is_exact_old_wrapper', old_inv['source_sha256'][key] == old_sha)
    new_inv['source_sha256'][key] = new_sha
    transitions = [dict(from_wrapper_sha256=old_sha, to_wrapper_sha256=new_sha)]
    check('approved_wrapper_only_adoption', evaluator.compatible_cache_invocations(old_inv, new_inv, transitions))
    for name, mutation in [
        ('data', lambda x: x.update(data_sha256='different-data')),
        ('temperature', lambda x: x['config'].update(temperature=0.123)),
        ('judge', lambda x: x['judge'].update(model_id='another-judge')),
        ('policy_weights', lambda x: x['policy_weight_identity'].update(index_sha256='other-weights')),
        ('engine_source', lambda x: x['source_sha256'].update({next(k for k in x['source_sha256'] if k != key): 'other-code'})),
    ]:
        bad = copy.deepcopy(new_inv)
        mutation(bad)
        check('reject_cache_change_' + name, not evaluator.compatible_cache_invocations(old_inv, bad, transitions))
    check('reject_unapproved_transition', not evaluator.compatible_cache_invocations(old_inv, new_inv, []))
    with tempfile.TemporaryDirectory(prefix='pyvision-budget-cache-') as directory:
        root = Path(directory)
        (root / 'references').mkdir()
        (root / 'references/evaluation-cache-compatibility.json').write_text(json.dumps(dict(approved_transitions=transitions)))
        output = root / 'output'
        output.mkdir()
        (output / 'invocation.json').write_text(json.dumps(old_inv))
        actual_root = evaluator.ROOT
        try:
            evaluator.ROOT = root
            old_hash = evaluator.invocation_fingerprint(old_inv)
            check('adoption_accepts_original_cache', evaluator.prepare_cache_adoptions(output, new_inv) == {old_hash})
            check('original_invocation_preserved', json.loads((output / 'invocation-history' / (old_hash + '.json')).read_text()) == old_inv)
            (output / 'invocation.json').write_text(json.dumps(new_inv))
            check('partial_migration_can_resume', evaluator.prepare_cache_adoptions(output, new_inv) == {old_hash})
            bad = copy.deepcopy(new_inv)
            bad['config']['temperature'] = 0.123
            try:
                evaluator.prepare_cache_adoptions(output, bad)
            except RuntimeError:
                rejected = True
            else:
                rejected = False
            check('incompatible_cache_rejected_before_overwrite', rejected and
                  json.loads((output / 'invocation.json').read_text()) == new_inv)
        finally:
            evaluator.ROOT = actual_root

    before, after = ast.parse(original.read_text()), ast.parse(current.read_text())
    def function(tree, name):
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    check('generation_budget_and_calls_unchanged', ast.dump(function(before, 'bounded_call')) == ast.dump(function(after, 'bounded_call')))
    check('score_aggregation_unchanged', ast.dump(function(before, 'summarize')) == ast.dump(function(after, 'summarize')))
    old_try = next(n for n in function(before, 'run').body if isinstance(n, ast.Try))
    new_try = next(n for n in function(after, 'run').body if isinstance(n, ast.Try))
    check('generation_and_grading_body_unchanged', ast.dump(ast.Module(body=old_try.body, type_ignores=[])) ==
          ast.dump(ast.Module(body=new_try.body, type_ignores=[])))
    def configuration(tree):
        return next(n for n in function(tree, 'main').body if isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == 'config' for t in n.targets))
    check('decoding_and_prompt_configuration_unchanged', ast.dump(configuration(before)) == ast.dump(configuration(after)))
    check('frozen_original_fixture_untouched', result_path.read_bytes() == original_result)
    report = dict(status='passed', checks=len(checked), check_names=checked,
                  old_wrapper_sha256=old_sha, new_wrapper_sha256=new_sha,
                  actual_error_result=str(result_path), actual_error_record_sha256=hashlib.sha256(original_result).hexdigest(),
                  actual_decoder_prompt_tokens=33130, context_limit=32768,
                  model_inference_performed=False, live_results_modified=False)
    (ROOT / 'manifests/context-budget-classification-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
