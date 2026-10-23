"""Exercise real native/patched scorers without loading models or reward services."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import random

import patch_numeric_reward_guard as patch


def load(source):
    namespace = {}
    exec(compile(source, '<isolated_native_numeric_scorer>', 'exec'), namespace)
    return namespace


def main():
    current = patch.TARGET.read_text()
    original = patch.BACKUP.read_text() if patch.REPLACEMENT in current else current
    assert original.count(patch.ANCHOR) == 1
    changed = original.replace(patch.ANCHOR, patch.REPLACEMENT)
    before, after = load(original), load(changed)
    def other_functions(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef)
                     or node.name != 'compute_score_rule_mra']
        return ast.dump(tree)
    assert other_functions(original) == other_functions(changed)
    training_bytes = (patch.ROOT / 'data/PyVision-Video-RL-Data/pyvision_video_rl_data.json').read_bytes()
    rows = json.loads(training_bytes)
    numeric = [row for row in rows if row['data_source'] == 'vsi_numerical']
    valid_gt = [row['reward_model']['ground_truth'] for row in numeric]
    assert len(valid_gt) == 5267
    assert all((value := before['_parse_number'](gt)) is not None and math.isfinite(value)
               for gt in valid_gt)
    cases = [('0', '0'), ('1', '0'), ('4', '4'), ('5', '10'), ('5.0001', '10'),
             ('9.5', '10'), ('10.5', '10'), ('-3', '2'),
             ('<think>inspect frame 10</think><answer>4</answer>', '4')]
    rng = random.Random(42)
    cases += [(str(rng.uniform(-100, 1000)), rng.choice(valid_gt)) for _ in range(2000)]
    cases += [(str(gt), gt) for gt in valid_gt]
    for prediction, truth in cases:
        assert before['compute_score_rule_mra'](prediction, truth) == after['compute_score_rule_mra'](prediction, truth)
    malformed = []
    for prediction in ['', '<answer>unknown</answer>', '<answer></answer>', 'I cannot determine it']:
        try:
            before['compute_score_rule_mra'](prediction, '4')
        except TypeError as error:
            old = dict(type=type(error).__name__, message=str(error))
        else:
            raise AssertionError('Expected native malformed-output exception')
        result = after['compute_score_rule_mra'](prediction, '4')
        assert result == dict(score=0.0, is_answer_right=False)
        malformed.append(dict(prediction=prediction, native_exception=old, guarded_result=result))
    for bad_gt in ['', 'unknown', float('inf'), float('nan')]:
        try:
            after['compute_score_rule_mra']('4', bad_gt)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid GT must raise, not become a scored model error')
    first_number = after['compute_score_rule_mra'](
        '<think>inspect frame 10</think><answer>4</answer>', '4')
    assert first_number == dict(score=0.0, is_answer_right=False)
    report = dict(status='passed', observed_utc=datetime.now(timezone.utc).isoformat(),
                  original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  guarded_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                  valid_prediction_cases_equivalent=len(cases),
                  all_numerical_training_ground_truths_validated=len(valid_gt),
                  training_data_sha256=hashlib.sha256(training_bytes).hexdigest(),
                  malformed_prediction_cases=malformed, invalid_ground_truth_cases_rejected=4,
                  all_other_source_AST_unchanged=True,
                  native_first_number_semantics_preserved=dict(
                      solution='<think>inspect frame 10</think><answer>4</answer>', truth='4',
                      result=first_number, note='The released scorer parses the first number of the full response, '
                                               'not the final answer tag. Preserved for fidelity; distinct from final Bench grading.'),
                  scope='CPU native scorer calls only; no training, model inference or judge API calls.')
    (patch.ROOT / 'manifests/numeric-reward-guard-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='passed', valid_cases=len(cases), malformed_cases=len(malformed),
                         validated_training_numeric_gt=len(valid_gt))))


if __name__ == '__main__':
    main()
