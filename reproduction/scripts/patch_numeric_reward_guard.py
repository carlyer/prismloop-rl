"""Keep native numeric MRA semantics; reject invalid GT and score nonnumeric output zero."""
from download_preparation_queue import ROOT, digest, write

TARGET = ROOT.parent / 'verl_agents/verl/utils/reward_score/rule_match_verify.py'
BACKUP = ROOT / 'references/rule_match_verify.original.py'
ANCHOR = '''    gt_number = _parse_number(ground_truth)

    acc_reward = _compute_mra(answer_number, gt_number)
'''
REPLACEMENT = '''    gt_number = _parse_number(ground_truth)

    # Local reproduction guard: malformed model output must not abort a batch.
    # Preserve the native first-number parser and exact valid-number MRA formula.
    import math
    if gt_number is None or not math.isfinite(gt_number):
        raise ValueError("Numerical ground truth must contain a finite number")
    if answer_number is None or not math.isfinite(answer_number):
        return {"score": 0.0, "is_answer_right": False}

    acc_reward = _compute_mra(answer_number, gt_number)
'''


def main():
    source = TARGET.read_text()
    if REPLACEMENT not in source:
        if source.count(ANCHOR) != 1:
            raise RuntimeError('Native numerical reward source changed')
        if not BACKUP.exists():
            BACKUP.write_text(source)
        elif BACKUP.read_text() != source:
            raise RuntimeError('Original numeric reward backup differs')
        TARGET.write_text(source.replace(ANCHOR, REPLACEMENT))
    write(ROOT / 'manifests/numeric-reward-guard-patch.json', dict(
        target=str(TARGET), original=str(BACKUP), original_sha256=digest(BACKUP),
        patched_sha256=digest(TARGET), reward_function_modified=True,
        valid_numeric_MRA_formula_modified=False, native_first_number_parser_modified=False,
        reward_change_scope='Unparseable prediction against finite nonzero GT previously raised '
                           'TypeError; now it gets score=0 and is_answer_right=False. '
                           'Nonfinite prediction retains a finite zero reward. Invalid GT still raises.',
        affected_sources=['vsi_numerical', '*_rule_verifiable'],
        existing_image_training_or_evaluators_restarted=False,
        validation='manifests/numeric-reward-guard-validation.json'))
    print('Applied native numerical reward output guard')


if __name__ == '__main__':
    main()
