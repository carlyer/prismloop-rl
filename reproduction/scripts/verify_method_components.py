"""Check released reward gating, advantages and filters against small reference cases."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'verl_agents'))
os.environ['LLM_AS_A_JUDGE_CONFIG_PATH'] = str(ROOT / 'judge-config.json')


def main():
    import numpy as np
    import torch
    from verl import DataProto
    from verl.workers.reward_manager.naive import NaiveRewardManager
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.trainer.ppo.filter_fn_utils import dynamic_sampling_fn, end_reason_filtering_fn
    from verl.workers.agent.parallel_env import EndReasonEnum
    report = {'test_type': 'deterministic component checks; no model inference or training'}
    data = DataProto.from_dict({
        'prompts': torch.ones((3, 1), dtype=torch.long), 'responses': torch.ones((3, 2), dtype=torch.long),
        'attention_mask': torch.ones((3, 3), dtype=torch.long),
        'env_reward': torch.tensor([[0.1, 0.1], [0.1, 0.1], [0.0, 0.0]])}, non_tensors={
        'reward_model': np.array([{'ground_truth': '3'}] * 3, dtype=object),
        'data_source': np.array(['wemath-pro'] * 3, dtype=object),
        'extra_info': np.array([{'correct': True}, {'correct': False}, {'correct': True}], dtype=object)})
    def reference_correctness(**kwargs):
        correct = kwargs['extra_info']['correct']
        return {'score': float(correct), 'is_answer_right': correct}
    manager = NaiveRewardManager(SimpleNamespace(decode=lambda tokens: '3'), 0, reference_correctness)
    rewards = manager(data).sum(-1)
    torch.testing.assert_close(rewards, torch.tensor([1.2, 0.0, 1.0]))
    report['reward_correctness_gate'] = rewards.tolist()
    advantages, _, _ = compute_grpo_outcome_advantage(
        torch.tensor([[0.0], [1.1], [1.3]]), torch.ones((3, 2)), np.array(['q'] * 3),
        norm_adv_by_std_in_grpo=False)
    torch.testing.assert_close(advantages[:, 0], torch.tensor([-0.8, 0.3, 0.5]))
    report['unnormalized_grpo_advantages'] = advantages[:, 0].tolist()
    groups = DataProto.from_dict({'token_level_scores': torch.tensor([[1.1], [1.1], [0.0], [0.0], [0.0], [1.2]])},
        non_tensors={'uid': np.array(['constant_correct'] * 2 + ['all_wrong'] * 2 + ['mixed'] * 2, dtype=object)})
    kept, _ = dynamic_sampling_fn(groups)
    assert kept.non_tensor_batch['uid'].tolist() == ['mixed', 'mixed']
    report['zero_variance_group_filter'] = 'Both constant groups removed; mixed-reward group kept'
    reasons = [EndReasonEnum.DONE, EndReasonEnum.ERROR_IN_ACTION, EndReasonEnum.OVER_LENGTH, EndReasonEnum.EXCEED_MAX_TURNS]
    batch = DataProto.from_dict({'index': torch.arange(4)}, non_tensors={'end_reason': np.array(reasons, dtype=object)})
    kept, _ = end_reason_filtering_fn(batch, {'end_reason_filter_reserve_names': ['DONE', 'ERROR_IN_ACTION', 'EXCEED_MAX_TURNS']})
    assert kept.batch['index'].tolist() == [0, 1, 3]
    report['end_reason_filter'] = [x.name for x in kept.non_tensor_batch['end_reason']]
    report['status'] = 'passed'
    (ROOT / 'manifests/method-components-preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
