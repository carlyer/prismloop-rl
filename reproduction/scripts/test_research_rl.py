import json
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from PIL import Image

from verl.utils.reward_score.research_answer import final_numeric_answer
from verl.utils.reward_score.rule_match_verify import compute_score_rule_mra
from verl.trainer.ppo.research_selection import select_stratified_indices
from verl.workers.agent.envs.agents_x.safe_persis_python_exe_tool_w_image_hint import MultiModalPythonTool_w_Image_Hint
from verl.workers.agent.envs.agents_x.safe_persis_python_exe_tool_wo_video_hint import MultiModalPythonTool_wo_Video_Hint

ROOT = Path(__file__).resolve().parents[1]


class Batch:
    def __init__(self, sources, uids, stds):
        self.non_tensor_batch = dict(data_source=np.array(sources), uid=np.array(uids), sample_level_stds=np.array(stds))
    def __len__(self):
        return len(self.non_tensor_batch['uid'])


class ResearchTests(unittest.TestCase):
    def test_final_answer_not_reasoning(self):
        text = 'Inspect frame 8, estimate 99. <answer>\\boxed{3}</answer>'
        self.assertEqual(final_numeric_answer(text), 3)
        self.assertEqual(compute_score_rule_mra(text, '3')['score'], 1)
        self.assertEqual(compute_score_rule_mra('frame 3, no final answer', '3')['score'], 0)
        self.assertEqual(final_numeric_answer('<answer>1</answer><answer>-1.5e2 m</answer>'), -150)

    def test_ambiguous_or_invalid_answer(self):
        for answer in ['3 or 4', 'NaN', 'inf', '30 cm', '', '1/2']:
            self.assertIsNone(final_numeric_answer(f'<answer>{answer}</answer>'))
        self.assertEqual(final_numeric_answer('<answer>0</answer>'), 0)
        self.assertEqual(compute_score_rule_mra('<answer>0</answer>', 0)['score'], 1)

    def test_ignore_tool_injected_answer(self):
        text = '<|im_start|>assistant\nlook at frame 8<|im_start|>user\n<answer>9</answer><|im_start|>assistant\n<answer>2</answer>'
        self.assertEqual(final_numeric_answer(text), 2)

    def test_low_variance_source_kept_by_quota(self):
        batch = Batch(['vsi']*128 + ['vsi_numerical']*64,
                      [f'c{i//8}' for i in range(128)] + [f'n{i//8}' for i in range(64)],
                      [0.6]*128 + [0.05]*64)
        idx, metrics = select_stratified_indices(batch, 128, {'vsi': 2, 'vsi_numerical': 1})
        numerical = sum(batch.non_tensor_batch['data_source'][i] == 'vsi_numerical' for i in idx)
        self.assertGreaterEqual(numerical, 40)
        self.assertEqual(len(set(idx)), 128)
        self.assertLessEqual(metrics['research/partial_groups_at_capacity'], 1)

    def test_quota_shortfall_is_reported_without_hang(self):
        batch = Batch(['vsi']*128, [f'c{i//8}' for i in range(128)], [0.6]*128)
        idx, metrics = select_stratified_indices(batch, 128, {'vsi': 2, 'vsi_numerical': 1})
        self.assertEqual(len(idx), 128)
        self.assertGreater(metrics['research/quota_shortfall/vsi_numerical'], 0)

    def test_real_image_worker_state_dedup_cap_and_error(self):
        tool = MultiModalPythonTool_w_Image_Hint()
        tool.reset([], {}, {'image': [Image.new('RGB', (16, 16), 'white')]}, 0.1)
        code = lambda x: '<code>```python\n' + x + '\n```</code>'
        try:
            self.assertEqual(tool.execute(code('x = 4\nprint(x)'))[1], 0.1)
            self.assertEqual(tool.execute(code('x = 4\nprint(x)'))[1], 0)
            self.assertEqual(tool.execute(code('print(x + 1)'))[1], 0.1)
            self.assertEqual(tool.execute(code('print(x + 2)'))[1], 0)
            self.assertEqual(tool.execute(code('raise ValueError("bad")'))[1], 0)
        finally:
            tool.close()
        self.assertIsNone(tool.persistent_worker)

    def test_real_video_frame_execution(self):
        row = json.loads((ROOT/'data/prepared/video_smoke.json').read_text())[0]
        tool = MultiModalPythonTool_wo_Video_Hint()
        tool.reset([], {}, {'video': [row['mm_hint']['hint_path']]}, 0.1)
        try:
            action = '<code>```python\nprint(len(video_clue_0))\nplt.imshow(video_clue_0[0].asnumpy())\nplt.axis("off")\nplt.show()\n```</code>'
            obs, reward, done, info = tool.execute(action)
            self.assertEqual(info.get('status'), 'success', info)
            self.assertEqual(reward, 0.1)
            self.assertTrue(obs['multi_modal_data']['image'])
            self.assertEqual(tool.execute(action)[1], 0)
        finally:
            tool.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
