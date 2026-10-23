import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_training import build, VARIANT_FEATURES
from verl.utils.reward_score.rule_match_verify import compute_score_rule_mra
from verl.workers.agent.envs.agents_x.research_tool_quality import quality_reward


class AblationTests(unittest.TestCase):
    def test_original_numeric_matches_preserved_runtime(self):
        path=Path('/tmp/pyvision-rl-runtime-20261019/PyVision-RL/verl_agents/verl/utils/reward_score/rule_match_verify.py')
        spec=importlib.util.spec_from_file_location('preserved_original_numeric',path)
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        for text,gt in [('frame 8 <answer>3</answer>','3'),('<answer>1.21</answer>','1'),('no numbers','4'),('0',0)]:
            with patch.dict(os.environ,PYVISION_ABLATION_FEATURES=''):
                self.assertEqual(compute_score_rule_mra(text,gt),original.compute_score_rule_mra(text,gt))

    def test_numeric_switch_only_changes_final_answer_input(self):
        text='frame 8 <answer>3</answer>'
        with patch.dict(os.environ,PYVISION_ABLATION_FEATURES='numeric'):
            self.assertEqual(compute_score_rule_mra(text,'3')['score'],1)
        with patch.dict(os.environ,PYVISION_ABLATION_FEATURES='quota,tool'):
            self.assertEqual(compute_score_rule_mra(text,'3')['score'],0)

    def test_original_tool_reward_preserved_including_error(self):
        @quality_reward
        def execute(tool,action,**kwargs):
            return 'error feedback',0.1,False,{'error':'execution failed'}
        with patch.dict(os.environ,PYVISION_ABLATION_FEATURES=''):
            self.assertEqual(execute(object(),'code')[1],0.1)
        with patch.dict(os.environ,PYVISION_ABLATION_FEATURES='tool'):
            self.assertEqual(execute(SimpleNamespace(),'code')[1],0)

    def test_factorial_design_is_complete(self):
        names=['original','numeric_only','quota_only','tool_only','without_numeric','without_quota','without_tool','full']
        self.assertEqual(len({tuple(sorted(VARIANT_FEATURES[n])) for n in names}),8)

    def test_configs_differ_only_in_algorithm_switches(self):
        def args(variant):
            return SimpleNamespace(mode='full',kind='video',checkpoint=None,validation=None,name='config-test',
                gpus='0,2',max_tool_turns=4,steps=160,no_tool_reward=False,no_std_sort=False,std_normalization=False,
                optimizer_foreach='false',save_freq=5,resume_from='/tmp/pyvision-rl-storage/checkpoints/video-recovery-20261019T1622Z-full/global_step_120',
                eval_only=False,checkpoint_dir=None,research_variant=variant)
        original=build(args('original'))[0];quota=build(args('quota_only'))[0]
        self.assertFalse(any('research_source_weights=' in x for x in original))
        self.assertEqual(set(quota)-set(original),{x for x in quota if 'research_source_weights=' in x})
        self.assertTrue(any('ppo_epochs=1' in x for x in original))
        self.assertTrue(any('concurrent_workers=16' in x for x in original))


if __name__=='__main__':
    unittest.main(verbosity=2)
