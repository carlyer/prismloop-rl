"""Ensure the final pipeline reuses a live Video run rather than duplicating it."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pipeline
from run_training import ROOT, build, resolve_validation


class ParallelVideoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'manifests').mkdir()
        self.name = 'video-test-full'
        (self.root / 'manifests/parallel-video-training-process.json').write_text(json.dumps(
            dict(pid=123, start_time_ticks='456', run_name=self.name)))
        self.root_patch = patch.object(pipeline, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_live_trainer_is_waited_for_until_the_checkpoint_completes(self):
        with patch.object(pipeline, 'completed_training', side_effect=[False, True]), \
                patch.object(pipeline, 'process_alive', return_value=True), \
                patch.object(pipeline, 'state'), patch.object(pipeline.time, 'sleep') as sleep:
            self.assertTrue(pipeline.wait_for_parallel_video(self.name))
            sleep.assert_called_once()

    def test_stopped_incomplete_trainer_requires_repair(self):
        with patch.object(pipeline, 'completed_training', return_value=False), \
                patch.object(pipeline, 'process_alive', return_value=False), \
                patch.object(pipeline, 'state'), patch.object(pipeline.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'prerequisite process stopped'):
                pipeline.wait_for_parallel_video(self.name)
            sleep.assert_not_called()

    def test_completed_trainer_is_reused_after_its_process_exits(self):
        with patch.object(pipeline, 'completed_training', return_value=True), \
                patch.object(pipeline, 'process_alive') as live, patch.object(pipeline, 'state'):
            self.assertTrue(pipeline.wait_for_parallel_video(self.name))
            live.assert_not_called()

    def test_another_run_cannot_be_silently_adopted(self):
        with self.assertRaisesRegex(RuntimeError, 'Another parallel Video run'):
            pipeline.wait_for_parallel_video('different-run-full')

    def test_parallel_gpu_selection_preserves_all_video_rl_overrides(self):
        options = json.loads((ROOT / 'outputs/video-recovery-20261019T1622Z-smoke/invocation.json').read_text())['args']
        options.update(mode='full', steps=None, name='video-configuration-test-full', gpus='1,4',
                       validation=None, checkpoint_dir='/tmp/video-configuration-test-full', save_freq=20)
        reference = argparse.Namespace(**options)
        resolve_validation(reference)
        parallel = argparse.Namespace(**dict(vars(reference), gpus='0,2', ray_temp_dir='/tmp/pv-video-full'))
        expected = build(reference)[0]
        actual = build(parallel)[0]
        self.assertEqual(actual, expected)
        self.assertIn('trainer.total_training_steps=700', actual)
        self.assertIn('actor_rollout_ref.rollout.n=8', actual)
        self.assertIn('trainer.test_freq=20', actual)


if __name__ == '__main__':
    unittest.main()
