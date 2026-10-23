"""Check explicit checkpoint recovery without loading weights or allocating GPUs."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from run_training import ROOT, build


class ResumeConfigurationTests(unittest.TestCase):
    def setUp(self):
        invocation = ROOT / 'outputs/image-reproduction-20261016T100702Z-full/invocation.json'
        self.options = json.loads(invocation.read_text())['args']
        self.options.update(resume_from=None, save_freq=None, optimizer_foreach='auto',
                            eval_only=False, name='configuration-test-only')

    def build(self, **updates):
        return build(argparse.Namespace(**dict(self.options, **updates)))[0]

    def test_resume_preserves_rl_settings_and_uses_new_output(self):
        checkpoint = ROOT / 'outputs/image-reproduction-20261016T100702Z-full/checkpoints/global_step_1'
        base = self.build()
        resumed = self.build(resume_from=str(checkpoint), save_freq=20)
        old = dict(x.split('=', 1) for x in base[3:])
        new = dict(x.split('=', 1) for x in resumed[3:])
        self.assertEqual(new.pop('trainer.resume_mode'), 'resume_path')
        self.assertEqual(new.pop('trainer.resume_from_path'), str(checkpoint.resolve()))
        self.assertEqual(new.pop('trainer.save_freq'), '20')
        old.pop('trainer.save_freq')
        self.assertEqual(new, old)
        self.assertIn('configuration-test-only', new['trainer.default_local_dir'])
        self.assertEqual(new['trainer.total_training_steps'], '700')

    def test_rejects_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'global_step_1'
            checkpoint.mkdir()
            (checkpoint / 'data.pt').touch()
            with self.assertRaisesRegex(ValueError, 'Incomplete resume checkpoint'):
                self.build(resume_from=str(checkpoint))

    def test_rejects_invalid_modes_and_interval(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            self.build(save_freq=0)
        with self.assertRaisesRegex(ValueError, 'full training only'):
            self.build(mode='smoke', resume_from='unused')


if __name__ == '__main__':
    unittest.main()
