import json
from pathlib import Path
import tempfile
import unittest

from research_queue_controller import target_step, completed, build_command, prune_owned_intermediate_checkpoints


class QueueTests(unittest.TestCase):
    def test_budget_is_additional_not_absolute(self):
        self.assertEqual(target_step(120), 160)
        self.assertEqual(target_step(20), 60)
        with self.assertRaises(ValueError):
            target_step(120, 700)

    def test_adopted_twenty_step_segment_does_not_finish_method(self):
        job = dict(initial_global_step=120, latest_completed_global_step=140)
        self.assertEqual(completed(job), 20)
        self.assertLess(completed(job), 40)

    def test_continuation_and_ppo2_keep_common_target(self):
        with tempfile.TemporaryDirectory() as d:
            receipt = Path(d)/'tests.json'
            receipt.write_text('{"status":"passed"}')
            registry = {v: dict(validated=True, test_receipt=str(receipt), reproduction_root=d,
                               launcher_args=['--research-variant',v]) for v in ['core','ppo2']}
            for method in registry:
                job = dict(method=method,kind='video',run_name='test',initial_global_step=120,
                           additional_global_steps=40,id='video-'+method)
                command = build_command(job, registry, '/frozen/global_step_120')
                self.assertEqual(command[command.index('--steps')+1], '160')
                self.assertEqual(command[command.index('--resume-from')+1], '/frozen/global_step_120')

    def test_unimplemented_method_and_baseline_deletion_rejected(self):
        with self.assertRaises(ValueError):
            build_command(dict(method='new'), {'new':dict(validated=False)}, '/checkpoint')
        with self.assertRaises(ValueError):
            prune_owned_intermediate_checkpoints(dict(output_root='/tmp/unowned-baseline',run_name='test'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
