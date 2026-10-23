"""Do not confuse unfinished, rounded, mismatched or earlier validation with a new result."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

import collect_training_validation as collector


def fixture(step=20, value=.6910994764397905, batches=2):
    return ('Size of val dataloader: 2\n' +
            f'Dumped generations to /offline/run/rollouts/{step}.json\n' +
            "test_gen_batch meta info: {'validate': True}\n"*batches +
            collector.TAG+repr({'val/vstar': value})+'\n' +
            f'step:{step} - actor/grad_norm:0.300 - val/vstar:{value:.3f} - x:1.000\n')


class ValidationTests(unittest.TestCase):
    def test_native_precision(self):
        row = collector.parse(fixture())[0]
        self.assertEqual(row['metrics']['val/vstar'], .6910994764397905)
        self.assertEqual(row['generation_batches'], 2)

    def test_in_progress_not_completed(self):
        self.assertEqual(collector.parse(fixture().rsplit('step:20', 1)[0]), [])

    def test_earlier_result_not_reused(self):
        text = fixture()+'Dumped generations to /offline/run/rollouts/40.json\n'
        text += "test_gen_batch meta info: {'validate': True}\n"
        self.assertEqual([row['step'] for row in collector.parse(text)], [20])

    def test_multiple_complete_periods(self):
        self.assertEqual([row['step'] for row in collector.parse(fixture()+fixture(40, .75))], [20, 40])

    def test_missing_dispatch_rejected(self):
        with self.assertRaises(ValueError):
            collector.parse(fixture(batches=1))

    def test_wrong_rollout_rejected(self):
        with self.assertRaises(ValueError):
            collector.parse(fixture().replace('rollouts/20.json', 'rollouts/19.json'))

    def test_mismatched_summary_rejected(self):
        with self.assertRaises(ValueError):
            collector.parse(fixture().replace('val/vstar:0.691', 'val/vstar:0.750'))

    def test_rounded_only_rejected(self):
        with self.assertRaises(ValueError):
            collector.parse(''.join(line for line in fixture().splitlines(keepends=True)
                                   if not line.startswith(collector.TAG)))

    def test_duplicate_dict_rejected(self):
        with self.assertRaises(ValueError):
            collector.parse(fixture().replace('step:20', collector.TAG+"{'val/vstar':0.691}\nstep:20"))

    def test_archive_continuity(self):
        previous = dict(completed_training_updates_observed=39,
                        validations=[{'step':20}], validation_input_sha256='same')
        collector.verify_continuity(previous, 40, {20,40}, 'same')
        for current, steps, input_hash in [(38,{20},'same'), (40,{40},'same'), (40,{20,40},'changed')]:
            with self.assertRaises(ValueError):
                collector.verify_continuity(previous, current, steps, input_hash)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ValidationTests))
    report = dict(status='passed' if result.wasSuccessful() else 'failed',
                  observed_utc=datetime.now(timezone.utc).isoformat(), tests=result.testsRun,
                  errors=len(result.errors), failures=len(result.failures),
                  collector_source_sha256=collector.digest(Path(collector.__file__).read_bytes()),
                  scope='CPU parser integrity cases only; no model, training, GPU, or judge calls.')
    (collector.ROOT / 'manifests/training-validation-archive-checks.json').write_text(
        json.dumps(report, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
