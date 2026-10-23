"""Regression checks for misleading benchmark acceptance and duplicate writers."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation_evidence import audit_records
import collect_results
import run_released_rl_control as released


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / 'results'
        self.folder.mkdir()
        self.data = self.root / 'data.json'
        self.data.write_text(json.dumps([{'id': 'a'}, {'id': 'b'}]))
        self.config = dict(benchmark='mathvision-mini', repeat=1, samples=2,
                           full_dataset_samples=2, limit=None, question_ids=None,
                           self_test_only=False,
                           data_sha256=hashlib.sha256(self.data.read_bytes()).hexdigest())
        self.fingerprint = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        for identifier, score in [('a', 1.0), ('b', 0.0)]:
            (self.folder / f'{identifier}-repeat0.json').write_text(json.dumps(dict(
                question_id=identifier, repeat=0, category='math', score=score,
                invocation_sha256=self.fingerprint)))
        self.report = dict(status='complete', full_dataset_evaluated=True, self_test_only=False,
                           real_model_inference_completed=True, infrastructure_failures=0,
                           score_is_provisional=False, samples=2, failures=0, score_mean=0.5,
                           category_macro_mean=0.5, categories={'math': dict(count=2, score_mean=0.5)})

    def audit(self):
        return audit_records(self.folder, self.config, self.report, self.data)

    def change_record(self, **updates):
        path = self.folder / 'b-repeat0.json'
        path.write_text(json.dumps(dict(json.loads(path.read_text()), **updates)))

    def test_full_records_pass_but_a_missing_answer_does_not(self):
        self.assertEqual(self.audit()['records_checked'], 2)
        (self.folder / 'b-repeat0.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete record coverage'):
            self.audit()

    def test_another_models_configuration_cannot_be_mixed_in(self):
        self.change_record(invocation_sha256='different-model-invocation')
        with self.assertRaisesRegex(ValueError, 'different invocation'):
            self.audit()

    def test_infrastructure_zero_cannot_be_accepted_as_a_wrong_answer(self):
        self.change_record(failure_type='infrastructure')
        with self.assertRaisesRegex(ValueError, 'Infrastructure failure'):
            self.audit()

    def test_nonfinite_score_or_inconsistent_summary_is_rejected(self):
        self.report['score_mean'] = 0.9
        with self.assertRaisesRegex(ValueError, 'Summary does not match'):
            self.audit()
        self.change_record(score=float('nan'))
        with self.assertRaisesRegex(ValueError, 'Invalid benchmark score'):
            self.audit()

    def test_vstar_single_sample_cannot_replace_avg32(self):
        self.config['benchmark'] = 'vstar'
        with self.assertRaisesRegex(ValueError, 'avg@32'):
            self.audit()

    def test_published_weights_cannot_be_labeled_as_local_training(self):
        with patch.object(collect_results, 'ROOT', self.root):
            config = {'policy_models': [{'root': str(self.root / 'models/PyVision-Image-7B-RL')}]}
            self.assertEqual(collect_results.model_stage(config), 'official_released_rl')
            storage = self.root / 'external-checkpoints/global_step_700/actor/huggingface'
            storage.mkdir(parents=True)
            output = self.root / 'outputs/local-full'
            output.mkdir(parents=True)
            (output / 'checkpoints').symlink_to(self.root / 'external-checkpoints')
            config['policy_models'][0]['root'] = str(storage)
            self.assertEqual(collect_results.model_stage(config), 'local_trained_rl')

    def test_full_control_waits_for_early_writer_and_rejects_failed_handoff(self):
        manifests = self.root / 'manifests'
        manifests.mkdir()
        (manifests / 'image-rl-early-control-process.json').write_text('{}')
        status = manifests / 'image-rl-early-control-status.json'
        status.write_text('{"status":"complete"}')
        with patch.object(released, 'ROOT', self.root), \
                patch.object(released, 'process_alive', side_effect=[True, False]), \
                patch.object(released.time, 'sleep') as sleep:
            released.wait_for_early_image_control(lambda _: None)
            sleep.assert_called_once()
        status.write_text('{"status":"failed"}')
        with patch.object(released, 'ROOT', self.root), \
                patch.object(released, 'process_alive', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'stopped without completing'):
                released.wait_for_early_image_control(lambda _: None)


if __name__ == '__main__':
    unittest.main()
