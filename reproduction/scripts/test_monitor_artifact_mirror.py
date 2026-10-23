"""Check bounded backfill without losing or repeatedly rewriting score records."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import monitor_recovery as monitor


class ArtifactMirrorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.outputs = self.root / 'outputs/trial'
        self.outputs.mkdir(parents=True)
        (self.outputs / '1-repeat0.json').write_text('{"score":1}')
        (self.outputs / '1-repeat0-messages.json').write_text('["large media payload"]')
        checkpoint = self.outputs / 'checkpoints'
        checkpoint.mkdir()
        (checkpoint / 'state.json').write_text('{"large_model_metadata":true}')
        self.archive = self.root / 'archive'
        self.archive.mkdir()
        monitor.SEEN_ARTIFACTS.clear()
        monitor.CREATED_ARTIFACT_DIRECTORIES.clear()
        self.patch = patch.object(monitor, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def target(self):
        return self.archive / 'ongoing-local-runtime-results/trial/1-repeat0.json'

    def test_scores_are_preserved_and_media_and_checkpoints_are_excluded(self):
        report = monitor.mirror_small_artifacts(self.archive)
        self.assertEqual(report['files_copied_this_pass'], 1)
        self.assertEqual(self.target().read_text(), '{"score":1}')
        self.assertFalse((self.target().parent / '1-repeat0-messages.json').exists())
        self.assertFalse((self.target().parent / 'checkpoints').exists())

    def test_restart_reuses_identical_persistent_records(self):
        monitor.mirror_small_artifacts(self.archive)
        before = self.target().stat().st_mtime_ns
        monitor.SEEN_ARTIFACTS.clear()
        report = monitor.mirror_small_artifacts(self.archive)
        self.assertEqual(report['files_reused_this_pass'], 1)
        self.assertEqual(report['files_copied_this_pass'], 0)
        self.assertEqual(before, self.target().stat().st_mtime_ns)

    def test_time_budget_yields_and_a_later_pass_finishes(self):
        with patch.object(monitor.time, 'monotonic', side_effect=[0, 31]):
            report = monitor.mirror_small_artifacts(self.archive)
        self.assertEqual(report['status'], 'backfill_pending')
        self.assertFalse(self.target().exists())
        report = monitor.mirror_small_artifacts(self.archive)
        self.assertEqual(report['status'], 'caught_up_for_observed_files')
        self.assertTrue(self.target().exists())

    def test_invalid_in_progress_json_is_retried_after_completion(self):
        source = self.outputs / '1-repeat0.json'
        source.write_text('{"score":')
        monitor.mirror_small_artifacts(self.archive)
        self.assertFalse(self.target().exists())
        source.write_text('{"score":0}')
        monitor.mirror_small_artifacts(self.archive)
        self.assertEqual(self.target().read_text(), '{"score":0}')


if __name__ == '__main__':
    unittest.main()
