"""Safety checks for freezing checkpoints while rolling retention continues."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collect_results
from evaluate_latest_local_checkpoints import pin_checkpoint


class CheckpointSelectionTests(unittest.TestCase):
    def make_checkpoint(self, root, step, complete=True):
        folder = root / 'checkpoints' / f'global_step_{step}'
        actor = folder / 'actor'
        actor.mkdir(parents=True)
        (folder / 'data.pt').write_bytes(b'dataloader-state')
        for prefix in ['model', 'optim', 'extra_state']:
            for rank in range(2):
                if not complete and prefix == 'optim' and rank == 1:
                    continue
                (actor / f'{prefix}_world_size_2_rank_{rank}.pt').write_bytes(b'checkpoint')
        return folder

    def test_skips_incomplete_checkpoint_and_survives_source_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'image-run'
            complete = self.make_checkpoint(run, 20)
            self.make_checkpoint(run, 40, complete=False)
            (run / 'checkpoints/latest_checkpointed_iteration.txt').write_text('40')
            result = pin_checkpoint(run, root / 'snapshot')
            self.assertEqual(result['step'], 20)
            original = complete / 'actor/model_world_size_2_rank_0.pt'
            frozen = Path(result['pinned_checkpoint']) / 'actor' / original.name
            self.assertEqual(original.stat().st_ino, frozen.stat().st_ino)
            original.unlink()
            self.assertEqual(frozen.read_bytes(), b'checkpoint')

    def test_does_not_select_uncommitted_higher_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'video-run'
            self.make_checkpoint(run, 20)
            self.make_checkpoint(run, 40)
            (run / 'checkpoints/latest_checkpointed_iteration.txt').write_text('20')
            self.assertEqual(pin_checkpoint(run, root / 'snapshot')['step'], 20)

    def test_rejects_initial_step_without_trained_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'image-run'
            self.make_checkpoint(run, 1)
            (run / 'checkpoints/latest_checkpointed_iteration.txt').write_text('1')
            with self.assertRaises(RuntimeError):
                pin_checkpoint(run, root / 'snapshot')

    def test_intermediate_model_is_not_final_700_step_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifests').mkdir()
            model = root / 'export'
            (root / 'manifests/local-checkpoint-evaluation-selection.json').write_text(json.dumps(
                {'checkpoints': {'image': {'model': str(model), 'step': 20}}}))
            with patch.object(collect_results, 'ROOT', root):
                stage = collect_results.model_stage({'policy_models': [{'root': str(model)}]})
            self.assertEqual(stage, 'local_intermediate_rl')


if __name__ == '__main__':
    unittest.main()
