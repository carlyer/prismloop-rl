import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_research_queue as evaluation


class ResultTests(unittest.TestCase):
    def test_training_statistics_exclude_initial_steps(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir();(root/'outputs/run/rollouts').mkdir(parents=True)
            (root/'logs/run.log').write_text('step:120 - actor/pg_loss:0.1 - timing_s/step:900.0\nstep:121 - actor/pg_loss:0.1 - timing_s/step:100.0\n')
            (root/'outputs/run/rollouts/121.json').write_text(json.dumps([{'end_reason':5},{'end_reason':0}]))
            with patch.object(evaluation,'ROOT',root):
                value=evaluation.training_metrics(dict(run_name='run',initial_global_step=120,target_global_step=160,output_root=str(root/'outputs/run')))
            self.assertEqual(value['logged_new_steps'],1)
            self.assertEqual(value['logged_step_seconds'],100)
            self.assertEqual(value['selected_training_tool_error_fraction'],0.5)

    def test_protocol_failure_removes_numeric_headline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'manifests').mkdir();folder=root/'model-vsibench';folder.mkdir();(folder/'summary.json').write_text('{}')
            job=dict(id='video-core',kind='video',method='core',status='trained',evaluation_benchmarks=['vsibench'])
            state=dict(methods={'video-core':dict(prefix_path=str(root/'model'))},initial={'video':dict(outputs={'vsibench':'/baseline'})})
            with patch.object(evaluation,'ROOT',root), patch.object(evaluation,'observe',return_value=({'score_percent':70,'native_vsi_percent':60},{})), patch.object(evaluation,'training_metrics',return_value={}), patch.object(evaluation,'compare_protocol',side_effect=ValueError('mismatch')):
                report=evaluation.collect({'jobs':[job]},state)
            row=report['results'][0]
            self.assertEqual(row['evaluation_status'],'failed_audit')
            self.assertNotIn('score_percent',row)

    def test_pending_is_not_zero_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'manifests').mkdir()
            job=dict(id='video-new',kind='video',method='new',status='planned',evaluation_benchmarks=['vsibench'])
            with patch.object(evaluation,'ROOT',root):
                report=evaluation.collect({'jobs':[job]},{'methods':{}})
            self.assertEqual(report['results'][0]['evaluation_status'],'pending')
            self.assertNotIn('score_percent',report['results'][0])


if __name__=='__main__':
    unittest.main(verbosity=2)
