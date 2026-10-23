"""Run published RL controls on an idle GPU while the local baseline trains."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from download_preparation_queue import ROOT, write
from pipeline import evaluate_suite, process_alive


if __name__ == '__main__':
    for variable in ['NO_PROXY', 'no_proxy']:
        os.environ[variable] = '127.0.0.1,localhost'
    with (ROOT / 'early-released-image-control.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ticks = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write(ROOT / 'manifests/image-rl-early-control-process.json', dict(
            pid=os.getpid(), start_time_ticks=ticks,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        fields = dict(status='running', stage='waiting_for_sft_services_to_finish_loading',
                      gpu='3', port=18965, benchmarks=['vstar', 'mathvision-mini'],
                      model_stage='official_released_rl', is_local_training_run=False,
                      full_paper_suite_complete=False)
        status = ROOT / 'manifests/image-rl-early-control-status.json'
        try:
            if not json.loads((ROOT / 'status.json').read_text()).get('two_step_weight_update_verified'):
                raise RuntimeError('Wait for verified Image training recovery first')
            write(status, fields)
            while True:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18964/v1/models', timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if not process_alive(ROOT / 'manifests/video-sft-control-process.json'):
                    raise RuntimeError('Video SFT owner stopped before its service became ready')
                time.sleep(10)
            fields.update(stage='evaluating_published_image_rl_selected_benchmarks')
            write(status, fields)
            evaluate_suite(ROOT / 'models/PyVision-Image-7B-RL', fields['benchmarks'],
                           'official-image-rl-control', gpu='3', port=18965, update_state=False)
            fields.update(status='complete', stage='selected_control_benchmarks_complete')
        except BaseException as exc:
            fields.update(status='failed', error=str(exc))
            raise
        finally:
            fields['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            write(status, fields)
