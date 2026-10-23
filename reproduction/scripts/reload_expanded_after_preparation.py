"""Load control coordination changes once the active media preparation ends."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from download_preparation_queue import ROOT, write
from pipeline import process_alive


def main():
    metadata = ROOT / 'manifests/expanded-evaluation-process.json'
    original = json.loads(metadata.read_text())
    snapshot = ROOT / 'manifests/expanded-observer-before-control-coordination.json'
    write(snapshot, original)
    result_path = ROOT / 'manifests/expanded-observer-reload.json'
    result = dict(status='running', stage='waiting_for_media_preparation',
                  original_process=original, reason='Wait for early controls before reusing their result folders')
    try:
        while process_alive(snapshot):
            status = json.loads((ROOT / 'manifests/expanded-evaluation-status.json').read_text())
            if status.get('stage') == 'waiting_for_image_video_training_and_main_benchmarks':
                break
            if status.get('status') == 'failed':
                raise RuntimeError('Media preparation failed; preserve and repair the failure before reloading')
            write(result_path, result)
            time.sleep(20)
        else:
            raise RuntimeError('Original observer stopped before its safe reload boundary')
        command = Path(f"/proc/{original['pid']}/cmdline").read_bytes().decode().split('\0')
        if str(ROOT / 'scripts/expanded_evaluation_pipeline.py') not in command:
            raise RuntimeError('Original observer identity changed')
        os.kill(original['pid'], signal.SIGTERM)
        while process_alive(snapshot):
            time.sleep(5)
        with (ROOT / 'logs/expanded-evaluation-pipeline.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/expanded_evaluation_pipeline.py')],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write(metadata, dict(pid=child.pid, start_time_ticks=ticks, reason=result['reason'],
                             restarted_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        result.update(status='reloaded', stage='waiting_with_control_coordination', new_pid=child.pid)
    except Exception as exc:
        result.update(status='failed', error_class=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write(result_path, result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
