"""Resume our suspended control evaluation after the replacement judge is healthy."""
import json
import os
import signal
import time
import urllib.request

from download_preparation_queue import ROOT, write


def main():
    path = ROOT / 'manifests/control-evaluation-suspension.json'
    record = json.loads(path.read_text())
    pgid = record['pgid']
    if record['status'] != 'temporarily_suspended':
        return
    try:
        for _ in range(30):
            state = json.loads((ROOT / 'status.json').read_text())
            if state.get('stage') in ['reusing_completed_image_smoke', 'waiting_for_smoke_workers_to_release_gpus', 'running_700_step_image_rl']:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18961/v1/models', timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
            time.sleep(10)
        else:
            record['note'] = 'Judge startup deadline exceeded; resume evaluation to preserve visible errors instead of leaving it suspended'
    finally:
        if os.getpgid(pgid) != pgid:
            raise RuntimeError('Recorded owned evaluation group changed')
        os.killpg(pgid, signal.SIGCONT)
        record.update(status='resumed', resumed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write(path, record)
        print('Resumed owned Image SFT evaluation', pgid, flush=True)


if __name__ == '__main__':
    main()
