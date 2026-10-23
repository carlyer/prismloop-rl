"""Download explicitly authorized extra benchmarks without disturbing training."""
import fcntl
import json
import os
import subprocess
import sys
import time

from download_preparation_queue import ROOT, fetch, write


def main():
    for name in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY']:
        if name in os.environ:
            os.environ[name.lower()] = os.environ[name]
    plan = json.loads((ROOT / 'extra-benchmarks-plan.json').read_text())
    state = ROOT / 'manifests/extra-benchmarks-status.json'
    done = []
    try:
        for task in plan['download_queue']:
            write(state, {'status': 'running', 'stage': task['name'], 'downloaded': done,
                          'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
            receipt = ROOT / 'manifests' / (task['name'] + '-prepared-assets.json')
            source = json.loads((ROOT / task['source_manifest']).read_text())
            previous = json.loads(receipt.read_text()) if receipt.exists() else {}
            if previous.get('status') != 'complete' or previous.get('revision') != source['revision']:
                fetch(task)
            done.append(task['name'])
        write(state, {'status': 'downloaded', 'stage': 'downloads_complete_adapters_pending',
                      'downloaded': done, 'evaluated': False})
    except BaseException as exc:
        write(state, {'status': 'failed', 'error': str(exc), 'downloaded': done})
        raise


if __name__ == '__main__':
    with (ROOT / 'extra-benchmarks.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
