"""Reload the corrected frame adapter after the active ZIP extraction finishes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from download_preparation_queue import ROOT, write


def owns_live_process(pid, script):
    try:
        if Path(f'/proc/{pid}/stat').read_text().split()[2] == 'Z':
            return False
        return script in Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparer-pid', type=int, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    preparer = str(ROOT / 'scripts/prepare_extra_benchmarks.py')
    parent = str(ROOT / 'scripts/expanded_evaluation_pipeline.py')
    status = ROOT / 'manifests/mvbench-adapter-reload.json'
    record = dict(status='running', stage='waiting_for_current_zip_extraction',
                  preparer_pid=args.preparer_pid, parent_pid=args.parent_pid,
                  reason='Match official TVQA first_idx=1; retain completed archive extraction',
                  validation=str(ROOT / 'manifests/mvbench-frame-index-validation.json'))
    try:
        receipt = json.loads((ROOT / 'manifests/mvbench-prepared-assets.json').read_text())
        archives = [ROOT / 'data/benchmarks/mvbench' / f['path'] for f in receipt['selected_files']
                    if f['path'].endswith('.zip')]
        if not archives:
            raise RuntimeError('Missing pinned archive inventory')
        while True:
            complete = sum(p.with_suffix(p.suffix + '.extracted.json').exists() for p in archives)
            record.update(completed_archives=complete, total_archives=len(archives))
            write(status, record)
            if complete == len(archives):
                break
            if not owns_live_process(args.preparer_pid, preparer):
                raise RuntimeError('Original preparation stopped before the extraction boundary')
            time.sleep(20)
        if owns_live_process(args.preparer_pid, preparer):
            os.kill(args.preparer_pid, signal.SIGTERM)
        while owns_live_process(args.preparer_pid, preparer):
            time.sleep(5)
        if owns_live_process(args.parent_pid, parent):
            os.kill(args.parent_pid, signal.SIGTERM)
        while owns_live_process(args.parent_pid, parent):
            time.sleep(5)
        marker = ROOT / 'manifests/benchmark-mvbench-public-19.json'
        if marker.exists():
            old = json.loads(marker.read_text())
            write(ROOT / 'manifests/mvbench-before-frame-index-repair.json', old)
            old['status'] = 'requires_frame_index_repair'
            write(marker, old)
        with (ROOT / 'logs/expanded-evaluation-pipeline.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', parent], stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        write(ROOT / 'manifests/expanded-evaluation-process.json', dict(
            pid=child.pid, restarted_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            reason=record['reason']))
        record.update(status='reloaded', stage='corrected_adapter_started', new_parent_pid=child.pid,
                      full_benchmark_preparation_completed=False)
    except Exception as exc:
        record.update(status='failed', error_class=type(exc).__name__, error=str(exc))
        raise
    finally:
        record['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write(status, record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
