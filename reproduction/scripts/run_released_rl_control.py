"""Queue released RL model controls after SFT controls release their GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import urllib.request

from download_preparation_queue import ROOT, write
from pipeline import evaluate_suite, process_alive

IMAGE = ['vstar', 'hrbench-4k', 'hrbench-8k', 'dynamath', 'mathverse', 'mathvision', 'wemath',
         'tirbench', 'mathvista', 'mmmu', 'charxiv-reasoning', 'charxiv-descriptive']
VIDEO = ['vsibench', 'tempcompass-mc', 'tempcompass-yn', 'tempcompass-caption-matching', 'mvbench-public-19']


def wait_for_early_image_control(progress):
    """The early sweep writes the same V* output directory as this full sweep."""
    metadata = ROOT / 'manifests/image-rl-early-control-process.json'
    status = ROOT / 'manifests/image-rl-early-control-status.json'
    if not metadata.exists():
        return
    progress('waiting_for_early_released_image_control_to_release_shared_outputs')
    while process_alive(metadata):
        time.sleep(20)
    record = json.loads(status.read_text()) if status.exists() else {}
    if record.get('status') != 'complete':
        raise RuntimeError('Early Image RL control stopped without completing; inspect its records before reuse')


def main(args):
    status = ROOT / 'manifests' / f'{args.kind}-rl-control-status.json'
    benchmarks = IMAGE if args.kind == 'image' else VIDEO
    fields = dict(status='running', gpu=args.gpu, port=args.port, benchmarks=benchmarks,
                  model_stage='official_released_rl', is_local_training_run=False,
                  paper_numbers_reproduced=False)
    def progress(stage):
        fields.update(stage=stage, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write(status, fields)
        print(stage, flush=True)
    try:
        progress('waiting_for_' + args.kind + '_sft_control_completion')
        sft_status = ROOT / 'manifests' / f'{args.kind}-sft-control-status.json'
        sft_meta = ROOT / 'manifests' / f'{args.kind}-sft-control-process.json'
        while not (sft_status.exists() and json.loads(sft_status.read_text()).get('status') == 'complete'):
            if sft_meta.exists() and not process_alive(sft_meta):
                recovery = ROOT / 'manifests' / f'{args.kind}-sft-context-recovery-process.json'
                recovery_status = ROOT / 'manifests' / f'{args.kind}-sft-context-recovery-status.json'
                if (process_alive(recovery) and recovery_status.exists() and
                        json.loads(recovery_status.read_text()).get('status') == 'running'):
                    time.sleep(20)
                    continue
                raise RuntimeError('SFT control stopped before completing; preserve and repair that control first')
            time.sleep(20)
        if args.kind == 'image':
            wait_for_early_image_control(progress)
        progress('waiting_for_official_' + args.kind + '_rl_checkpoint')
        receipts = [ROOT / 'manifests' / f'{args.kind}-rl-model{suffix}.json'
                    for suffix in ['-prepared-assets', '']]
        while not any(p.exists() and json.loads(p.read_text()).get('status') == 'complete' for p in receipts):
            if not process_alive(ROOT / 'manifests/preparation-queue-process.json'):
                raise RuntimeError('Official reference checkpoint unavailable after downloader stopped')
            time.sleep(20)
        progress('waiting_for_local_judge')
        while True:
            try:
                with urllib.request.urlopen('http://127.0.0.1:18961/v1/models', timeout=5) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(20)
        progress('evaluating_official_' + args.kind + '_rl')
        groups = [benchmarks[:4], benchmarks[4:]] if args.kind == 'video' else [benchmarks]
        model = ROOT / 'models' / f'PyVision-{args.kind.capitalize()}-7B-RL'
        for group in groups:
            evaluate_suite(model, group, f'official-{args.kind}-rl-control',
                           gpu=args.gpu, port=args.port, update_state=False)
        fields.update(status='complete')
        progress('official_' + args.kind + '_rl_control_complete')
    except BaseException as exc:
        fields.update(status='failed', error_class=type(exc).__name__, error=str(exc))
        write(status, fields)
        raise


if __name__ == '__main__':
    for variable in ['NO_PROXY', 'no_proxy']:
        os.environ[variable] = '127.0.0.1,localhost'
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['image', 'video'], required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    with (ROOT / f'{args.kind}-rl-control.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Process identity prevents PID reuse after long runs.
        ticks = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write(ROOT / 'manifests' / f'{args.kind}-rl-control-process.json', dict(
            pid=os.getpid(), start_time_ticks=ticks,
            started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        main(args)
