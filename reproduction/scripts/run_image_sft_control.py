"""Use idle GPUs for official SFT controls while RL trains."""
import argparse
import fcntl
import json
import time
import urllib.request
import sys

from download_preparation_queue import ROOT, write
from pipeline import completed_training, evaluate_suite, process_alive, run
from run_training import assert_idle


def main(args):
    path = ROOT / 'manifests' / (args.kind + '-sft-control-status.json')
    benchmarks = (['vstar', 'hrbench-4k', 'hrbench-8k', 'dynamath', 'mathverse', 'mathvision', 'wemath', 'tirbench',
                   'mathvista', 'mmmu', 'charxiv-reasoning', 'charxiv-descriptive'] if args.kind == 'image' else
                  ['vsibench', 'tempcompass-mc', 'tempcompass-yn', 'tempcompass-caption-matching', 'mvbench-public-19'])
    try:
        if args.kind == 'video':
            write(path, dict(status='running', stage='waiting_for_video_sft_assets', gpu=args.gpu,
                             port=args.port, benchmarks=benchmarks))
            receipt = ROOT / 'manifests/video-sft-model-prepared-assets.json'
            while not (receipt.exists() and json.loads(receipt.read_text()).get('status') == 'complete'):
                if not process_alive(ROOT / 'manifests/preparation-queue-process.json'):
                    raise RuntimeError('Video SFT asset downloader stopped before completion')
                time.sleep(20)
        write(path, dict(status='running', stage='waiting_for_judge', gpu=args.gpu, port=args.port, benchmarks=benchmarks))
        while True:
            try:
                with urllib.request.urlopen('http://127.0.0.1:18961/v1/models', timeout=5) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if not process_alive(ROOT / 'manifests/pipeline-process.json'):
                raise RuntimeError('Primary judge owner stopped before the control evaluation')
            time.sleep(20)
        if args.preflight_training:
            if args.kind != 'video':
                raise ValueError('The early concurrent preflight is intended for Video RL')
            smoke_name = 'video-engineering-smoke-gpu' + args.gpu
            write(path, dict(status='running', stage='running_video_rl_engineering_smoke',
                             gpu=args.gpu, model_run=smoke_name, paper_training_completed=False))
            if not completed_training(smoke_name, 1):
                run([sys.executable, str(ROOT / 'scripts/run_training.py'), '--kind', 'video', '--mode', 'smoke',
                     '--gpus', args.gpu, '--name', smoke_name,
                     '--optimizer-foreach', 'false',
                     '--ray-temp-dir', '/tmp/pyvision-rl-ray-video-engineering'], 'train-video-early-smoke.log')
            if not completed_training(smoke_name, 1):
                raise RuntimeError('Early Video RL smoke did not save its checkpoint')
            weight_report = ROOT / 'outputs' / smoke_name / 'optimizer-update-validation.json'
            nonzero_verified = (weight_report.exists() and
                                json.loads(weight_report.read_text()).get('nonzero_weight_update_verified') is True)
            write(ROOT / 'manifests/video-early-training-smoke.json', dict(
                status='passed', run=smoke_name, gpu=args.gpu, engineering_preflight_only=True,
                formal_700_step_training_complete=False,
                first_optimizer_call_learning_rate=0.0,
                nonzero_weight_update_verified=nonzero_verified,
                note='Published scheduler starts at zero LR; a separate weight report is required for nonzero-update proof'))
            while True:
                try:
                    assert_idle([args.gpu])
                    break
                except RuntimeError:
                    time.sleep(10)
        write(path, dict(status='running', stage='evaluating_official_' + args.kind + '_sft', gpu=args.gpu, port=args.port,
                         benchmarks=benchmarks, paper_numbers_reproduced=False))
        groups = [benchmarks[:4], benchmarks[4:]] if args.kind == 'video' else [benchmarks]
        for group in groups:
            evaluate_suite(ROOT / 'models' / f'PyVision-{args.kind.capitalize()}-7B-SFT', group,
                           f'official-{args.kind}-sft-control', gpu=args.gpu, port=args.port, update_state=False)
        write(path, dict(status='complete', stage=args.kind + '_sft_control_complete', benchmarks=benchmarks))
    except BaseException as exc:
        write(path, dict(status='failed', error=str(exc), benchmarks=benchmarks))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['image', 'video'], default='image')
    parser.add_argument('--gpu', default='7')
    parser.add_argument('--port', type=int, default=18963)
    parser.add_argument('--preflight-training', action='store_true', help='Run an early Video RL update before inference controls')
    args = parser.parse_args()
    with (ROOT / (args.kind + '-sft-control.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args)
