"""Prepare extra datasets and evaluate trained/SFT/released-RL checkpoints."""
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request

from download_preparation_queue import ROOT, write
from pipeline import completed_training, evaluate_suite, process_alive, run, state, stop_owned
from run_training import assert_idle

IMAGE_MAIN = ['vstar', 'hrbench-4k', 'hrbench-8k', 'dynamath', 'mathverse', 'mathvision', 'wemath', 'tirbench']
IMAGE_EXTRA = ['mathvision-mini', 'mathvista', 'mmmu', 'charxiv-reasoning', 'charxiv-descriptive']
VIDEO_EXTRA = ['tempcompass-mc', 'tempcompass-yn', 'tempcompass-caption-matching', 'mvbench-public-19']
STATUS = ROOT / 'manifests/expanded-evaluation-status.json'


def progress(stage, **fields):
    previous = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    previous.update(stage=stage, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    write(STATUS, previous)
    print(stage, flush=True)


def prepare_extras():
    targets = {'mathvista': ['mathvista'], 'mmmu': ['mmmu'],
               'charxiv': ['charxiv-reasoning', 'charxiv-descriptive'],
               'tempcompass': VIDEO_EXTRA[:3], 'mvbench': ['mvbench-public-19']}
    while targets:
        for name, benchmarks in list(targets.items()):
            markers = [ROOT / 'manifests' / ('benchmark-' + b + '.json') for b in benchmarks]
            if all(p.exists() and json.loads(p.read_text()).get('status') == 'ready' for p in markers):
                del targets[name]
                continue
            receipt = ROOT / 'manifests' / (name + '-prepared-assets.json')
            if receipt.exists() and json.loads(receipt.read_text()).get('status') == 'complete':
                progress('preparing_' + name, status='running')
                run([sys.executable, str(ROOT / 'scripts/prepare_extra_benchmarks.py'), name],
                    'prepare-extra-' + name + '.log')
        if targets:
            queue = ROOT / 'manifests/extra-benchmarks-status.json'
            if queue.exists() and json.loads(queue.read_text()).get('status') == 'failed':
                raise RuntimeError('Extra benchmark downloader failed; inspect its status')
            progress('waiting_for_extra_downloads', pending=list(targets), status='running')
            time.sleep(20)


def evaluate(model, benchmarks, name, complete):
    if name.startswith('official-') and name.endswith('-control'):
        _, kind, stage, _ = name.split('-')
        metadata = ROOT / 'manifests' / f'{kind}-{stage}-control-process.json'
        status_path = ROOT / 'manifests' / f'{kind}-{stage}-control-status.json'
        while metadata.exists() and process_alive(metadata):
            control = json.loads(status_path.read_text()) if status_path.exists() else {}
            if control.get('status') == 'complete':
                break
            progress('waiting_for_existing_' + name, status='running')
            time.sleep(20)
    for benchmark in benchmarks:
        metadata = ROOT / 'manifests' / f'{name}-{benchmark}-process.json'
        while metadata.exists() and process_alive(metadata):
            progress('waiting_for_existing_' + name + '-' + benchmark, status='running')
            time.sleep(20)
    progress('evaluating_' + name, status='running', model=str(model))
    evaluate_suite(model, benchmarks, name)
    for benchmark in benchmarks:
        folder = ROOT / 'outputs' / (name + '-' + benchmark)
        report = json.loads((folder / 'summary.json').read_text())
        if report['status'] != 'complete' or not report.get('full_dataset_evaluated') or report.get('self_test_only'):
            raise RuntimeError('Evaluation is incomplete: ' + str(folder))
        complete[name + '/' + benchmark] = dict(output=str(folder), **report)
        write(ROOT / 'manifests/expanded-evaluation-results.json', complete)


def main():
    judge = None
    shared_judge = os.environ.get('PYVISION_REUSE_JUDGE') == '1'
    try:
        prepare_extras()
        progress('waiting_for_image_video_training_and_main_benchmarks', status='running', pending=[])
        while True:
            primary = json.loads((ROOT / 'status.json').read_text())
            if primary.get('image_main_evaluation_completed') and primary.get('video_main_evaluation_completed'):
                break
            if primary.get('status') == 'failed' and not process_alive(ROOT / 'manifests/pipeline-process.json'):
                raise RuntimeError('Primary reproduction pipeline failed; preserve results and repair it before continuing')
            time.sleep(20)
        image_run, video_run = primary['full_run'], primary['video_full_run']
        for name in [image_run, video_run]:
            if not completed_training(name, 700):
                raise RuntimeError('Missing verified step-700 run: ' + name)
        while True:
            try:
                assert_idle(['1'] + ([] if shared_judge else ['5']))
                break
            except RuntimeError:
                time.sleep(20)
        state('starting_expanded_evaluation', status='running')
        if not shared_judge:
            with (ROOT / 'logs/expanded-evaluation-judge.log').open('a') as log:
                judge = subprocess.Popen([sys.executable, str(ROOT / 'scripts/serve_judge.py')],
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(120):
            if judge and judge.poll() is not None:
                raise RuntimeError('Supplementary evaluation judge failed to start')
            try:
                with urllib.request.urlopen('http://127.0.0.1:18961/v1/models', timeout=5) as response:
                    if response.status == 200:
                        selection = json.loads((ROOT / 'judge-selection.json').read_text())
                        if not any(model.get('id') == 'judge' and model.get('root') == selection['model_path']
                                   for model in json.load(response)['data']):
                            raise RuntimeError('Supplementary endpoint serves a different judge model')
                        break
            except OSError:
                pass
            time.sleep(10)
        else:
            raise RuntimeError('Supplementary judge startup timed out')
        complete = {}
        def local_model(name):
            return ROOT / 'outputs' / name / 'checkpoints/global_step_700/actor/huggingface'
        evaluate(local_model(image_run), IMAGE_EXTRA, 'local-image-rl-extra', complete)
        evaluate(local_model(video_run), VIDEO_EXTRA, 'local-video-rl-extra', complete)
        for kind, benchmarks in [('image', IMAGE_MAIN + IMAGE_EXTRA), ('video', ['vsibench'] + VIDEO_EXTRA)]:
            for stage in ['sft', 'rl']:
                model = ROOT / 'models' / f'PyVision-{kind.capitalize()}-7B-{stage.upper()}'
                # Both official reference checkpoints are already in the primary download plan.
                receipt = ROOT / 'manifests' / f'{kind}-{stage}-model-prepared-assets.json'
                original = ROOT / 'manifests' / f'{kind}-{stage}-model.json'
                while not any(p.exists() and json.loads(p.read_text()).get('status') == 'complete'
                              for p in [receipt, original]):
                    if not process_alive(ROOT / 'manifests/preparation-queue-process.json'):
                        raise RuntimeError('Reference model unavailable: ' + str(model))
                    time.sleep(20)
                evaluate(model, benchmarks, f'official-{kind}-{stage}-control', complete)
        progress('expanded_evaluations_finished_review_results', status='complete',
                 result_file=str(ROOT / 'manifests/expanded-evaluation-results.json'),
                 paper_numbers_reproduced=False)
        state('training_and_expanded_benchmarks_finished_review_results', status='results_ready_for_review',
              expanded_evaluation_completed=True, paper_results_reproduced=False)
    except BaseException as exc:
        progress('failed', status='failed', error=str(exc))
        raise
    finally:
        stop_owned(judge)


if __name__ == '__main__':
    for variable in ['NO_PROXY', 'no_proxy']:
        os.environ[variable] = '127.0.0.1,localhost'
    with (ROOT / 'expanded-evaluation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
