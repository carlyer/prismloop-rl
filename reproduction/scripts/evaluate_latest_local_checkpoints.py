"""After matched controls finish, freeze and evaluate the latest saved local RL models."""
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from download_preparation_queue import ROOT, write, digest
from evaluation_evidence import audit_records
from pipeline import evaluate_suite, process_alive, run
from run_qwen_matched_controls import compare_protocol
from collect_results import vsi_task_metrics

SELECTION = ROOT / 'manifests/local-checkpoint-evaluation-selection.json'
STATUS = ROOT / 'manifests/local-checkpoint-evaluation-status.json'
REPORT = ROOT / 'manifests/local-checkpoint-evaluation-comparison.json'
BENCHMARKS = {'image': ['vstar', 'hrbench-4k'], 'video': ['vsibench']}
MODELS = {'image': 'PyVision-Image-7B-SFT', 'video': 'PyVision-Video-7B-SFT'}


def read(path):
    return json.loads(path.read_text())


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def identity():
    return dict(pid=os.getpid(), start_time_ticks=Path(f'/proc/{os.getpid()}/stat')
                .read_text().rsplit(')', 1)[1].split()[19], started_utc=now(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def publish_evidence():
    """Let the existing persistent monitor mirror new metadata, including later-created reports."""
    folder = ROOT / 'outputs/official-local-checkpoint-evaluation-evidence'
    folder.mkdir(exist_ok=True)
    patterns = ['local-checkpoint-evaluation-*.json', 'local-*-checkpoint-evaluation-*.json',
                'local-*-selected-*-validation.json']
    for pattern in patterns:
        for source in (ROOT / 'manifests').glob(pattern):
            target = folder / source.name
            if not target.exists():
                try:
                    target.symlink_to(source)
                except FileExistsError:
                    if target.resolve() != source.resolve():
                        raise


def progress(stage, **fields):
    write(STATUS, dict(status='running', stage=stage, updated_utc=now(), **fields))
    publish_evidence()
    print(stage, flush=True)


def pin_checkpoint(run_folder, destination):
    """Hard-link an immutable, fully committed checkpoint before rolling retention removes it."""
    checkpoint_root = run_folder / 'checkpoints'
    committed = int((checkpoint_root / 'latest_checkpointed_iteration.txt').read_text().strip())
    for source in sorted(checkpoint_root.glob('global_step_*'),
                         key=lambda p: int(p.name.rsplit('_', 1)[-1]), reverse=True):
        step = int(source.name.rsplit('_', 1)[-1])
        if step > committed or step < 2:
            continue
        models = list((source / 'actor').glob('model_world_size_*_rank_*.pt'))
        if not models:
            continue
        match = re.fullmatch(r'model_world_size_(\d+)_rank_\d+\.pt', models[0].name)
        world_size = int(match.group(1))
        required = [source / 'data.pt']
        required += [source / 'actor' / f'{prefix}_world_size_{world_size}_rank_{rank}.pt'
                     for prefix in ['model', 'optim', 'extra_state'] for rank in range(world_size)]
        if not all(p.is_file() and p.stat().st_size > 0 for p in required):
            continue
        destination.mkdir(parents=True, exist_ok=False)
        target = destination / 'checkpoints' / source.name
        target.parent.mkdir()
        copied = []
        def link_or_copy(src, dst):
            try:
                os.link(src, dst)
            except OSError as exc:
                if exc.errno != errno.EXDEV or Path(src).stat().st_size > 64 * 1024**2:
                    raise
                shutil.copy2(src, dst)
                copied.append(str(Path(src).relative_to(source)))
            return dst
        shutil.copytree(source, target, copy_function=link_or_copy)
        for p in required:
            frozen = target / p.relative_to(source)
            if frozen.stat().st_size != p.stat().st_size:
                raise RuntimeError('Checkpoint changed while pinning: ' + str(p))
        (target.parent / 'latest_checkpointed_iteration.txt').write_text(str(step) + '\n')
        return dict(step=step, source_run=str(run_folder), source_checkpoint=str(source.resolve()),
                    pinned_run=str(destination), pinned_checkpoint=str(target),
                    model=str(destination / 'merged-model'),
                    pinning='hard links to committed immutable checkpoint files; small cross-filesystem files copied',
                    copied_small_files=copied, selected_utc=now(),
                    benchmarks=BENCHMARKS.get('video' if 'video' in run_folder.name else 'image', []))
    raise RuntimeError('No complete non-initial checkpoint available: ' + str(run_folder))


def complete_inference_assets(model, source_model):
    """Copy exact SFT tokenizer/processor assets without changing merged weights/config."""
    names = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'added_tokens.json', 'vocab.json', 'merges.txt', 'preprocessor_config.json',
             'processor_config.json', 'video_preprocessor_config.json',
             'chat_template.json', 'chat_template.jinja', 'generation_config.json']
    required = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                'preprocessor_config.json']
    for name in required:
        if not (source_model / name).is_file():
            raise RuntimeError('Source model lacks required inference asset: ' + name)
    protected = {name: digest(model / name) for name in ['config.json', 'model.safetensors.index.json']}
    copied, backups = {}, []
    for name in names:
        source, target = source_model / name, model / name
        if not source.is_file():
            continue
        expected = digest(source)
        if target.exists() and digest(target) != expected:
            backup = model.parent / 'processor-assets-before' / name
            backup.parent.mkdir(exist_ok=True)
            if not backup.exists():
                shutil.copy2(target, backup)
            backups.append(str(backup))
        if not target.exists() or digest(target) != expected:
            temporary = target.with_name(target.name + f'.{os.getpid()}.assets-tmp')
            shutil.copy2(source, temporary)
            temporary.replace(target)
        if digest(target) != expected:
            raise RuntimeError('Inference asset copy differs from SFT: ' + name)
        copied[name] = expected
    if any(digest(model / name) != expected for name, expected in protected.items()):
        raise RuntimeError('Protected model config or index changed during asset repair')
    from transformers import AutoTokenizer, AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    reference = AutoTokenizer.from_pretrained(source_model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    for text in ['<answer>\\boxed{C}</answer>', '<code>\nprint(1)\n</code>', '<|vision_start|><|image_pad|><|vision_end|>']:
        if tokenizer.encode(text) != reference.encode(text):
            raise RuntimeError('Export tokenizer differs from SFT tokenizer')
    return dict(status='passed', model=str(model), source_model=str(source_model),
                exact_source_asset_sha256=copied, tokenizer_class=type(tokenizer).__name__,
                processor_class=type(processor).__name__, tokenizer_fixture_ids_match_sft=True,
                protected_model_config_and_index_unchanged=True, backups=backups, observed_utc=now())


def checkpoint_worker(kind):
    selection = read(SELECTION)
    selected = selection['checkpoints'][kind]
    label = selected['evaluation_prefix']
    worker_status = ROOT / 'manifests' / f'local-{kind}-checkpoint-evaluation-status.json'
    with (ROOT / f'local-{kind}-checkpoint-evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(ROOT / 'manifests' / f'local-{kind}-checkpoint-evaluation-process.json', identity())
        fields = dict(status='running', kind=kind, checkpoint_step=selected['step'],
                      selected_checkpoint=selected['source_checkpoint'], benchmarks=BENCHMARKS[kind])
        def state(stage):
            fields.update(stage=stage, updated_utc=now())
            write(worker_status, fields)
            publish_evidence()
        try:
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT.parent / 'verl_agents'), OMP_NUM_THREADS='8')
            source_model = ROOT / 'models' / MODELS[kind]
            state('checking_saved_optimizer_and_weight_update')
            run([sys.executable, str(ROOT / 'scripts/validate_optimizer_update.py'),
                 '--run', selected['pinned_run'], '--source-model', str(source_model),
                 '--minimum-step', str(selected['step']), '--no-wait', '--report',
                 str(ROOT / 'manifests' / f'local-{kind}-selected-optimizer-validation.json')],
                label + '-optimizer-validation.log', env=env)
            state('merging_pinned_checkpoint_on_cpu')
            if Path(selected['model']).exists():
                validation = ROOT / 'manifests' / f'local-{kind}-selected-merged-validation.json'
                record = read(validation) if validation.exists() else {}
                model = Path(selected['model'])
                if (record.get('status') != 'passed' or record.get('model') != str(model)
                        or digest(model / 'config.json') != record.get('config_sha256')
                        or not record.get('model_shard_sha256')
                        or any(digest(model / name) != expected
                               for name, expected in record['model_shard_sha256'].items())):
                    raise RuntimeError('Existing export lacks a matching complete weight validation; preserve before retrying')
            else:
                run([sys.executable, str(ROOT.parent / 'verl_agents/scripts/model_merger.py'), '--backend', 'fsdp',
                     '--local_dir', str(Path(selected['pinned_checkpoint']) / 'actor'),
                     '--hf_model_path', str(source_model), '--target_dir', selected['model']],
                    label + '-merge.log', env=env)
            state('checking_tokenizer_and_processor_assets')
            assets = complete_inference_assets(Path(selected['model']), source_model)
            write(ROOT / 'manifests' / f'local-{kind}-selected-inference-assets-validation.json', assets)
            state('checking_all_merged_parameters_and_precision')
            run([sys.executable, str(ROOT / 'scripts/validate_merged_model.py'),
                 '--model', selected['model'], '--source-model', str(source_model), '--report',
                 str(ROOT / 'manifests' / f'local-{kind}-selected-merged-validation.json')],
                label + '-merged-validation.log', env=env)
            state('evaluating_full_benchmarks')
            evaluate_suite(Path(selected['model']), BENCHMARKS[kind], label,
                           gpu='1' if kind == 'image' else '4',
                           port=18970 if kind == 'image' else 18971, update_state=False)
            fields.update(status='complete')
            state('full_benchmark_inference_finished')
        except BaseException as exc:
            fields.update(status='failed', error=str(exc))
            state('needs_attention')
            raise


def compare_results(selection):
    result = dict(status='pending', observed_utc=now(), checkpoints=selection['checkpoints'],
                  results=[], pending=[], errors=[], paper_results_reproduced=False,
                  scope='Intermediate local RL checkpoints versus official SFT under the same local evaluation',
                  limitations=['Single checkpoint evaluation; score differences are not statistical significance.',
                               'VStar is also used for training-time validation.',
                               'Released PyVision VSI scoring and native VSI rule scoring are reported separately.'])
    for kind in ['image', 'video']:
        selected = selection['checkpoints'][kind]
        for benchmark in BENCHMARKS[kind]:
            new = ROOT / 'outputs' / (selected['evaluation_prefix'] + '-' + benchmark)
            base = ROOT / 'outputs' / f'official-{kind}-sft-control-{benchmark}'
            if not (new / 'summary.json').exists():
                result['pending'].append(new.name)
                continue
            try:
                config, report = read(new / 'invocation.json'), read(new / 'summary.json')
                base_config, base_report = read(base / 'invocation.json'), read(base / 'summary.json')
                compare_protocol(config, base_config)
                roots = {Path(m['root']).resolve() for m in config.get('policy_models', [])}
                if roots != {Path(selected['model']).resolve()}:
                    raise ValueError('Inference service did not use the frozen local checkpoint')
                evidence = audit_records(new, config, report, ROOT / 'data/evaluation' / (benchmark + '.json'))
                base_evidence = audit_records(base, base_config, base_report,
                                             ROOT / 'data/evaluation' / (benchmark + '.json'))
                score, base_score = 100 * report['score_mean'], 100 * base_report['score_mean']
                row = dict(kind=kind, benchmark=benchmark, checkpoint_step=selected['step'],
                           local_rl_percent=score, sft_percent=base_score,
                           difference_percentage_points=score-base_score,
                           metric='mean answer score', evidence=evidence, sft_evidence=base_evidence,
                           output=str(new), inference_protocol_matches_sft=True)
                if benchmark == 'vsibench':
                    from native_vsi_metrics import rescore
                    score = 100 * vsi_task_metrics(report)['task_macro_mean']
                    base_score = 100 * vsi_task_metrics(base_report)['task_macro_mean']
                    native = rescore(new, report)['metrics']['overall']
                    base_native = rescore(base, base_report)['metrics']['overall']
                    row.update(local_rl_percent=score, sft_percent=base_score,
                               difference_percentage_points=score-base_score,
                               metric='released PyVision scoring, eight-task macro',
                               native_vsi_local_rl_percent=native, native_vsi_sft_percent=base_native,
                               native_vsi_difference_percentage_points=native-base_native)
                result['results'].append(row)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                result['errors'].append(dict(run=new.name, error=str(exc)))
    if not result['pending'] and not result['errors']:
        result['status'] = 'complete'
    elif result['errors']:
        result['status'] = 'needs_attention'
    write(REPORT, result)
    publish_evidence()
    lines = ['Latest saved local RL checkpoints versus official SFT',
             'Intermediate checkpoints; not final 700-step reproduction results.',
             'Model | Checkpoint step | Benchmark | Local RL (%) | SFT (%) | Difference (pp)']
    for row in result['results']:
        lines.append(f"{row['kind']} | {row['checkpoint_step']} | {row['benchmark']} | "
                     f"{row['local_rl_percent']:.3f} | {row['sft_percent']:.3f} | "
                     f"{row['difference_percentage_points']:+.3f}")
        if row['benchmark'] == 'vsibench':
            lines.append('VSI native rules: local RL={:.3f}, SFT={:.3f}, difference={:+.3f} pp'.format(
                row['native_vsi_local_rl_percent'], row['native_vsi_sft_percent'],
                row['native_vsi_difference_percentage_points']))
    lines += ['Status: ' + result['status'], 'Pending: ' + ', '.join(result['pending']),
              'Observed UTC: ' + result['observed_utc']]
    target = ROOT / 'LOCAL_CHECKPOINT_RL_EFFECT_COMPARISON.txt'
    temp = target.with_name(target.name + f'.{os.getpid()}.tmp')
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(target)
    return result


def main():
    with (ROOT / 'latest-local-checkpoint-evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(ROOT / 'manifests/local-checkpoint-evaluation-process.json', identity())
        try:
            progress('waiting_for_completed_qwen_controls')
            prerequisite = ROOT / 'manifests/qwen-sft-rl-matched-comparison.json'
            while True:
                qwen_status = ROOT / 'manifests/qwen-instruct-control-status.json'
                ready = qwen_status.exists() and read(qwen_status).get('status') == 'complete'
                owners_done = not process_alive(ROOT / 'manifests/qwen-instruct-control-process.json')
                if ready and owners_done:
                    break
                if not process_alive(ROOT / 'manifests/qwen-matched-comparison-process.json'):
                    raise RuntimeError('Prerequisite comparison stopped incomplete; inspect before running checkpoint evaluations')
                publish_evidence()
                time.sleep(30)
            progress('pinning_latest_complete_checkpoints')
            if SELECTION.exists():
                selection = read(SELECTION)
                chosen = selection['checkpoints']
            else:
                state = read(ROOT / 'status.json')
                tag = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
                chosen = {}
                for kind, run_name in [('image', state['full_run']), ('video', state['video_full_run'])]:
                    prefix = f'official-local-{kind}-checkpoint-{tag}'
                    folder = ROOT / 'outputs' / (prefix + '-snapshot')
                    chosen[kind] = pin_checkpoint(ROOT / 'outputs' / run_name, folder)
                    chosen[kind]['evaluation_prefix'] = prefix + '-step' + str(chosen[kind]['step'])
                    chosen[kind]['benchmarks'] = BENCHMARKS[kind]
                selection = dict(selected_utc=now(), checkpoints=chosen,
                                 prerequisite_comparison=str(prerequisite),
                                 selection_rule='Latest committed complete checkpoint after Qwen controls finish; each GPU waits for its existing owner')
                write(SELECTION, selection)
            progress('exporting_and_evaluating_pinned_checkpoints', selected_steps={k:v['step'] for k,v in chosen.items()})
            children = []
            for kind in ['image', 'video']:
                metadata = ROOT / 'manifests' / f'local-{kind}-checkpoint-evaluation-process.json'
                worker_status = ROOT / 'manifests' / f'local-{kind}-checkpoint-evaluation-status.json'
                if (process_alive(metadata) or
                        (worker_status.exists() and read(worker_status).get('status') == 'complete')):
                    continue
                with (ROOT / 'logs' / f'official-local-{kind}-checkpoint-controller.log').open('a') as log:
                    children.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--kind', kind],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            while (any(child.poll() is None for child in children) or
                   any(process_alive(ROOT / 'manifests' / f'local-{kind}-checkpoint-evaluation-process.json')
                       for kind in ['image', 'video'])):
                compare_results(selection)
                time.sleep(60)
            report = compare_results(selection)
            if (report['status'] != 'complete' or any(
                    read(ROOT / 'manifests' / f'local-{kind}-checkpoint-evaluation-status.json').get('status') != 'complete'
                    for kind in ['image', 'video'])):
                raise RuntimeError('Checkpoint evaluation incomplete; inspect worker statuses and evidence')
            run([sys.executable, str(ROOT / 'scripts/collect_results.py')], 'official-local-checkpoint-collect-results.log')
            write(STATUS, dict(status='complete', stage='intermediate_rl_vs_sft_comparison_complete',
                              report=str(REPORT), updated_utc=now(), final_700_step_training_complete=False))
            publish_evidence()
        except BaseException as exc:
            write(STATUS, dict(status='failed', stage='needs_attention', error=str(exc), updated_utc=now()))
            publish_evidence()
            raise


if __name__ == '__main__':
    for key in ['NO_PROXY', 'no_proxy']:
        os.environ[key] = '127.0.0.1,localhost'
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['image', 'video'])
    arguments = parser.parse_args()
    checkpoint_worker(arguments.kind) if arguments.kind else main()
