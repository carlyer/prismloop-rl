"""Queue the remaining paper baseline after verified Image recovery finishes."""
from datetime import datetime, timezone
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

from pipeline import completed_training, process_alive
from run_training import ROOT, REPO, PYTHON, assert_idle
from expanded_evaluation_pipeline import IMAGE_MAIN, IMAGE_EXTRA, VIDEO_EXTRA


def write(path, value):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def localize_prepared_inputs():
    source = ROOT / 'data'
    if not source.is_symlink():
        return
    destination = ROOT / 'data-local-build'
    if destination.is_symlink():
        raise RuntimeError('Refusing to prepare inputs through an unexpected build-directory symlink')
    destination.mkdir(exist_ok=True)
    for child in source.resolve(strict=True).iterdir():
        target = destination / child.name
        if child.name == 'prepared':
            shutil.copytree(child, target, dirs_exist_ok=True)
        elif not target.exists() and not target.is_symlink():
            target.symlink_to(child, target_is_directory=child.is_dir())
    # Keep the path continuously available to concurrent control processes.
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 2)
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(source))
    destination.rename(ROOT / 'data-original-readonly-link')


def ensure_shared_judge():
    selection = json.loads((ROOT / 'judge-selection.json').read_text())
    endpoint = f"http://127.0.0.1:{selection['port']}/v1/models"
    def healthy():
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                models = json.load(response)['data']
        except OSError:
            return False
        if not any(m.get('id') == 'judge' and m.get('root') == selection['model_path'] for m in models):
            raise RuntimeError('Existing judge endpoint advertises a different model')
        return True
    deadline = time.monotonic() + 1200
    while not healthy():
        if time.monotonic() > deadline:
            raise RuntimeError('Shared judge could not become ready')
        try:
            assert_idle(selection['gpus'].split(','))
        except RuntimeError:
            time.sleep(10)
            continue
        command = [str(PYTHON), '-u', str(ROOT / 'scripts/serve_judge.py')]
        with (ROOT / 'logs/shared-baseline-judge.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        ticks = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write(ROOT / 'manifests/recovery-judge-process.json', dict(
            pid=process.pid, start_time_ticks=ticks, command=command))
        while not healthy():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Shared baseline judge failed to start')
            time.sleep(5)
        break


def compare():
    results = json.loads((ROOT / 'manifests/collected-results.json').read_text())
    status = json.loads((ROOT / 'status.json').read_text())
    reference = json.loads((ROOT / 'references/paper-reference-metrics.json').read_text())
    accepted = {r['run']: r for r in results['completed_full_evaluations']}
    required = {}
    for kind, benchmarks in [('image', IMAGE_MAIN + IMAGE_EXTRA), ('video', ['vsibench'] + VIDEO_EXTRA)]:
        for stage, origin in [('sft', 'official_sft'), ('rl', 'official_released_rl')]:
            required.update({f'official-{kind}-{stage}-control-{benchmark}': origin for benchmark in benchmarks})
    required.update({'local-image-rl-extra-' + b: 'local_trained_rl' for b in IMAGE_EXTRA})
    required.update({'local-video-rl-extra-' + b: 'local_trained_rl' for b in VIDEO_EXTRA})
    for name, origin in required.items():
        if name not in accepted or accepted[name].get('model_stage') != origin:
            raise RuntimeError('Complete the requested evaluation with verified model origin: ' + name)
    image = status['full_run'].removesuffix('-full')
    video = status['video_full_run'].removesuffix('-full')
    rows, controls = [], []
    for kind, prefix, benchmarks, paper_values in [
        ('image', image, reference['table1_benchmarks'], reference['table1_percent']['PyVision-Image']),
        ('video', video, ['vsibench'], [reference['table2_vsibench_percent']['PyVision-Video']]),
    ]:
        if not completed_training(prefix + '-full', 700):
            raise RuntimeError('Missing successful step-700 training: ' + prefix)
        validation = json.loads((ROOT / 'manifests' / (kind + '-final-merged-model-validation.json')).read_text())
        if validation.get('status') != 'passed':
            raise RuntimeError('Final merged model validation did not pass')
        optimizer = json.loads((ROOT / 'manifests' / (kind + '-final-optimizer-update-validation.json')).read_text())
        if (optimizer.get('status') != 'passed_nonzero_weight_update'
                or optimizer.get('checkpoint_step') != 700
                or optimizer.get('nonzero_weight_update_verified') is not True):
            raise RuntimeError('Final step-700 weights lack direct nonzero-update verification')
        for benchmark, paper_score in zip(benchmarks, paper_values):
            name = prefix + '-official-' + benchmark
            if name not in accepted:
                raise RuntimeError('Missing accepted full paper benchmark: ' + name)
            row = accepted[name]
            if row.get('model_stage') != 'local_trained_rl':
                raise RuntimeError('A published model cannot substitute for local training: ' + name)
            score = row['category_macro_percent'] if benchmark == 'vsibench' else row['micro_accuracy_percent']
            rows.append(dict(benchmark=benchmark, run=name, local_percent=score,
                             paper_percent=paper_score, delta_percentage_points=score-paper_score,
                             relative_change_percent=100*(score/paper_score-1),
                             absolute_gap_over_5pp=abs(score-paper_score) > 5,
                             gap_flag_definition='Magnitude heuristic, not a statistical significance test',
                             native_vsi_overall_percent=row.get('native_vsi_overall_percent'),
                             samples=row['samples'], output=row['output']))
            local_config = json.loads((Path(row['output']) / 'invocation.json').read_text())
            def protocol(config):
                return {key: value for key, value in config.items()
                        if key not in ['policy_models', 'policy_weight_identity']}
            for stage in ['sft', 'rl']:
                control = accepted[f'official-{kind}-{stage}-control-{benchmark}']
                control_config = json.loads((Path(control['output']) / 'invocation.json').read_text())
                same_protocol = protocol(local_config) == protocol(control_config)
                control_score = (control['category_macro_percent'] if benchmark == 'vsibench'
                                 else control['micro_accuracy_percent'])
                controls.append(dict(benchmark=benchmark, control_stage=control['model_stage'],
                                     local_percent=score, control_percent=control_score,
                                     same_evaluation_protocol=same_protocol,
                                     delta_percentage_points=score-control_score if same_protocol else None,
                                     local_run=name, control_run=control['run']))
    report = dict(status='baseline_training_and_paper_benchmarks_complete',
                  observed_utc=datetime.now(timezone.utc).isoformat(), comparisons=rows,
                  local_vs_controls=controls,
                  differences=['Two L20X training GPUs instead of eight H100s',
                               'Local Qwen2.5-VL-32B judge instead of paper 72B judge',
                               'Recorded MathVision/MathVerse/DynaMath split ambiguities',
                               'Video generation cap retains released-code value 16; paper overview says 32',
                               'Video validation uses a fixed local 400-question sample; author IDs unavailable',
                               'Malformed numerical predictions receive zero rather than aborting training'],
                  all_accepted_controls=results['completed_full_evaluations'],
                  expanded_evaluations_verified=True,
                  strict_paper_protocol_match=False, research_phase_ready=True,
                  improvement_experiments_completed=False)
    write(ROOT / 'manifests/paper-baseline-comparison.json', report)
    lines = ['PyVision-RL 训练与评测复现对比',
             '本机模型从官方 SFT 权重完成 Image/Video 各 700 步 RL。',
             '本机硬件和 32B judge 与论文设置不同；数值差距不能全部归因于训练方法。',
             '论文差距超过 5 个百分点仅标记为幅度较大，不表示统计显著。', '',
             'Bench | 本机(%) | 论文(%) | 差值(百分点) | 差距超过5点']
    for row in rows:
        lines.append(f"{row['benchmark']} | {row['local_percent']:.3f} | {row['paper_percent']:.3f} | "
                     f"{row['delta_percentage_points']:+.3f} | {row['absolute_gap_over_5pp']}")
    lines += ['', '本机模型与同协议对照：', 'Bench | 对照权重 | 对照(%) | 本机差值(百分点)']
    for row in controls:
        difference = (f"{row['delta_percentage_points']:+.3f}" if row['same_evaluation_protocol']
                      else '协议不一致，需核对后再比较')
        lines.append(f"{row['benchmark']} | {row['control_stage']} | {row['control_percent']:.3f} | {difference}")
    lines += ['', '完整逐题验收与权重来源：manifests/collected-results.json',
              '扩展 Bench 已单独记录；public-19 不能冒充完整 MVBench。',
              '基线完成后进入改进实验阶段；此报告不表示改进已带来提升。']
    (ROOT / 'PAPER_BASELINE_COMPARISON.txt').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parent = ROOT / 'manifests/recovery-coordinator-process.json'
    owner = json.loads(parent.read_text())
    path = ROOT / 'manifests/baseline-continuation-status.json'
    record = dict(status='running', stage='queued_after_image_training',
                  prerequisite_owner=owner, research_experiments_started=False)
    def progress(stage, **extra):
        record.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write(path, record)
        print(stage, flush=True)
    try:
        progress(record['stage'])
        while True:
            current = json.loads(parent.read_text())
            if (current['pid'], current.get('start_time_ticks')) != (owner['pid'], owner.get('start_time_ticks')):
                progress('superseded_by_new_recovery_owner', status='superseded')
                return
            status = json.loads((ROOT / 'status.json').read_text())
            if status.get('full_run') != owner['prefix'] + '-full':
                raise RuntimeError('Current Image run changed; verify the new owner before handoff')
            if completed_training(status['full_run'], 700):
                break
            if not process_alive(parent):
                raise RuntimeError('Image owner stopped before successful step-700 completion')
            time.sleep(30)
        # The completed Image trainer is independent of the coordinator process.
        # Handoff only this verified owner; keep its judge and SFT process groups alive.
        if owner.get('supervision_mode') != 'training_process_alias_no_termination' and process_alive(parent):
            descriptor = os.pidfd_open(owner['pid'])
            try:
                if process_alive(parent):
                    signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            finally:
                os.close(descriptor)
        progress('image_complete_handoff_preserving_shared_judge_and_controls')
        ensure_shared_judge()
        progress('preparing_local_writable_training_inputs')
        localize_prepared_inputs()
        command = [str(PYTHON), '-u', str(ROOT / 'scripts/pipeline.py')]
        progress('running_image_paper_evaluation_then_video_training_and_evaluation')
        with (ROOT / 'logs/baseline-continuation-pipeline.log').open('ab') as log:
            environment = dict(os.environ, PYVISION_REUSE_JUDGE='1')
            process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        ticks = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write(ROOT / 'manifests/pipeline-process.json', dict(pid=process.pid, start_time_ticks=ticks,
                                                          command=command))
        code = process.wait()
        if code:
            raise RuntimeError('Baseline pipeline failed; preserve its completed phases and inspect the log')
        progress('waiting_for_requested_controls_and_expanded_evaluations')
        controls = [ROOT / 'manifests' / name for name in [
            'image-sft-control-process.json', 'video-sft-control-process.json',
            'official-image-sft-control-mathvision-mini-process.json', 'image-rl-early-control-process.json',
            'image-rl-control-process.json', 'video-rl-control-process.json', 'expanded-evaluation-process.json']]
        while any(path.exists() and process_alive(path) for path in controls):
            time.sleep(30)
        for kind in ['image', 'video']:
            for stage in ['sft', 'rl']:
                control = json.loads((ROOT / 'manifests' / f'{kind}-{stage}-control-status.json').read_text())
                if control.get('status') != 'complete':
                    raise RuntimeError(kind + ' ' + stage + ' controls need repair before final comparison')
        expanded = json.loads((ROOT / 'manifests/expanded-evaluation-status.json').read_text())
        if expanded.get('status') != 'complete':
            raise RuntimeError('Expanded evaluations need repair before final comparison')
        progress('collecting_complete_results_and_paper_differences')
        subprocess.run([str(PYTHON), str(ROOT / 'scripts/collect_results.py')], cwd=REPO, check=True)
        report = compare()
        original = (ROOT / 'origin-reproduction').resolve()
        write(original / 'manifests/paper-baseline-comparison.json', report)
        shutil.copy2(ROOT / 'PAPER_BASELINE_COMPARISON.txt', original / 'PAPER_BASELINE_COMPARISON.txt')
        progress('baseline_complete_ready_for_analysis_and_improvement_experiments', status='complete',
                 comparison=str(ROOT / 'manifests/paper-baseline-comparison.json'))
    except BaseException as exc:
        progress('needs_attention', status='failed', error=str(exc))
        raise


if __name__ == '__main__':
    for variable in ['NO_PROXY', 'no_proxy']:
        os.environ[variable] = '127.0.0.1,localhost'
    with (ROOT / 'baseline-continuation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ticks = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write(ROOT / 'manifests/baseline-continuation-process.json', dict(
            pid=os.getpid(), start_time_ticks=ticks,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        main()
