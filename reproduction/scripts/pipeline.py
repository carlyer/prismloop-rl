#!/usr/bin/env python3
"""Continue the authorized Image RL reproduction after long asset downloads."""
import json
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from run_training import ROOT, REPO, PYTHON, assert_idle

STATE = ROOT / "status.json"


def update_state_fields(**fields):
    with (ROOT / 'state.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = json.loads(STATE.read_text()) if STATE.exists() else {}
        record.update(updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields)
        tmp = STATE.with_name(STATE.name + '.tmp-' + str(os.getpid()))
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(STATE)
    return record


def state(stage, **fields):
    update_state_fields(stage=stage, **fields)
    print(stage, flush=True)


def process_alive(meta):
    try:
        record = json.loads(meta.read_text())
        fields = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and (record.get('start_time_ticks') is None or
                                     str(record['start_time_ticks']) == fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return False


def wait_for(check, stage, process_meta=None):
    state(stage, status="running")
    while not check():
        if process_meta and not process_alive(ROOT / "manifests" / process_meta):
            raise RuntimeError(f"{stage}: prerequisite process stopped; inspect reproduction/logs")
        time.sleep(20)


def run(command, log_name, env=None):
    with (ROOT / "logs" / log_name).open("a") as log:
        log.write("\nCOMMAND: " + json.dumps(command) + "\n")
        log.flush()
        p = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            code = p.wait()
        except BaseException:
            os.killpg(p.pid, signal.SIGTERM)
            raise
    if code:
        raise RuntimeError(f"Command failed ({code}); see {log_name}")


def stop_owned(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


def completed_training(name, step):
    folder = ROOT / 'outputs' / name
    receipt = folder / 'exit-status.json'
    actor = folder / 'checkpoints' / f'global_step_{step}' / 'actor'
    return (receipt.exists() and json.loads(receipt.read_text()).get('returncode') == 0
            and bool(list(actor.glob('model_world_size_*_rank_*.pt'))))


def wait_for_parallel_video(name):
    """Reuse the independently supervised Video trainer; never write its run twice."""
    metadata = ROOT / 'manifests/parallel-video-training-process.json'
    if not metadata.exists():
        return False
    owner = json.loads(metadata.read_text())
    if owner.get('run_name') != name:
        raise RuntimeError('Another parallel Video run owns the training metadata; inspect before continuing')
    wait_for(lambda: completed_training(name, 700), 'waiting_for_parallel_video_training',
             'parallel-video-training-process.json')
    return True


def evaluate_suite(model, benchmarks, name, gpu='1', port=18962, update_state=True):
    policy = None
    def wait(check, stage, process_meta=None):
        if update_state:
            return wait_for(check, stage, process_meta)
        print(stage, flush=True)
        while not check():
            if process_meta and not process_alive(ROOT / 'manifests' / process_meta):
                raise RuntimeError(stage + ': prerequisite process stopped')
            time.sleep(20)
    def benchmark_ready(benchmark):
        if not (ROOT / 'data/evaluation' / (benchmark + '.json')).exists():
            return False
        if benchmark == 'vstar':
            marker, expected = ROOT / 'manifests/vstar.json', 'complete'
        elif benchmark == 'vsibench':
            marker, expected = ROOT / 'manifests/vsibench-local.json', 'ready'
        else:
            marker, expected = ROOT / 'manifests' / ('benchmark-' + benchmark + '.json'), 'ready'
        return marker.exists() and json.loads(marker.read_text()).get('status') == expected
    for benchmark in benchmarks:
        wait(lambda b=benchmark: benchmark_ready(b),
                 'waiting_for_benchmark_' + benchmark, 'preparation-queue-process.json')
    def policy_gpu_ready():
        try:
            assert_idle([gpu])
            return True
        except RuntimeError:
            return False
    wait(policy_gpu_ready, "waiting_for_evaluation_gpu")
    with (ROOT / "logs" / (name + "-policy.log")).open("a") as log:
        try:
            policy = subprocess.Popen([sys.executable, str(ROOT / "scripts/serve_policy.py"), "--model", str(model),
                                       '--gpus', gpu, '--port', str(port)],
                                      cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(120):
                if policy.poll() is not None:
                    raise RuntimeError("Evaluation policy service failed; inspect its log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(10)
            else:
                raise RuntimeError("Evaluation policy service startup timed out")
            for benchmark in benchmarks:
                if update_state:
                    state("evaluating_" + benchmark, current_evaluation_model=str(model))
                command = [str(PYTHON), str(ROOT / "scripts/evaluate_official.py"), "--benchmark", benchmark,
                           "--output", str(ROOT / "outputs" / (name + "-" + benchmark)),
                           '--endpoint', f'http://127.0.0.1:{port}/v1']
                if benchmark == "vstar":
                    command += ["--repeat", "32"]
                run(command, name + "-" + benchmark + ".log")
        finally:
            stop_owned(policy)


def main():
    judge = None
    judge_log = None
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    shared_judge = os.environ.get('PYVISION_REUSE_JUDGE') == '1'
    try:
        selection_path = ROOT / "judge-selection.json"
        selection = json.loads(selection_path.read_text())
        calibration = json.loads((ROOT / "manifests/judge-calibration.json").read_text())
        if (calibration.get("status") != "passed" or
                calibration.get("selection_sha256") != hashlib.sha256(selection_path.read_bytes()).hexdigest()):
            raise RuntimeError("Selected judge has not passed its calibration")
        verifier = REPO / "verl_agents/verl/utils/reward_score/multi_task_verifier.py"
        if calibration.get("upstream_verifier_sha256") != hashlib.sha256(verifier.read_bytes()).hexdigest():
            raise RuntimeError("Reward function changed after judge calibration")
        state("judge_calibration_verified", judge_model=selection["model_id"],
              judge_matches_paper=selection["matches_paper_judge"], judge_calibration_passed=True,
              status='running', error=None)
        wait_for(lambda: (ROOT / "manifests/environment-ready.json").is_file(),
                 "waiting_for_environment", "bootstrap-process.json")
        def assets_ready():
            names = ["image-rl-data", "image-sft-model"]
            return all((ROOT / "manifests" / f"{n}.json").exists() and
                       json.loads((ROOT / "manifests" / f"{n}.json").read_text()).get("status") == "complete"
                       for n in names)
        wait_for(assets_ready, "waiting_for_official_assets", "download-process.json")
        preparation = ROOT / 'manifests/image-data-validation.json'
        image_files = [ROOT / 'data/prepared/image_train.json', ROOT / 'data/prepared/image_smoke.json']
        prepared_ok = (preparation.exists() and all(p.is_file() for p in image_files)
                       and json.loads(preparation.read_text()).get('all_media_paths_exist'))
        if not prepared_ok:
            state("preparing_image_data")
            run([sys.executable, str(ROOT / "scripts/prepare_data.py"), "image"], "prepare-image.log")
        else:
            state('reusing_previously_validated_image_data', image_data_prepared=True)
        wait_for(lambda: (ROOT / "data/prepared/vstar_repeat32.json").exists(),
                 "waiting_for_independent_vstar_benchmark", "vstar-process.json")
        def gpus_ready():
            try:
                assert_idle(["1", "4"] + ([] if shared_judge else selection["gpus"].split(",")))
                return True
            except RuntimeError:
                return False
        wait_for(gpus_ready, "waiting_for_training_and_judge_gpus")
        if shared_judge:
            state('reusing_shared_local_judge')
            with urllib.request.urlopen(f"http://127.0.0.1:{selection['port']}/v1/models", timeout=10) as response:
                models = json.load(response)['data']
            if not any(model.get('id') == 'judge' and model.get('root') == selection['model_path']
                       for model in models):
                raise RuntimeError('Shared judge does not match the selected model')
        else:
            state("starting_selected_local_judge")
            judge_log = (ROOT / "logs/judge.log").open("a")
            judge = subprocess.Popen([sys.executable, str(ROOT / "scripts/serve_judge.py")],
                                     cwd=REPO, stdout=judge_log, stderr=subprocess.STDOUT, start_new_session=True)
            for attempt in range(120):
                if judge.poll() is not None:
                    raise RuntimeError("Selected local judge failed to start; see judge.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{selection['port']}/v1/models", timeout=5) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(10)
            else:
                raise RuntimeError("Judge did not become healthy within 20 minutes")
        name = (previous['full_run'].removesuffix('-full') if previous.get('full_run')
                else "image-reproduction-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
        if not completed_training(name + '-smoke', 1):
            state("running_one_step_training_smoke", training_started=True)
            run([sys.executable, str(ROOT / "scripts/run_training.py"), "--mode", "smoke", "--name", name + "-smoke"],
                "train-smoke.log")
        else:
            state('reusing_completed_image_smoke', smoke_training_passed=True, training_smoke_passed=True)
        smoke_actor = ROOT / "outputs" / (name + "-smoke") / "checkpoints/global_step_1/actor"
        if not list(smoke_actor.glob("model_world_size_*_rank_*.pt")):
            raise RuntimeError("Smoke process exited without a model checkpoint")
        def training_gpus_ready():
            try:
                assert_idle(["1", "4"])
                return True
            except RuntimeError:
                return False
        wait_for(training_gpus_ready, "waiting_for_smoke_workers_to_release_gpus")
        state("running_700_step_image_rl", smoke_training_passed=True, full_run=name + "-full")
        validation = str(ROOT / "data/prepared/vstar.json")
        if not completed_training(name + '-full', 700):
            run([sys.executable, str(ROOT / "scripts/run_training.py"), "--mode", "full", "--name", name + "-full",
                 "--validation", validation], "train-full.log")
        checkpoint = ROOT / "outputs" / (name + "-full") / "checkpoints/global_step_700/actor"
        if not list(checkpoint.glob("model_world_size_*_rank_*.pt")):
            raise RuntimeError("Full run exited without the expected step-700 checkpoint")
        state("merging_final_checkpoint", image_rl_700_steps_completed=True)
        model = checkpoint / "huggingface"
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO / "verl_agents"))
        run([str(PYTHON), str(ROOT / 'scripts/validate_optimizer_update.py'),
             '--run', str(ROOT / 'outputs' / (name + '-full')),
             '--source-model', str(ROOT / 'models/PyVision-Image-7B-SFT'),
             '--minimum-step', '700', '--no-wait',
             '--report', str(ROOT / 'manifests/image-final-optimizer-update-validation.json')],
            'validate-image-final-optimizer.log', env=env)
        run([str(PYTHON), str(REPO / "verl_agents/scripts/model_merger.py"), "--backend", "fsdp",
             "--local_dir", str(checkpoint), "--hf_model_path", str(ROOT / "models/PyVision-Image-7B-SFT")],
            "merge-checkpoint.log", env=env)
        run([str(PYTHON), str(ROOT / 'scripts/validate_merged_model.py'),
             '--model', str(model), '--source-model', str(ROOT / 'models/PyVision-Image-7B-SFT'),
             '--report', str(ROOT / 'manifests/image-final-merged-model-validation.json')],
            'validate-image-final-merged-model.log', env=env)
        state("starting_official_image_evaluation", final_checkpoint=str(model))
        evaluate_suite(model, ["vstar", "hrbench-4k", "hrbench-8k", "dynamath", "mathverse", "mathvision", "wemath", "tirbench"], name + "-official")
        state("image_baseline_training_and_evaluation_complete", image_main_evaluation_completed=True,
              vstar_avg32_evaluation_completed=True, paper_results_reproduced=False)
        plan = json.loads((ROOT / "preparation-plan.json").read_text()) if (ROOT / "preparation-plan.json").exists() else {}
        if plan.get("continue_video_after_image"):
            for asset in ["video-rl-data", "video-sft-model"]:
                wait_for(lambda a=asset: (ROOT / "manifests" / (a + "-prepared-assets.json")).exists(),
                         "waiting_for_" + asset, "preparation-queue-process.json")
            run([sys.executable, str(ROOT / "scripts/prepare_data.py"), "video"], "prepare-video.log")
            video_name = (previous['video_full_run'].removesuffix('-full') if previous.get('video_full_run')
                          else "video-reproduction-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
            state('preparing_video_training', video_full_run=video_name + '-full')
            checkpoint_root = Path(previous.get('checkpoint_storage', ROOT / 'outputs/checkpoints'))
            wait_for(training_gpus_ready, "waiting_for_video_training_gpus")
            if not completed_training(video_name + '-smoke', 1):
                state("running_video_training_smoke")
                run([sys.executable, str(ROOT / "scripts/run_training.py"), "--kind", "video", "--mode", "smoke",
                     "--name", video_name + "-smoke", '--checkpoint-dir',
                     str(checkpoint_root / (video_name + '-smoke'))], "train-video-smoke.log")
            smoke = ROOT / "outputs" / (video_name + "-smoke") / "checkpoints/global_step_1/actor"
            if not list(smoke.glob("model_world_size_*_rank_*.pt")):
                raise RuntimeError("Video smoke did not produce a model checkpoint")
            wait_for(training_gpus_ready, "waiting_for_video_smoke_workers")
            state("running_700_step_video_rl")
            wait_for_parallel_video(video_name + '-full')
            if not completed_training(video_name + '-full', 700):
                run([sys.executable, str(ROOT / "scripts/run_training.py"), "--kind", "video", "--mode", "full",
                     "--name", video_name + "-full", '--save-freq', '20', '--checkpoint-dir',
                     str(checkpoint_root / (video_name + '-full'))], "train-video-full.log")
            video_checkpoint = ROOT / "outputs" / (video_name + "-full") / "checkpoints/global_step_700/actor"
            if not list(video_checkpoint.glob("model_world_size_*_rank_*.pt")):
                raise RuntimeError("Video run did not produce its step-700 checkpoint")
            state("merging_video_checkpoint", video_rl_700_steps_completed=True)
            run([str(PYTHON), str(ROOT / 'scripts/validate_optimizer_update.py'),
                 '--run', str(ROOT / 'outputs' / (video_name + '-full')),
                 '--source-model', str(ROOT / 'models/PyVision-Video-7B-SFT'),
                 '--minimum-step', '700', '--no-wait',
                 '--report', str(ROOT / 'manifests/video-final-optimizer-update-validation.json')],
                'validate-video-final-optimizer.log', env=env)
            run([str(PYTHON), str(REPO / "verl_agents/scripts/model_merger.py"), "--backend", "fsdp",
                 "--local_dir", str(video_checkpoint), "--hf_model_path", str(ROOT / "models/PyVision-Video-7B-SFT")],
                "merge-video-checkpoint.log", env=env)
            run([str(PYTHON), str(ROOT / 'scripts/validate_merged_model.py'),
                 '--model', str(video_checkpoint / 'huggingface'),
                 '--source-model', str(ROOT / 'models/PyVision-Video-7B-SFT'),
                 '--report', str(ROOT / 'manifests/video-final-merged-model-validation.json')],
                'validate-video-final-merged-model.log', env=env)
            evaluate_suite(video_checkpoint / "huggingface", ["vsibench"], video_name + "-official")
            state("image_video_baselines_finished_review_results", status="baselines_finished",
                  video_main_evaluation_completed=True, paper_results_reproduced=False,
                  note="Review measured metrics, protocol ambiguities and failures before scheduling original-paper ablations. Missing Video SFT data still prevents a from-scratch full reproduction.")
    except BaseException as exc:
        state("failed", status="failed", error=str(exc))
        raise
    finally:
        stop_owned(judge)
        if judge_log:
            judge_log.close()


if __name__ == "__main__":
    with (ROOT / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
