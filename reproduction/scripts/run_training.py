#!/usr/bin/env python3
"""Launch isolated, explicitly gated algorithm ablations with shared runtime fixes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request
from runtime_env import configure_runtime

VARIANT_FEATURES = {
    'original': [], 'numeric_only': ['numeric'], 'quota_only': ['quota'],
    'tool_only': ['tool'], 'full': ['numeric', 'quota', 'tool'],
    'without_numeric': ['quota', 'tool'], 'without_quota': ['numeric', 'tool'],
    'without_tool': ['numeric', 'quota'], 'ppo2': ['numeric', 'quota', 'tool'],
}

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PYTHON = Path("/opt/pyvision-rl-venv/bin/python")


def resolve_validation(args):
    origin = 'explicit'
    receipt = None
    if not args.validation and args.kind == 'video' and args.mode == 'full' and not args.eval_only:
        from prepare_video_validation import OUTPUT, RECEIPT, verify_prepared
        verify_prepared()
        args.validation = str(OUTPUT)
        origin = 'fixed_local_400_question_vsi_training_monitor'
        receipt = dict(path=str(RECEIPT), sha256=hashlib.sha256(RECEIPT.read_bytes()).hexdigest())
    if not args.validation:
        return None
    path = Path(args.validation).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                origin=origin, preparation_receipt=receipt)


def assert_idle(devices):
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    usage = {line.split(",")[0].strip(): int(line.split(",")[1])
             for line in result.strip().splitlines()}
    busy = {d: usage.get(d, -1) for d in devices if d not in usage or usage[d] > 1024}
    if busy:
        raise RuntimeError(f"Selected GPUs are unavailable or occupied: {busy}; no existing process was stopped")


def build(args):
    smoke = args.mode == "smoke"
    model = (Path(args.checkpoint).resolve() if args.checkpoint else
             ROOT / "models" / f"PyVision-{args.kind.capitalize()}-7B-SFT")
    train = ROOT / "data/prepared" / f"{args.kind}_{'smoke' if smoke else 'train'}.json"
    val = Path(args.validation).resolve() if args.validation else ROOT / "data/prepared" / f"{args.kind}_smoke.json"
    name = args.name or f"{args.kind}-{args.mode}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out = ROOT / "outputs" / name
    opts = {
        "data.train_files": f"[{train}]", "data.val_files": f"[{val}]",
        "data.train_batch_size": 2 if smoke else 16,
        "data.max_prompt_length": 2048 if smoke else 32000,
        "data.max_response_length": 4096 if smoke else 20480,
        "data.return_raw_chat": True, "data.filter_overlong_prompts": False,
        "data.with_mm_hint": args.kind == "image",
        "+data.min_pixels": 3136, "+data.max_pixels": 262144 if smoke else 2000000,
        "+data.cache_dir": str(ROOT / "cache/data"),
        "+data.prompt_template_path": str(REPO / "verl_agents/verl/utils/dataset/rl_system_prompt_template.json"),
        "+data.gen_batch_size": 8 if smoke else 32,
        "+data.max_video_gen_batch_size": 8 if smoke else 16,
        "+data.gen_batch_size_align_method": "up_resample_image",
        "algorithm.adv_estimator": "grpo", "algorithm.kl_ctrl.kl_coef": 0.0,
        "algorithm.norm_adv_by_std_in_grpo": False,
        "algorithm.filter_groups.enable": True,
        "+algorithm.filter_groups.std_sort_enable": True,
        "algorithm.filter_groups.max_num_gen_batches": 12 if smoke else 0,
        "algorithm.filter_groups.metric": "[seq_reward,hasimage,trajlength,vtoken_images_num_consis,end_reason]",
        "+algorithm.filter_groups.end_reason_filter_reserve_names": "[DONE,EXCEED_MAX_TURNS,ERROR_IN_ACTION]",
        "actor_rollout_ref.model.path": str(model),
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 2 if smoke else 16,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.checkpoint.contents": "[model,hf_model,optimizer,extra]",
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.n": 4 if smoke else 8,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 6144 if smoke else 52480,
        "actor_rollout_ref.rollout.max_num_seqs": 8 if smoke else 32,
        "+actor_rollout_ref.rollout.mm_processor_kwargs": "{min_pixels:3136,max_pixels:" + str(262144 if smoke else 2000000) + "}",
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.65,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.free_cache_engine": False,
        "actor_rollout_ref.rollout.enable_chunked_prefill": False,
        "actor_rollout_ref.rollout.agent.activate_agent": True,
        "actor_rollout_ref.rollout.agent.max_vllm_videos": 0,
        "actor_rollout_ref.rollout.agent.single_response_max_tokens": 1024 if smoke else 10240,
        "actor_rollout_ref.rollout.agent.max_turns": args.max_tool_turns + 1,
        "actor_rollout_ref.rollout.agent.concurrent_workers": 4 if smoke else 32,
        "actor_rollout_ref.rollout.agent.tool_using_cumulative_reward_per_turn": 0.1,
        "+actor_rollout_ref.rollout.val_kwargs.max_turn_in_val": 30,
        "+actor_rollout_ref.rollout.val_kwargs.eval_enable": False,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.01,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "trainer.rollout_data_dir": str(out / "rollouts"),
        "trainer.the_first_batch_rollout_data_dir": str(out / "first_rollouts"),
        "trainer.logger": "[console]", "trainer.val_before_train": False,
        "trainer.n_gpus_per_node": len(args.gpus.split(",")), "trainer.nnodes": 1,
        "trainer.save_freq": 1 if smoke else 100,
        "+trainer.save_first_step": True,
        "trainer.test_freq": 20 if args.validation and not smoke else -1,
        "trainer.total_epochs": 100,
        "trainer.total_training_steps": args.steps or (2 if smoke else (int(Path(args.resume_from).name.rsplit("_",1)[1]) if args.resume_from else 0)+40),
        "trainer.max_actor_ckpt_to_keep": 2 if smoke else 5,
        "trainer.project_name": "pyvision-reproduction", "trainer.experiment_name": name,
        "trainer.default_local_dir": str(Path(args.checkpoint_dir).resolve()) if getattr(args, "checkpoint_dir", None) else str(out / "checkpoints"),
        "ray_init.num_cpus": 24,
    }
    if args.no_tool_reward:
        opts["actor_rollout_ref.rollout.agent.tool_using_cumulative_reward_per_turn"] = 0.0
    if args.no_std_sort:
        opts["+algorithm.filter_groups.std_sort_enable"] = False
    if args.std_normalization:
        opts["algorithm.norm_adv_by_std_in_grpo"] = True
    if args.optimizer_foreach != 'auto':
        opts['+actor_rollout_ref.actor.optim.foreach'] = args.optimizer_foreach == 'true'
    if getattr(args, 'save_freq', None) is not None:
        if args.save_freq < 1:
            raise ValueError('--save-freq must be positive')
        opts['trainer.save_freq'] = args.save_freq
    if getattr(args, 'resume_from', None):
        if smoke or args.eval_only:
            raise ValueError('--resume-from is for full training only')
        checkpoint = Path(args.resume_from).resolve(strict=True)
        if not checkpoint.name.startswith('global_step_'):
            raise ValueError('Resume directory must be a global_step checkpoint')
        step = int(checkpoint.name.removeprefix('global_step_'))
        if step >= opts['trainer.total_training_steps']:
            raise ValueError('Resume checkpoint must precede the target training step')
        if not (checkpoint / 'data.pt').is_file():
            raise ValueError('Resume checkpoint lacks saved dataloader state')
        for rank in range(len(args.gpus.split(','))):
            for prefix in ['model', 'optim', 'extra_state']:
                filename = f'{prefix}_world_size_{len(args.gpus.split(","))}_rank_{rank}.pt'
                if not (checkpoint / 'actor' / filename).is_file():
                    raise ValueError('Incomplete resume checkpoint: ' + filename)
        opts['trainer.resume_mode'] = 'resume_path'
        opts['trainer.resume_from_path'] = str(checkpoint)
    if args.eval_only:
        if not args.validation:
            raise ValueError("--eval-only requires an independent --validation dataset")
        opts.update({"trainer.val_before_train": True, "+trainer.val_only": True,
                     "trainer.resume_mode": "disable"})
    # Research variants use the same rollout/optimizer base settings. Changes
    # are explicit in the invocation and isolated from the stopped baseline.
    from collections import Counter
    counts = Counter(row['data_source'] for row in json.loads(train.read_text()))
    weights = {k: v / sum(counts.values()) for k, v in counts.items()}
    if 'quota' in VARIANT_FEATURES[args.research_variant]:
        opts['+algorithm.research_source_weights'] = '{' + ','.join(f'{k}:{v:.12f}' for k, v in sorted(weights.items())) + '}'
    opts['+trainer.research_save_first_update'] = True
    opts['trainer.max_actor_ckpt_to_keep'] = 1
    opts['trainer.project_name'] = 'pyvision-research'
    opts['actor_rollout_ref.rollout.agent.concurrent_workers'] = 16
    opts['actor_rollout_ref.actor.ppo_epochs'] = 2 if args.research_variant == 'ppo2' else 1
    opts['algorithm.filter_groups.max_num_gen_batches'] = 12
    def encode(v):
        return str(v).lower() if isinstance(v, bool) else str(v)
    cmd = [str(PYTHON), "-m", "verl.trainer.main_ppo"] + [f"{k}={encode(v)}" for k, v in opts.items()]
    return cmd, out, train, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research-variant", choices=list(VARIANT_FEATURES), default="original")
    p.add_argument("--kind", choices=["image", "video"], default="image")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--gpus", default="1,4")
    p.add_argument("--ray-temp-dir", default="/tmp/pyvision-rl-ray", help="Separate Ray directory for concurrent engineering checks")
    p.add_argument('--optimizer-foreach', choices=['auto', 'true', 'false'], default='auto')
    p.add_argument("--steps", type=int)
    p.add_argument("--max-tool-turns", type=int, choices=[2, 4], default=4,
                   help="Paper tool budget; rollout includes one additional final-answer turn")
    p.add_argument("--name")
    p.add_argument("--validation", help="Independent held-out benchmark JSON; optional")
    p.add_argument("--checkpoint", help="Local merged Hugging Face model; default: official SFT")
    p.add_argument("--checkpoint-dir", help="Store large checkpoints outside the workspace; keep an output/checkpoints symlink")
    p.add_argument('--resume-from', help='Restore actor, optimizer, RNG and dataloader from an existing global_step directory')
    p.add_argument('--save-freq', type=int, help='Override checkpoint interval without changing RL optimization')
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--config-only", action="store_true")
    p.add_argument("--no-tool-reward", action="store_true")
    p.add_argument("--no-std-sort", action="store_true")
    p.add_argument("--std-normalization", action="store_true")
    args = p.parse_args()
    validation_input = resolve_validation(args)
    cmd, out, train, model = build(args)
    config = ROOT / "judge-config.json"
    env = configure_runtime(os.environ.copy())
    env['PYVISION_ABLATION_FEATURES'] = ','.join(VARIANT_FEATURES[args.research_variant])
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, VLLM_USE_V1="0", WANDB_MODE="offline",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
               LLM_AS_A_JUDGE_CONFIG_PATH=str(config), HYDRA_FULL_ERROR="1",
               PYTHONPATH=str(REPO / "verl_agents"),
               RAY_TMPDIR=args.ray_temp_dir, RAY_ADDRESS="local")
    if args.config_only:
        return subprocess.call(cmd + ["--cfg", "job"], cwd=REPO, env=env)
    if not train.is_file() or not (model / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("Download and prepare the official data and SFT checkpoint first")
    weight_index = json.loads((model / "model.safetensors.index.json").read_text())
    for filename in set(weight_index["weight_map"].values()):
        weight = model / filename
        if not weight.is_file() or Path(str(weight) + ".aria2").exists():
            raise RuntimeError(f"Model shard not completely downloaded: {filename}")
    judge = json.loads(config.read_text())
    with urllib.request.urlopen(judge["base_url"] + "/models", timeout=10) as r:
        if r.status != 200:
            raise RuntimeError("Judge is not healthy")
    assert_idle(args.gpus.split(","))
    out.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir:
        storage = Path(args.checkpoint_dir).resolve()
        storage.mkdir(parents=True, exist_ok=True)
        link = out / 'checkpoints'
        if link.is_symlink():
            if link.resolve() != storage:
                raise ValueError('Existing checkpoint symlink points to another storage directory')
        elif link.exists():
            raise ValueError('Refusing to replace an existing checkpoint directory')
        else:
            link.symlink_to(storage, target_is_directory=True)
        print('Checkpoint storage: ' + str(storage), flush=True)
    if (out / 'invocation.json').exists():
        history = out / 'attempt-history' / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        history.mkdir(parents=True, exist_ok=False)
        for filename in ['invocation.json', 'exit-status.json']:
            source = out / filename
            if source.exists():
                (history / filename).write_bytes(source.read_bytes())
        # A running retry must not retain the preceding attempt's terminal state.
        (out / 'exit-status.json').unlink(missing_ok=True)
    from snapshot_reproduction_sources import snapshot
    source_snapshot = snapshot(out, 'prelaunch')
    (out / "invocation.json").write_text(json.dumps({
        "args": vars(args), "command": cmd, "validation_enabled": bool(args.validation),
        "ablation_feature_flags": VARIANT_FEATURES[args.research_variant],
        "common_runtime_fixes": ["spawn tools","bounded cleanup","concurrency 16","invalid-numeric zero guard"],
        "validation_input": validation_input,
        "training_input": {"path": str(train), "bytes": train.stat().st_size,
                           "sha256": hashlib.sha256(train.read_bytes()).hexdigest()},
        "source_snapshot": source_snapshot,
        "engineering_weight_change_verification_required": not args.eval_only and args.mode == 'smoke' and (args.steps or 2) >= 2,
        "paper_hardware_reproduced": len(args.gpus.split(",")) == 8,
        "judge_selection": json.loads((ROOT / "judge-selection.json").read_text()),
        "judge_calibration": json.loads((ROOT / "manifests/judge-calibration.json").read_text()),
        "rollout_configuration_patch": json.loads((ROOT / "manifests/rollout-configuration-patch.json").read_text()) if (ROOT / "manifests/rollout-configuration-patch.json").exists() else None,
        "tensor_padding_patch": json.loads((ROOT / "manifests/tensor-padding-patch.json").read_text()) if (ROOT / "manifests/tensor-padding-patch.json").exists() else None,
        "training_runtime_patches": json.loads((ROOT / "manifests/training-runtime-patches.json").read_text()) if (ROOT / "manifests/training-runtime-patches.json").exists() else None,
        "optimizer_configuration_patch": json.loads((ROOT / "manifests/optimizer-configuration-patch.json").read_text()) if (ROOT / "manifests/optimizer-configuration-patch.json").exists() else None,
        "tool_cleanup_patch": json.loads((ROOT / "manifests/tool-cleanup-patch.json").read_text()) if (ROOT / "manifests/tool-cleanup-patch.json").exists() else None,
        "numeric_reward_guard_patch": json.loads((ROOT / "manifests/numeric-reward-guard-patch.json").read_text()) if (ROOT / "manifests/numeric-reward-guard-patch.json").exists() else None,
        "upstream_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    }, indent=2))
    print(shlex.join(cmd), flush=True)
    session_link = Path(args.ray_temp_dir) / 'ray/session_latest'
    preceding_session = session_link.resolve() if session_link.exists() else None
    result = subprocess.call(cmd, cwd=REPO, env=env)
    training_result = result
    log_archive = None
    if session_link.exists() and session_link.resolve() != preceding_session:
        try:
            from archive_training_logs import archive
            log_report = archive(session_link, out, final=True)
            log_archive = dict(status=log_report['status'], receipt=str(out / 'runtime-log-archive.json'))
        except Exception as exc:
            log_archive = dict(status='failed', error=str(exc))
        print('Training log archive: ' + json.dumps(log_archive), flush=True)
    weight_check = None
    if result == 0 and not args.eval_only and args.mode == 'smoke' and (args.steps or 2) >= 2:
        # The published scheduler has LR=0 on its first call, even with zero warmup.
        # A second step and a direct weight comparison verify a nonzero update.
        check_env = env.copy()
        check_env['CUDA_VISIBLE_DEVICES'] = ''
        weight_check = subprocess.call([
            str(PYTHON), str(ROOT / 'scripts/validate_optimizer_update.py'),
            '--run', str(out), '--source-model', str(model),
            '--minimum-step', str(args.steps or 2), '--no-wait',
            '--report', str(out / 'optimizer-update-validation.json'),
        ], cwd=REPO, env=check_env)
        if weight_check:
            result = weight_check
    (out / "exit-status.json").write_text(json.dumps({
        "returncode": result, "training_returncode": training_result,
        "weight_change_validation_returncode": weight_check,
        "runtime_log_archive": log_archive,
    }, indent=2))
    return result


if __name__ == "__main__":
    sys.exit(main())
