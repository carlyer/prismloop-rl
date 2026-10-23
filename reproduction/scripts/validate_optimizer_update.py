"""Inspect a saved optimizer and compare sampled actor weights with official SFT."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from download_preparation_queue import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--minimum-step', type=int, default=2)
    parser.add_argument('--expect-zero-initial', action='store_true')
    parser.add_argument('--no-wait', action='store_true', help='Fail if a finished run did not save the required step')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    report = dict(status='waiting_for_checkpoint', run=str(args.run), minimum_step=args.minimum_step,
                  scope='Exact saved optimizer LR/step and small FP32 actor-weight samples; not all-parameter verification')
    write(args.report, report)
    tracker = args.run / 'checkpoints/latest_checkpointed_iteration.txt'
    try:
        while True:
            try:
                step = int(tracker.read_text().strip())
            except (FileNotFoundError, ValueError):
                step = 0
            if step >= args.minimum_step:
                break
            if args.no_wait:
                raise RuntimeError('Finished training did not save the required checkpoint')
            time.sleep(30)
        import torch
        from safetensors import safe_open

        actor = args.run / f'checkpoints/global_step_{step}/actor'
        optimizer_path, = actor.glob('optim_world_size_*_rank_0.pt')
        model_path, = actor.glob('model_world_size_*_rank_0.pt')
        optimizer = torch.load(optimizer_path, map_location='cpu', mmap=True, weights_only=False)
        lrs = [float(group['lr']) for group in optimizer['param_groups']]
        steps = [float(getattr(s['step'], '_local_tensor', s['step']).item())
                 for s in list(optimizer['state'].values())[:10] if 'step' in s]
        if not lrs or not all(lr > 0 for lr in lrs) or not steps:
            raise RuntimeError('Optimizer LR/state did not advance as expected')
        weights = torch.load(model_path, map_location='cpu', mmap=True, weights_only=False)
        index = json.loads((args.source_model / 'model.safetensors.index.json').read_text())
        checks = []
        for key in ['model.language_model.norm.weight', 'lm_head.weight',
                    'model.language_model.layers.0.self_attn.q_proj.weight',
                    'model.language_model.embed_tokens.weight']:
            tensor = weights[key]
            local = getattr(tensor, '_local_tensor', tensor)
            placements = getattr(tensor, 'placements', None)
            if placements and not all(p.is_replicate() or (p.is_shard() and p.dim == 0) for p in placements):
                raise RuntimeError('Unsupported shard layout for comparison: ' + str(placements))
            source_key = key if key in index['weight_map'] else key.replace('model.language_model.', 'model.', 1)
            with safe_open(args.source_model / index['weight_map'][source_key], framework='pt', device='cpu') as f:
                original = f.get_slice(source_key)
                if local.ndim == 1:
                    before, after = original[:local.shape[0]].float(), local.float()
                    selection = 'all rank-0 local elements'
                else:
                    rows = [0, 32, 198, 220]
                    before = torch.stack([original[i, :256] for i in rows]).float()
                    after = local[rows, :256].float()
                    selection = 'global rows 0,32,198,220; first 256 columns; rank 0'
            difference = (after - before).abs()
            if not torch.isfinite(before).all().item() or not torch.isfinite(after).all().item():
                raise RuntimeError('Non-finite sampled model weights: ' + key)
            checks.append(dict(key=key, source_key=source_key, selection=selection,
                               checked_elements=difference.numel(), changed_elements=int((difference > 0).sum()),
                               maximum_absolute_change=float(difference.max())))
        changed = any(c['changed_elements'] for c in checks)
        report.update(checkpoint_step=step, optimizer_learning_rates_after_scheduler=lrs,
                      sampled_optimizer_steps=steps, weight_checks=checks,
                      nonzero_weight_update_verified=changed)
        if args.expect_zero_initial:
            if step != 1 or changed:
                raise RuntimeError('Expected the published scheduler initial zero-LR step')
            report.update(status='verified_zero_initial_lr_step', initial_optimizer_learning_rate=0.0)
        else:
            if not changed or not all(s >= max(2, args.minimum_step) for s in steps):
                raise RuntimeError('Sampled weights or optimizer steps do not prove the requested update count')
            report['status'] = 'passed_nonzero_weight_update'
    except Exception as exc:
        report.update(status='failed', error_class=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['observed_utc'] = datetime.now(timezone.utc).isoformat()
        write(args.report, report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
