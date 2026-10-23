"""Prove a research update relative to its resumed checkpoint, not SFT."""
import argparse
import json
import os
from pathlib import Path

from download_preparation_queue import write
from run_research_experiments import SPEC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=SPEC, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    before_root = Path(SPEC[args.kind]['checkpoint'])/'actor'
    after_root = args.checkpoint/'actor'
    report = dict(initial_checkpoint=str(before_root.parent), checkpoint=str(args.checkpoint), checks=[])
    for rank in range(2):
        filename = f'model_world_size_2_rank_{rank}.pt'
        before = torch.load(before_root/filename, mmap=True, map_location='cpu', weights_only=False)
        after = torch.load(after_root/filename, mmap=True, map_location='cpu', weights_only=False)
        for key in ['model.language_model.norm.weight', 'model.language_model.layers.0.self_attn.q_proj.weight']:
            a = getattr(before[key], '_local_tensor', before[key]).reshape(-1)[:8192].float()
            b = getattr(after[key], '_local_tensor', after[key]).reshape(-1)[:8192].float()
            if not torch.isfinite(b).all():
                raise RuntimeError('Nonfinite research weight sample')
            report['checks'].append(dict(rank=rank, parameter=key, changed_elements=int((a!=b).sum()), max_abs_change=float((a-b).abs().max())))
        filename = f'optim_world_size_2_rank_{rank}.pt'
        before_opt = torch.load(before_root/filename, mmap=True, map_location='cpu', weights_only=False)
        after_opt = torch.load(after_root/filename, mmap=True, map_location='cpu', weights_only=False)
        def steps(opt):
            return [float(getattr(s['step'], '_local_tensor', s['step']).item()) for s in list(opt['state'].values())[:10] if 'step' in s]
        bs, ats = steps(before_opt), steps(after_opt)
        if not bs or len(bs)!=len(ats) or not all(a>b for a,b in zip(ats,bs)):
            raise RuntimeError('Optimizer did not advance relative to resumed checkpoint')
        report[f'rank_{rank}_optimizer_before'] = bs
        report[f'rank_{rank}_optimizer_after'] = ats
    if not any(c['changed_elements'] for c in report['checks']):
        raise RuntimeError('Research weight samples did not change')
    report.update(status='passed', scope='FP32 samples on both ranks and optimizer counters relative to resumed baseline; not all parameters')
    write(args.report, report)


if __name__ == '__main__':
    main()
