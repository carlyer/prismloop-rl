#!/usr/bin/env python3
"""Rebuild PrismLoop-RL's original SVG artwork and measured-result figure."""
import csv
from render_flowcharts import render_flowcharts
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'

BANNER = '''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="440" viewBox="0 0 1440 440" role="img" aria-labelledby="title desc">
<title id="title">PrismLoop-RL</title>
<desc id="desc">Tool-Aware Reinforcement Learning for Visual Agents. Image and video, Python interaction, and GRPO.</desc>
<defs>
<linearGradient id="bg" x2="1" y2="1"><stop stop-color="#0b1224"/><stop offset="1" stop-color="#17213b"/></linearGradient>
<linearGradient id="prism" x1="0" y1="1" x2="1" y2="0"><stop stop-color="#59e0c5"/><stop offset="1" stop-color="#9c94ff"/></linearGradient>
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#c3d6ff" stroke-opacity=".05"/></pattern>
</defs>
<rect width="1440" height="440" rx="24" fill="url(#bg)"/>
<rect width="1440" height="440" rx="24" fill="url(#grid)"/>
<path d="M0 438H1440" stroke="url(#prism)" stroke-width="4"/>
<g font-family="Inter,DejaVu Sans,Arial,sans-serif">
<rect x="66" y="53" width="8" height="8" rx="4" fill="#59e0c5"/>
<text x="87" y="63" fill="#b2c1dc" font-size="14" letter-spacing="3">VISUAL AGENT LEARNING</text>
<text x="64" y="174" fill="#f4f7ff" font-size="78" font-weight="750" letter-spacing="-3">PrismLoop<tspan fill="#72e1cf">-RL</tspan></text>
<text x="68" y="226" fill="#c5d1e7" font-size="27">Tool-Aware Reinforcement Learning</text>
<text x="68" y="265" fill="#c5d1e7" font-size="27">for Visual Agents</text>
<g font-size="14" fill="#cbd6eb">
<rect x="68" y="315" width="144" height="38" rx="19" fill="#ffffff" fill-opacity=".06" stroke="#40516e"/>
<text x="140" y="339" text-anchor="middle">IMAGE + VIDEO</text>
<rect x="224" y="315" width="166" height="38" rx="19" fill="#ffffff" fill-opacity=".06" stroke="#40516e"/>
<text x="307" y="339" text-anchor="middle">PYTHON FEEDBACK</text>
<rect x="402" y="315" width="100" height="38" rx="19" fill="#ffffff" fill-opacity=".06" stroke="#40516e"/>
<text x="452" y="339" text-anchor="middle">GRPO</text>
</g>
<text x="68" y="398" fill="#8496b5" font-size="14">A PyVision-RL reproduction and extension project</text>
</g>
<g transform="translate(1120 208)">
<circle r="143" fill="none" stroke="#8b9cbe" stroke-opacity=".18"/>
<circle r="113" fill="none" stroke="#59e0c5" stroke-opacity=".25" stroke-dasharray="4 12"/>
<path d="M-116-83 A143 143 0 0 1 132-55" fill="none" stroke="#a098ff" stroke-width="4"/>
<path d="M126-68L136-50L116-52" fill="none" stroke="#a098ff" stroke-width="4"/>
<path d="M116 83 A143 143 0 0 1-132 55" fill="none" stroke="#59e0c5" stroke-width="4"/>
<path d="M-126 68L-136 50L-116 52" fill="none" stroke="#59e0c5" stroke-width="4"/>
<path d="M0-85L88 64H-88Z" fill="url(#prism)" fill-opacity=".12" stroke="url(#prism)" stroke-width="3"/>
<path d="M0-85V16L88 64M0 16L-88 64" fill="none" stroke="url(#prism)" stroke-width="2"/>
<circle cy="16" r="13" fill="#c7fff0"/>
</g>
</svg>'''



def render_results():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    with (ROOT / 'results/benchmark-results.csv').open() as f:
        rows = list(csv.DictReader(f))
    scores = {(r['run'], r['benchmark']): float(r['score_percent']) for r in rows}
    specs = [
        ('VStar', 'vstar', 'image', 'official-local-image-checkpoint-20261020T092620Z-step20-vstar'),
        ('HRBench-4K', 'hrbench-4k', 'image', 'official-local-image-checkpoint-20261020T092620Z-step20-hrbench-4k'),
        ('VSI macro', 'vsibench', 'video', 'research-eval-video-initial-checkpoint-20261021-vsibench'),
    ]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    fig.patch.set_facecolor('#f6f8fc')
    for ax, (title, benchmark, kind, initial) in zip(axes, specs):
        ax.set_facecolor('#f6f8fc')
        start = scores[initial, benchmark]
        values = [scores[f'research-eval-{kind}-{method}-20261021-{benchmark}', benchmark] - start
                  for method in ['core', 'ppo2']]
        bars = ax.barh([1, 0], values, height=.4, color=['#7785b4', '#16a58c'])
        for b, value in zip(bars, values):
            ax.text(value+.07, b.get_y()+b.get_height()/2, f'+{value:.2f}', va='center',
                    color='#17213b', fontsize=12, fontweight='bold')
        ax.set_title(title, loc='left', fontsize=15, fontweight='bold', pad=14, color='#17213b')
        ax.set_yticks([1, 0], ['Core', 'Core + PPO2'])
        ax.set_xlim(0, 3.4)
        ax.set_ylim(-.55, 1.55)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel('Gain over starting checkpoint (pp)', color='#52617c', fontsize=10)
        ax.tick_params(axis='both', length=0, labelcolor='#52617c')
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#dce3ef', linewidth=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle('Observed gains after 40 additional rollout steps', x=.035, y=.98,
                 ha='left', color='#17213b', fontsize=17, fontweight='bold')
    fig.text(.035, .025, 'Single seed · includes continued-training effects · PPO2 uses twice the optimizer updates',
             fontsize=10, color='#63718b')
    fig.subplots_adjust(left=.11, right=.98, top=.73, bottom=.24, wspace=.66)
    fig.savefig(ASSETS / 'short-run-gains.png', dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / 'banner.svg').write_text(BANNER)
    render_flowcharts()
    render_results()
    print('Rendered banner, bilingual flowcharts and short-run-gains.png')


if __name__ == '__main__':
    main()
