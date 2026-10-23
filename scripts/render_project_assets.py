#!/usr/bin/env python3
"""Rebuild PrismLoop-RL's original SVG artwork and measured-result figure."""
import csv
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

ARCHITECTURE = '''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="736" viewBox="0 0 1440 736" role="img" aria-labelledby="title desc">
<title id="title">PrismLoop-RL: from visual interaction to policy improvement</title>
<desc id="desc">An image or video question enters a policy and Python execution loop. Final-answer and bounded tool rewards guide source-aware trajectory selection, followed by GRPO updates with FSDP. Implemented extensions and evidence audits are shown separately from future work.</desc>
<defs>
<linearGradient id="panel" x2="1" y2="1"><stop stop-color="#17243d"/><stop offset="1" stop-color="#101a30"/></linearGradient>
<linearGradient id="accent" x2="1"><stop stop-color="#34c8b0"/><stop offset="1" stop-color="#a098ed"/></linearGradient>
<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#8290ac"/></marker>
<marker id="teal" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#25a890"/></marker>
<pattern id="dots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#dfe6f1"/></pattern>
</defs>
<rect width="1440" height="736" rx="22" fill="#f7f9fc"/>
<rect width="1440" height="736" rx="22" fill="url(#dots)" opacity=".6"/>
<g font-family="Inter,DejaVu Sans,Arial,sans-serif">
<text x="36" y="43" fill="#62718b" font-size="12" font-weight="700" letter-spacing="2.5">PRISMLOOP-RL / SYSTEM DESIGN</text>
<text x="36" y="82" fill="#17243d" font-size="28" font-weight="700">See. Reason. Execute. Learn.</text>
<text x="1404" y="78" fill="#62718b" font-size="14" text-anchor="end">Image + video agents · shared GRPO backbone</text>
<!-- Four stages -->
<rect x="36" y="120" width="186" height="296" rx="17" fill="#ffffff" stroke="#dce4ef"/>
<rect x="250" y="120" width="478" height="296" rx="17" fill="url(#panel)"/>
<rect x="756" y="120" width="300" height="296" rx="17" fill="#ffffff" stroke="#dce4ef"/>
<rect x="1084" y="120" width="320" height="296" rx="17" fill="#ffffff" stroke="#dce4ef"/>
<g font-size="11" font-weight="700" letter-spacing="1.8">
<text x="55" y="150" fill="#62718b">01 / PERCEIVE</text>
<text x="270" y="150" fill="#99b4d8">02 / INTERACT</text>
<text x="776" y="150" fill="#62718b">03 / SHAPE + SELECT</text>
<text x="1104" y="150" fill="#62718b">04 / OPTIMIZE</text>
</g>
<!-- Visual inputs -->
<rect x="56" y="179" width="70" height="57" rx="9" fill="#eaf0fb" stroke="#c9d6ec"/>
<path d="M65 225L83 205L94 216L105 205L117 225Z" fill="#98a9d7"/>
<circle cx="108" cy="192" r="5" fill="#afbbdf"/>
<rect x="135" y="179" width="67" height="57" rx="9" fill="#e5f6f0" stroke="#b6e2d3"/>
<path d="M161 195L180 207L161 219Z" fill="#34a78d"/>
<text x="55" y="272" fill="#17243d" font-size="17" font-weight="700">Visual question</text>
<g fill="#62718b" font-size="14"><text x="55" y="303">Image / video</text><text x="55" y="329">Source + target</text><text x="55" y="384">8 sibling trajectories</text></g>
<!-- Agent/tool loop -->
<text x="270" y="186" fill="#f3f7ff" font-size="21" font-weight="700">Policy ↔ Python environment</text>
<rect x="270" y="211" width="192" height="100" rx="11" fill="#223451" stroke="#405475"/>
<rect x="497" y="211" width="211" height="100" rx="11" fill="#223451" stroke="#405475"/>
<text x="286" y="239" fill="#c7f4e7" font-size="16" font-weight="700">Qwen2.5-VL</text>
<text x="286" y="266" fill="#c6d3e8" font-size="13">Reasoning + tool code</text>
<text x="286" y="291" fill="#91a7c7" font-size="12">&lt;code&gt; ... &lt;/code&gt;</text>
<text x="513" y="239" fill="#c7f4e7" font-size="16" font-weight="700">Python executor</text>
<text x="513" y="266" fill="#c6d3e8" font-size="13">Crop · inspect · calculate</text>
<text x="513" y="291" fill="#91a7c7" font-size="12">Text / images / errors</text>
<path d="M462 247H491" fill="none" stroke="#8290ac" stroke-width="2" marker-end="url(#arrow)"/>
<path d="M602 311V335H366V317" fill="none" stroke="#42c5ab" stroke-width="2" marker-end="url(#teal)"/>
<text x="484" y="357" fill="#a9c6d8" font-size="12" text-anchor="middle">Read feedback → continue reasoning</text>
<text x="270" y="389" fill="#d7e2f4" font-size="14">Final answer + full interaction trajectory</text>
<!-- Signals -->
<g fill="#edf8f4"><rect x="776" y="179" width="260" height="45" rx="9"/><rect x="776" y="238" width="260" height="45" rx="9"/><rect x="776" y="297" width="260" height="45" rx="9"/></g>
<g fill="#1c806c" font-size="14" font-weight="700"><text x="789" y="207">Final-answer reward</text><text x="789" y="266">Bounded tool bonus</text><text x="789" y="325">Source-aware selection</text></g>
<text x="776" y="383" fill="#62718b" font-size="13">Filter → replenish → select 128</text>
<!-- Updates -->
<text x="1104" y="192" fill="#17243d" font-size="22" font-weight="700">GRPO + FSDP</text>
<text x="1104" y="223" fill="#62718b" font-size="14">Selected rollout trajectories</text>
<path d="M1104 244H1384" stroke="#e0e6f0"/>
<g font-size="14" fill="#52617c"><text x="1104" y="273">Epochs = 1</text><text x="1384" y="273" text-anchor="end">1 update / step</text><text x="1104" y="302">Epochs = 2</text><text x="1384" y="302" text-anchor="end">2 updates / step</text></g>
<rect x="1104" y="331" width="280" height="43" rx="9" fill="#f0edfc"/>
<text x="1244" y="358" fill="#7061a4" font-size="14" font-weight="700" text-anchor="middle">Checkpoint + export validation</text>
<!-- Forward edges and policy feedback -->
<g stroke="#8290ac" stroke-width="2" fill="none" marker-end="url(#arrow)"><path d="M222 265H243"/><path d="M728 265H749"/><path d="M1056 265H1077"/></g>
<path d="M1244 416V451H489V424" fill="none" stroke="#25a890" stroke-width="2" marker-end="url(#teal)"/>
<rect x="693" y="438" width="288" height="25" rx="12" fill="#f7f9fc"/>
<text x="837" y="455" text-anchor="middle" fill="#19836e" font-size="13">Updated policy → next rollout batch</text>
<!-- Three implementation cards -->
<g fill="#ffffff" stroke="#dce4ef"><rect x="36" y="489" width="440" height="143" rx="15"/><rect x="500" y="489" width="440" height="143" rx="15"/><rect x="964" y="489" width="440" height="143" rx="15"/></g>
<g fill="#19a38b"><circle cx="63" cy="520" r="4"/><circle cx="527" cy="520" r="4"/><circle cx="991" cy="520" r="4"/></g>
<g fill="#17243d" font-size="17" font-weight="700"><text x="77" y="526">Answer parsing</text><text x="541" y="526">Task mixture</text><text x="1005" y="526">Tool execution</text></g>
<g fill="#62718b" font-size="14"><text x="57" y="557">Use the final numeric answer;</text><text x="57" y="580">retain the original MRA metric.</text><text x="521" y="557">Allocate source quotas; rank</text><text x="521" y="580">question groups within each source.</text><text x="985" y="557">Check success, nonempty feedback,</text><text x="985" y="580">code/output novelty, and bonus caps.</text></g>
<g fill="#8b98ae" font-size="11"><text x="57" y="610">research_answer.py</text><text x="521" y="610">research_selection.py</text><text x="985" y="610">research_tool_quality.py</text></g>
<!-- Evidence strip -->
<rect x="36" y="656" width="1368" height="49" rx="12" fill="#17243d"/>
<text x="57" y="687" fill="#73d9c5" font-size="13" font-weight="700" letter-spacing="1">EVIDENCE</text>
<text x="175" y="687" fill="#dce6f7" font-size="14">VStar · HRBench · VSI</text>
<text x="511" y="687" fill="#dce6f7" font-size="14">Weight-update audits · protocol checks · explicit ablation switches</text>
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
    (ASSETS / 'architecture.svg').write_text(ARCHITECTURE)
    render_results()
    print('Rendered banner.svg, architecture.svg and short-run-gains.png')


if __name__ == '__main__':
    main()
