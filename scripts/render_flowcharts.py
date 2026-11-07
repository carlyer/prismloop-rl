#!/usr/bin/env python3
"""Render bilingual, dependency-free SVG diagrams from the published snapshot.

Method labels describe the archived configuration, not arbitrary launcher settings.
Counts, scores and continuation progress come from results/; rerender after updates.
"""
import csv
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
INK, MUTED, TEAL, PURPLE = '#17243d', '#596b85', '#13866f', '#7160a9'


class Diagram:
    def __init__(self, title, subtitle, height, zh=False):
        self.zh = zh
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{height}" viewBox="0 0 1440 {height}" role="img" aria-labelledby="title desc" lang="{'zh-CN' if zh else 'en'}">
<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#8392aa"/></marker>
<marker id="loop" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="{TEAL}"/></marker></defs>
<rect width="1440" height="{height}" rx="24" fill="#f5f7fb"/>
<path d="M36 24H1404" stroke="{TEAL}" stroke-width="4"/>
<g font-family="Noto Sans CJK SC,Microsoft YaHei,PingFang SC,DejaVu Sans,sans-serif">''']
        self.text(40, 59, 'PRISMLOOP-RL / ' + ('研究流程' if zh else 'RESEARCH WORKFLOW'), 13, TEAL, True)
        self.text(40, 106, title, 32, INK, True)
        self.text(40, 142, subtitle, 17)

    def text(self, x, y, value, size=18, color=MUTED, bold=False):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}">{escape(str(value))}</text>')

    def lines(self, x, y, values, size=18, color=MUTED, gap=30):
        for i, value in enumerate(values):
            self.text(x, y + i * gap, value, size, color)

    def box(self, x, y, w, h, number, title, lines, accent=TEAL, dark=False):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{"#17243d" if dark else "#ffffff"}" stroke="#dae2ee"/>')
        self.text(x + 22, y + 32, number, 13, '#72dfc8' if dark else accent, True)
        self.text(x + 22, y + 69, title, 23, '#f5f8ff' if dark else INK, True)
        self.lines(x + 22, y + 105, lines, 18, '#cbd8ed' if dark else MUTED)

    def arrow(self, path, loop=False, dashed=False):
        dash = 'stroke-dasharray="7 6"' if dashed else ''
        self.parts.append(f'<path d="{path}" fill="none" stroke="{TEAL if loop else "#8392aa"}" stroke-width="2.3" stroke-linejoin="round" {dash} marker-end="url(#{"loop" if loop else "arrow"})"/>')

    def save(self, name):
        (ASSETS / name).write_text('\n'.join(self.parts + ['</g></svg>']) + '\n', encoding='utf-8')


def architecture(zh):
    def tr(en, cn):
        return cn if zh else en

    d = Diagram(tr('From visual interaction to policy improvement', '从视觉交互到策略更新'),
                tr('Implemented core pipeline · Image and Video are trained as separate models',
                   '已实现的核心训练流程 · Image 与 Video 分别训练，保留上游 GRPO 算法'), 1000, zh)
    d.box(40, 180, 420, 245, '01 / INPUT', tr('Visual question + source', '视觉问题与数据来源'), tr([
        'Image: visible to policy and Python',
        'Video: VideoReader in Python',
        'Video frames are sampled on demand',
        '8 sibling trajectories per question',
    ], ['图像：模型与 Python 均可访问', '视频：Python 通过 VideoReader 访问',
        '视频初始不提供帧；按需采样并显示', '每个问题采样 8 条同组轨迹']))
    d.box(510, 180, 420, 245, '02 / INTERACT', tr('Qwen2.5-VL ↔ Python', 'Qwen2.5-VL ↔ Python'), tr([
        'Reason → <code> → execute', 'Read text / images / errors → reason',
        'Persistent execution state across turns', 'Repeat until final answer / stop condition',
    ], ['推理 → <code> → 执行 Python', '读取文本 / 图像 / 错误 → 继续推理',
        '执行状态在多轮交互间保留', '循环直到最终答案或停止条件']), dark=True)
    d.box(980, 180, 420, 245, '03 / TRAJECTORY', tr('Answer + interaction trace', '最终答案与完整交互轨迹'), tr([
        'Model actions + tool observations', 'Final <answer> contains a boxed value',
        'Action mask marks trainable tokens', 'Keep sibling IDs and source labels',
    ], ['包含模型动作与工具返回的观察', '最终答案：<answer> 中的 boxed 答案',
        'action mask 标记参与训练的 token', '保留问题组 ID 与数据源标签']))
    d.arrow('M460 300H504')
    d.arrow('M930 300H974')
    d.arrow('M1190 425V514')
    d.box(980, 520, 420, 260, '04 / REWARD → ADVANTAGE', tr('Score, then compute GRPO', '先计算奖励，再计算 GRPO 优势'), tr([
        'Answer: judge or final numeric MRA', 'Tool: success, nonempty, deduplicated',
        '0.1 × at most 2 credits; answer gate', 'Group advantages before selection',
        'No reward-std normalization; no KL loss',
    ], ['答案奖励：judge 或最终数值 MRA', '工具奖励：成功、非空、代码与反馈去重',
        '每次 0.1，最多 2 次；保留答案正确性门控', '在筛选前计算组内相对优势',
        '不按奖励标准差归一化；不加 KL loss']))
    d.box(510, 520, 420, 260, '05 / FILTER → SELECT', tr('Source-aware selection', '按数据源配额选择问题组'), tr([
        'Filter invalid / zero-variance candidates', 'Rank by reward std within each source',
        'Best-effort quotas; prefer whole groups', 'Select 128; replenish candidates if needed',
        'Final group may split; no advantage redo',
    ], ['过滤无效轨迹 / 零方差组等候选', '各来源内部按奖励标准差排序',
        '尽量满足来源配额，优先保留完整组', '选择 128 条轨迹；不足时补充生成',
        '末组可能拆分；不重新计算优势']))
    d.box(40, 520, 420, 260, '06 / UPDATE', tr('GRPO + FSDP', 'GRPO + FSDP 更新策略'), tr([
        'Learning rate 1e-6; visual tower frozen', 'Core: PPO epochs=1 → 1 update / step',
        'PPO2: epochs=2 → 2 updates / step', 'Same rollout steps ≠ same compute',
        'Updated policy generates next rollouts',
    ], ['学习率 1e-6；冻结视觉编码器', 'Core：PPO epochs=1，每步更新 1 次',
        'PPO2：epochs=2，每步更新 2 次', '相同 rollout 步数不等于相同计算预算',
        '新策略继续生成下一批轨迹']), accent=PURPLE)
    d.arrow('M980 650H936')
    d.arrow('M510 650H466')
    d.arrow('M250 520V470H720V431', loop=True)
    d.text(285, 458, tr('Updated policy → next interaction batch', '策略更新 → 下一批交互采样'), 16, TEAL, True)
    d.parts.append('<rect x="40" y="820" width="1360" height="140" rx="16" fill="#e9eef7"/>')
    d.text(62, 853, tr('Implementation boundaries', '实现边界与共享执行环境'), 20, INK, True)
    d.lines(62, 887, tr([
        'Numeric extraction is a Video factor; Image sources use a judge. Numeric tool gating retains the upstream threshold.',
        'Duplicate code OR feedback receives no new tool credit. This heuristic does not establish causal usefulness.',
        'Shared runtime: spawn workers and bounded cleanup. Model / optimizer / RNG / dataloader state supports continuation.',
    ], ['数值答案提取用于 Video 数值样本；Image 来源使用 judge。数值样本的工具门控保留上游阈值。',
        '代码或反馈任一重复，均不再获得工具奖励；该启发式规则不证明工具调用的因果贡献。',
        '共享执行环境：spawn 工作进程与有界清理。模型、优化器、随机数和数据加载器状态共同支持续训。']), 17, gap=27)
    d.save('architecture.zh-CN.svg' if zh else 'architecture.svg')


def experiments(zh, rows, continuation, evaluations):
    def tr(en, cn):
        return cn if zh else en

    d = Diagram(tr('Experiment branches and the path to measured results', '实验分支与评测证据流程'),
                tr(f"Evidence synchronized {continuation['synchronized_on']} · Solid arrows: published path; dashed arrows: pending work",
                   f"证据同步日期 {continuation['synchronized_on']} · 实线：已发布流程；虚线：待完成工作"), 1160, zh)
    d.box(40, 180, 420, 220, 'A / MATCHED START', tr('Restore the starting state', '从同一完整状态启动'), tr([
        'Image: step 20   |   Video: step 120', 'Model + optimizer + RNG + dataloader',
        'Matched start within each modality',
    ], ['Image：第 20 步；Video：第 120 步', '恢复模型、优化器、随机数和数据加载器',
        '同一模态的比较分支使用同一起点']))
    d.box(510, 180, 420, 220, 'B / COMPLETED SHORT RUNS', tr('Core and PPO2: +40 steps', 'Core 与 PPO2：各新增 40 步'), tr([
        'Image → 60   |   Video → 160', '4 trained branches; 8 measured results',
        'PPO2 doubles optimizer updates / step',
    ], ['Image → 60 步；Video → 160 步', '4 个训练分支，8 项扩展实验评测',
        'PPO2 每个 rollout 步的参数更新翻倍']), dark=True)
    d.box(980, 180, 420, 220, 'C / EVALUATE + AUDIT', tr('Export → audited records', '模型导出 → 评测记录核验'), tr([
        'Merge weights; audit updates and export', 'Fixed evaluator → record coverage audit',
        'VStar avg@32; VSI macro ≠ native VSI',
    ], ['合并权重，检查参数更新与导出', '固定评测协议 → 逐题记录与覆盖率核验',
        'VStar avg@32；VSI 宏平均与 native 分列']))
    d.arrow('M460 290H504')
    d.arrow('M930 290H974')
    d.arrow('M1190 400V464')
    d.arrow('M250 400V464', dashed=True)
    d.arrow('M720 400V464', dashed=True)
    d.box(40, 470, 420, 285, 'D / CONTROLLED ABLATIONS', tr('Factorial design · pending', '同预算组件消融 · 尚未完成'), tr([
        'Image: 2² quota × tool', 'Video: 2³ numeric × quota × tool',
        '+40 steps; epochs=1; shared runtime', 'Reuse 2 full-core cells; 10 new runs',
        'Original controls: evaluation queued', 'Remaining component cells: queued',
    ], ['Image：2²，配额 × 工具奖励', 'Video：2³，数值 × 配额 × 工具奖励',
        '均新增 40 步，epochs=1，共享执行修复', '复用 2 个 full-core；另需 10 个训练任务',
        '原方法同期对照：归档为待评测', '其余组件组合：归档为排队中']), accent=PURPLE)
    pending = sum(not r['score_percent'] for r in evaluations)
    d.box(510, 470, 420, 285, 'E / LONG CONTINUATIONS', tr('Resume all 4 branches', '沿 4 个已有分支继续训练'), tr([
        'Image 60 / Video 160 → total step 700', 'Milestones: 200 / 400 / 700',
        'Resume full state and branch history', f'{pending} milestone evaluations have no score',
        'No measured step-700 result published', 'Pending exports / evaluations → C',
    ], ['Image 60 / Video 160 → 总计 700 步', '评测里程碑：200 / 400 / 700',
        '恢复完整状态，延续各自的训练历史', f'{pending} 项归档里程碑评测尚无成绩',
        '尚无已发布的 700 步实测结果', '待导出、评测与核验 → C']), accent=PURPLE)
    d.box(980, 470, 420, 285, 'F / PUBLISHED MEASUREMENTS', tr(f'{len(rows)} measured result rows', f'{len(rows)} 行已发布实测结果'), tr([
        'benchmark-results.csv', 'Includes official SFT / RL references', 'Core / PPO2 extension scores (%):',
    ], ['benchmark-results.csv', '包含官方 SFT / RL 权重参考成绩', 'Core / PPO2 扩展实验成绩（%）：']))
    scores = {(r['run'], r['benchmark']): float(r['score_percent']) for r in rows}
    for i, (label, kind, benchmark) in enumerate([
        ('VStar', 'image', 'vstar'), ('HRBench-4K', 'image', 'hrbench-4k'),
        ('HRBench-8K', 'image', 'hrbench-8k'), ('VSI macro', 'video', 'vsibench'),
    ]):
        # HRBench-8K was archived later; select by the method/run prefix.
        values = []
        for method in ['core', 'ppo2']:
            candidates = [v for (run, bench), v in scores.items()
                          if run.startswith((f'research-eval-{kind}-{method}-', f'supplementary-{kind}-{method}-')) and bench == benchmark]
            if len(candidates) != 1:
                raise ValueError(f'Expected one archived {kind}/{method}/{benchmark} score')
            values.append(candidates[0])
        d.text(1002, 664 + i * 23, label, 16, INK)
        d.text(1180, 664 + i * 23, f'{values[0]:.2f} / {values[1]:.2f}', 16, TEAL, True)
    d.parts.append('<rect x="40" y="790" width="1360" height="162" rx="16" fill="#e9eef7"/>')
    observed = continuation['observed_utc'].replace('T', ' ').replace('Z', ' UTC')
    d.text(62, 824, tr(f'Historical training snapshot · {observed}', f'历史训练快照 · {observed}'), 21, INK, True)
    jobs = {j['id']: j for j in continuation['jobs']}
    for i, key in enumerate(['image-ppo2', 'image-core', 'video-ppo2', 'video-core']):
        j = jobs[key]
        x = 62 + i * 335
        d.text(x, 859, key, 18, TEAL, True)
        d.text(x, 889, tr(f"Completed {j['last_completed_step']} / saved {j['latest_saved_step']}",
                          f"已完成 {j['last_completed_step']} / 已保存 {j['latest_saved_step']}"), 18, INK)
        d.text(x, 921, tr(f"Recorded: {j['status']}", f"当时状态：{j['status']}"), 16)
    d.text(40, 993, tr('Interpretation limits', '结果解读边界'), 21, INK, True)
    d.lines(40, 1027, tr([
        'Historical progress is not live status. A checkpoint or queued evaluation is not a benchmark score.',
        'Short-run gains include continued training; original controls and full ablations are pending. Single seed, no significance claim.',
        'HRBench-8K has no starting-checkpoint score; PPO2 is below core there. Official RL references have unmatched budgets.',
        'The 700-step planning forecast is separate from measurements and is not plotted here.',
    ], ['历史快照不代表实时状态；保存 checkpoint 或评测排队均不等于已有 benchmark 成绩。',
        '短程收益包含继续训练的影响；同期原方法对照与完整消融未完成。当前为单种子实验，不能宣称统计显著。',
        'HRBench-8K 缺少起点成绩，且 PPO2 低于 core；官方发布 RL 权重的训练预算未与本实验对齐。',
        '700 步规划预测与实测结果独立保存，本图不将预测绘制成实验成绩。']), 17, gap=28)
    d.save('experiment-flow.zh-CN.svg' if zh else 'experiment-flow.svg')


def render_flowcharts():
    ASSETS.mkdir(exist_ok=True)
    with (ROOT / 'results/benchmark-results.csv').open() as f:
        rows = list(csv.DictReader(f))
    with (ROOT / 'results/continuation-evaluation-status.csv').open() as f:
        evaluations = list(csv.DictReader(f))
    continuation = json.loads((ROOT / 'results/continuation-training-snapshot.json').read_text())
    for zh in [False, True]:
        architecture(zh)
        experiments(zh, rows, continuation, evaluations)


if __name__ == '__main__':
    render_flowcharts()
    print('Rendered architecture and experiment-flow SVGs in English and Chinese')
