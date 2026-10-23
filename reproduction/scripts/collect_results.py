"""Collect actual full evaluations; keep paper numbers and partial probes separate."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from download_preparation_queue import ROOT, write
from evaluation_evidence import audit_records


def model_stage(config):
    """Identify the evaluated weights, independently of the output directory name."""
    roots = {Path(model['root']).resolve() for model in config.get('policy_models', []) if model.get('root')}
    if len(roots) != 1:
        return 'unverified_model_origin'
    root, = roots
    if root == (ROOT / 'models/Qwen2.5-VL-7B-Instruct').resolve():
        return 'official_qwen_instruct'
    selection = ROOT / 'manifests/local-checkpoint-evaluation-selection.json'
    if selection.exists():
        for chosen in json.loads(selection.read_text()).get('checkpoints', {}).values():
            if root == Path(chosen['model']).resolve():
                return 'local_intermediate_rl'
    for kind in ['Image', 'Video']:
        for suffix, stage in [('SFT', 'official_sft'), ('RL', 'official_released_rl')]:
            if root == (ROOT / 'models' / f'PyVision-{kind}-7B-{suffix}').resolve():
                return stage
    try:
        relative = root.relative_to((ROOT / 'outputs').resolve())
    except ValueError:
        # Checkpoints can live outside outputs through a directory symlink.
        for folder in (ROOT / 'outputs').iterdir():
            if root == (folder / 'checkpoints/global_step_700/actor/huggingface').resolve():
                return 'local_trained_rl'
    else:
        if relative.parts[-4:] == ('checkpoints', 'global_step_700', 'actor', 'huggingface'):
            return 'local_trained_rl'
    return 'unverified_model_origin'


def vsi_task_metrics(report):
    totals = {}
    direction_means = []
    for category, metrics in report['categories'].items():
        task = 'object_rel_direction' if category.startswith('object_rel_direction_') else category
        if category.startswith('object_rel_direction_'):
            direction_means.append(metrics['score_mean'])
        entry = totals.setdefault(task, {'count': 0, 'score_sum': 0.0})
        entry['count'] += metrics['count']
        entry['score_sum'] += metrics['count'] * metrics['score_mean']
    expected = {'object_counting', 'object_size_estimation', 'room_size_estimation', 'object_abs_distance',
                'object_rel_direction', 'object_rel_distance', 'obj_appearance_order', 'route_planning'}
    if set(totals) != expected:
        raise ValueError('Full VSI evaluation must contain all eight canonical tasks')
    tasks = {name: dict(count=m['count'], score_mean=m['score_sum']/m['count']) for name, m in totals.items()}
    if len(direction_means) != 3:
        raise ValueError('VSI requires all three direction difficulty groups')
    tasks['object_rel_direction']['score_mean'] = sum(direction_means)/3
    return dict(tasks=tasks, task_macro_mean=sum(m['score_mean'] for m in tasks.values())/8,
                definition='Equal mean across easy/medium/hard direction scores, then equal mean across eight tasks; released PyVision per-answer scores')


def main():
    status = json.loads((ROOT / 'status.json').read_text())
    reference = json.loads((ROOT / 'references/paper-reference-metrics.json').read_text())
    paper = dict(zip(reference['table1_benchmarks'], reference['table1_percent']['PyVision-Image']))
    paper['vsibench'] = reference['table2_vsibench_percent']['PyVision-Video']
    results, probes, incomplete = [], [], []
    for summary in sorted((ROOT / 'outputs').glob('*/summary.json')):
        invocation = summary.parent / 'invocation.json'
        if not invocation.exists():
            continue
        report = json.loads(summary.read_text())
        config = json.loads(invocation.read_text())
        if report.get('self_test_only') or not report.get('full_dataset_evaluated'):
            probes.append(str(summary.parent))
            continue
        benchmark = config['benchmark']
        evidence_error = None
        try:
            evidence = audit_records(summary.parent, config, report,
                                     ROOT / 'data/evaluation' / (benchmark + '.json'))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            evidence = dict(status='failed', error=str(exc))
            evidence_error = str(exc)
        canonical_vsi = vsi_task_metrics(report) if benchmark == 'vsibench' else None
        stage = model_stage(config)
        sft_control = stage == 'official_sft'
        paper_score = (reference.get('appendix_image_sft_percent', {}).get(benchmark)
                       if sft_control else paper.get(benchmark))
        if stage == 'official_qwen_instruct':
            paper_score = dict(zip(reference['table1_benchmarks'],
                                   reference['table1_percent']['Qwen2.5-VL-7B'])).get(benchmark)
        row = dict(run=summary.parent.name, benchmark=benchmark, samples=report['samples'],
                   status=report['status'], failures=report['failures'],
                   infrastructure_failures=report.get('infrastructure_failures', 0),
                   micro_accuracy_percent=100 * report['score_mean'],
                   category_macro_percent=100 * report['category_macro_mean'],
                   output=str(summary.parent), policy=config.get('policy_models'),
                   supplementary_metric_protocol=config.get('supplementary_metric_protocol'),
                   model_stage=stage, is_local_training_run=stage in {'local_trained_rl', 'local_intermediate_rl'},
                   paper_reported_percent=paper_score, strict_paper_protocol_match=False)
        row['record_evidence'] = evidence
        if canonical_vsi:
            row['raw_subtype_macro_percent'] = row['category_macro_percent']
            row['category_macro_percent'] = 100 * canonical_vsi['task_macro_mean']
            row['vsi_eight_task_metrics'] = canonical_vsi
            write(summary.parent / 'official-task-metrics.json', canonical_vsi)
            from native_vsi_metrics import rescore
            native = rescore(summary.parent, report)
            row['native_vsi_overall_percent'] = native['metrics']['overall']
            row['native_vsi_metrics'] = native
        if evidence_error or stage == 'unverified_model_origin':
            row['status'] = 'incomplete_evidence'
        if row['status'] != 'complete' or row['infrastructure_failures']:
            incomplete.append(row)
        else:
            results.append(row)
    # A missing summary is not evidence that a run is live or that its records
    # are valid. Inventory requested full-dataset runs without accepting scores.
    unfinalized = []
    for invocation in sorted((ROOT / 'outputs').glob('*/invocation.json')):
        if (invocation.parent / 'summary.json').exists():
            continue
        config = json.loads(invocation.read_text())
        if (not config.get('benchmark') or config.get('self_test_only')
                or config.get('limit') is not None or config.get('question_ids') is not None
                or not config.get('full_dataset_samples')
                or config.get('samples') != config['full_dataset_samples']):
            continue
        records = sum(bool(re.fullmatch(r'.+-repeat\d+\.json', p.name))
                      for p in invocation.parent.iterdir())
        unfinalized.append(dict(run=invocation.parent.name, benchmark=config['benchmark'],
                                status='no_summary_observed', observed_record_files=records,
                                expected_records=config['samples'] * config['repeat'],
                                output=str(invocation.parent),
                                scores_accepted=False, process_liveness_checked=False))
    observed_utc = datetime.now(timezone.utc).isoformat()
    record = dict(status='results_collected_not_automatically_accepted', observed_utc=observed_utc,
                  training_state=status,
                  completed_full_evaluations=results, incomplete_evaluations=incomplete,
                  unfinalized_full_dataset_runs=unfinalized,
                  excluded_partial_or_simulated_runs=probes,
                  judge_matches_paper=False, exact_paper_numbers_reproduced=False,
                  protocol_notes=['Local 32B judge and L20X training differ from paper',
                                  'VSI headline uses category macro; micro accuracy retained',
                                  'MathVerse/DynaMath/MathVision subset ambiguities retained',
                                  'MVBench-public-19 excludes NTU pose media',
                                  'CharXiv local judge substitution and TempCompass rule-only variant labeled'])
    write(ROOT / 'manifests/collected-results.json', record)
    lines = ['PyVision-RL 本机训练与评测结果',
             '仅纳入完整评测；工程小样本与模拟响应单独排除。',
             '当前训练阶段：' + status.get('stage', 'unknown'),
             '本地 judge：Qwen2.5-VL-32B-Instruct；与论文 72B judge 不同。', '',
             'VSI 同时列出发布 PyVision 判分和作者原生规则判分；二者不混用。',
             '运行 | Bench | 题数 | 逐题平均(%) | 类别平均(%) | VSI原生(%) | 论文报告(%)']
    for row in results:
        lines.append(f"{row['run']} | {row['benchmark']} | {row['samples']} | {row['micro_accuracy_percent']:.3f} | "
                     f"{row['category_macro_percent']:.3f} | {row.get('native_vsi_overall_percent')} | {row['paper_reported_percent']}")
    if not results:
        lines.append('暂无完整 Bench 结果。')
    lines += ['', '统计时间（UTC）：' + observed_utc,
              '未完成的评测任务数：' + str(len(incomplete) + len(unfinalized)),
              '已有汇总但未通过完整验收：' + str(len(incomplete)),
              '尚无汇总的完整集评测任务：' + str(len(unfinalized))]
    for row in unfinalized:
        lines.append(f"{row['run']} | 已观察逐题记录 {row['observed_record_files']}/{row['expected_records']}；"
                     '未计入正式成绩，文件数不代表结果有效或进程存活。')
    lines += [
              '排除的工程/模拟测试数：' + str(len(probes)),
              '分数可否严格对照论文须结合 METHODS_AND_PROTOCOL.txt 中的数据划分与判分差异判断。']
    (ROOT / 'REPRODUCTION_RESULTS.txt').write_text('\n'.join(lines) + '\n')
    print(f'Collected {len(results)} full evaluations; {len(probes)} partial/simulated runs excluded')


if __name__ == '__main__':
    main()
