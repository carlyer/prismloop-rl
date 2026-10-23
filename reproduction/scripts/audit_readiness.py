"""Report actual preparation state separately from training/evaluation results."""
import json
import hashlib
from pathlib import Path
import time

from download_preparation_queue import ROOT, write


def read(relative):
    path = ROOT / relative
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    image_metrics = read('manifests/image-training-metrics.json')
    image_updates = image_metrics.get('completed_updates', [])
    latest_image_update = max(image_updates, key=lambda row: row['step']) if image_updates else {}
    image_log_archive = read('outputs/' + image_metrics.get('run', 'pending') + '/runtime-log-archive.json')
    training = {}
    for name in ['image-rl-data', 'image-sft-model', 'video-rl-data', 'video-sft-model', 'image-sft-data', 'video-sft-data']:
        record = read('manifests/' + name + '.json')
        prepared = read('manifests/' + name + '-prepared-assets.json')
        training[name] = {'download_status': prepared.get('status', record.get('status', 'missing')),
                          'repo_id': record.get('repo_id'), 'revision': record.get('revision')}
        if name == 'video-sft-data':
            training[name]['training_samples_available'] = False
            training[name]['note'] = 'Metadata-only official release; 44K training examples unavailable'
    benchmarks = {}
    for name in ['vstar', 'hrbench-4k', 'hrbench-8k', 'dynamath', 'mathverse', 'mathvision', 'mathvision-mini', 'wemath', 'tirbench', 'vsibench',
                 'mathvista', 'mmmu', 'charxiv-reasoning', 'charxiv-descriptive',
                 'tempcompass-mc', 'tempcompass-yn', 'tempcompass-caption-matching', 'mvbench-public-19']:
        record = read('manifests/benchmark-' + name + '.json')
        if name == 'vstar':
            record = read('manifests/vstar.json')
        if name == 'vsibench':
            record = read('manifests/vsibench-local.json')
        benchmarks[name] = {'status': record.get('status', 'queued'),
                            'samples': record.get('samples', record.get('questions')),
                            'protocol_note': record.get('protocol_note', record.get('annotation_version_note'))}
        if name == 'vsibench' or name.startswith(('tempcompass-', 'mvbench')):
            media = read('manifests/media-validation-' + name + '.json')
            source = ROOT / 'data/evaluation' / (name + '.json')
            media_status = media.get('status', 'pending')
            checked, failures, source_matches = None, None, None
            if media and source.exists():
                content = source.read_bytes()
                source_matches = (hashlib.sha256(content).hexdigest() ==
                                  media.get('source_sha256', {}).get(name))
                if source_matches:
                    paths = {row['video_path'] for row in json.loads(content)}
                    passed_paths = {row['path'] for row in media.get('results', [])
                                    if row.get('status') == 'passed'}
                    checked = len(paths & passed_paths)
                    failures = len(paths - passed_paths)
                    if failures:
                        media_status = 'incomplete_or_failed_media_coverage'
                else:
                    media_status = 'stale_validation_inputs_changed'
            elif media:
                media_status = 'prepared_input_missing'
            benchmarks[name].update(media_decode_check=media_status,
                                    media_failures=failures,
                                    checked_unique_videos=checked,
                                    media_validation_source_sha256_matches=source_matches,
                                    validation_run_benchmarks=sorted(media.get('source_sha256', {})),
                                    validation_run_unique_videos=media.get('unique_videos'))
    result = {'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'phase': 'reproduction and preparation; research exploration deferred',
              'training_assets': training, 'evaluation_datasets': benchmarks,
              'training_pipeline': read('status.json'),
              'image_periodic_validation_archive': read('outputs/' + image_metrics.get('run', 'pending') +
                                                        '/training-validation/periodic-index.json'),
              'periodic_validation_archive_checks': read('manifests/training-validation-archive-checks.json'),
              'rl_training_validation_media_id_observation': read('manifests/rl-training-validation-media-id-audit.json'),
              'video_training_validation_preparation': {
                  'preparation': read('manifests/video-training-validation400.json'),
                  'native_input_checks': read('manifests/video-training-validation400-checks.json'),
                  'full_config_resolution': read('manifests/video-full-validation-config-check.json')},
              'numerical_reward_output_compatibility_guard': {
                  'patch': read('manifests/numeric-reward-guard-patch.json'),
                  'validation': read('manifests/numeric-reward-guard-validation.json')},
              'historical_video_smoke_retained_source_coverage': read('manifests/video-early-smoke-source-coverage.json'),
              'tool_subprocess_cleanup': {
                  'reclamation': read('manifests/expired-image-tool-reclamation.json'),
                  'future_process_patch': read('manifests/tool-cleanup-patch.json'),
                  'cpu_validation': read('manifests/tool-cleanup-validation.json'),
                  'current_image_observer': read('manifests/expired-image-tools-observer.json')},
              'image_formal_training_progress': {
                  'run': image_metrics.get('run'),
                  'completed_updates_observed': latest_image_update.get('step', 0),
                  'target_updates': 700,
                  'observed_utc': image_metrics.get('observed_utc'),
                  'gradient_norm': latest_image_update.get('metrics', {}).get('actor/grad_norm'),
                  'source': image_metrics.get('source'),
                  'snapshot_only': True},
              'image_training_log_archive': {
                  k: v for k, v in image_log_archive.items() if k != 'files'},
              'image_source_snapshot_observation': read('manifests/image-source-snapshot-observation.json'),
              'image_first_internal_validation': read('manifests/image-validation-step20.json'),
              'image_saved_rollout_integrity': {
                  k: v for k, v in read('manifests/image-rollout-integrity.json').items() if k != 'files'},
              'early_video_training_smoke': read('manifests/video-early-training-smoke.json'),
              'image_smoke_checkpoint_conversion': {
                  'merger_exit': read('manifests/image-smoke-merge-exit-status.json'),
                  'loaded_model': read('manifests/image-smoke-merged-model-validation.json'),
                  'engineering_preflight_only': True},
              'first_formal_image_step': {k: v for k, v in read('manifests/image-first-full-update.json').items()
                                            if k in ['status', 'run', 'observed_utc', 'checkpoint_tracker',
                                                     'backward_gradient_norm', 'full_700_steps_complete',
                                                     'initial_optimizer_learning_rate', 'nonzero_weight_update_verified']},
              'published_scheduler_audit': read('manifests/zero-warmup-first-step-audit.json'),
              'later_image_weight_update_verification': read('manifests/image-nonzero-weight-update-validation.json'),
              'sft_controls': {kind: read('manifests/' + kind + '-sft-control-status.json') for kind in ['image', 'video']},
              'released_rl_controls': {kind: read('manifests/' + kind + '-rl-control-status.json') for kind in ['image', 'video']},
              'released_rl_reference_assets': {
                  kind: {
                      'download': {k: v for k, v in read('manifests/' + kind + '-rl-model-prepared-assets.json').items()
                                   if k in ['status', 'repo_id', 'revision', 'target', 'lfs_checksums_verified']},
                      'weight_index_validation': {
                          k: v for k, v in read('manifests/' + kind + '-rl-reference-weight-index-validation.json').items()
                          if k not in ['shards']}}
                  for kind in ['image', 'video']},
              'mathvision_mini_sft_control': {
                  'process_metadata': read('manifests/official-image-sft-control-mathvision-mini-process.json'),
                  'summary': read('outputs/official-image-sft-control-mathvision-mini/summary.json')},
              'evaluation_judge_audit': {
                  k: v for k, v in read('manifests/evaluation-judge-format-audit.json').items() if k != 'results'},
              'native_training_judge_request_audit': {
                  k: v for k, v in read('manifests/native-training-judge-audit.json').items()
                  if k not in ['request_failures', 'mismatches']},
              'inflight_evaluation_integrity': read('manifests/inflight-evaluation-audit.json'),
              'evaluation_context_budget_classification_fix': {
                  'patch': read('manifests/evaluation-context-budget-patch.json'),
                  'classifier_validation': read('manifests/context-budget-classification-validation.json'),
                  'recovery_validation': read('manifests/sft-context-recovery-validation.json'),
                  'control_recovery': {kind: read('manifests/' + kind + '-sft-context-recovery-status.json')
                                       for kind in ['image', 'video']},
                  'application_scope': 'New invocations use the corrected classifier; in-flight processes retain '
                                       'their loaded version. Approved cached results are adopted on a subsequent run.'},
              'image_action_error_static_audit': read('manifests/image-step13-action-error-audit.json'),
              'expanded_observer_coordination_reload': read('manifests/expanded-observer-reload.json'),
              'baseline_reference_preparation': {
                  'catalog': str(ROOT / 'BASELINE_METHODS.txt'),
                  'official_readmes': len(read('references/baseline-method-readiness.json').get('canonical_methods', [])),
                  'checkpoint_metadata_count': read('references/baseline-method-readiness.json').get('model_metadata_verified', 0),
                  'documentation_execution_requirements': str(ROOT / 'references/baseline-execution-requirements.json'),
                  'supplemental_documents': len(read('references/baseline-supplemental-documents.json').get('records', [])),
                  'full_code_audit_and_local_comparison': 'pending'},
              'additional_asset_queue': read('manifests/preparation-queue-status.json'),
              'supplementary_benchmark_queue': read('manifests/extra-benchmarks-status.json'),
              'goal_acceptance': str(ROOT / 'GOAL_AND_ACCEPTANCE.txt'),
              'method_component_checks': read('manifests/method-components-preflight.json').get('status', 'pending'),
              'evaluator_check': read('outputs/evaluation-protocol-self-test/summary.json'),
              'full_paper_reproduced': False,
              'unresolved': ['Video SFT 44K samples not available in inspected official repository',
                             'Exact MathVision/MathVerse/DynaMath paper preprocessing subsets require confirmation',
                             'Local judge and GPU hardware differ from paper settings',
                             'Original ablations are configured but not executed']}
    write(ROOT / 'manifests/full-reproduction-readiness.json', result)
    lines = ['PyVision-RL 复现准备状态（自动生成）', result['updated_utc'],
             '以下是准备状态，不是已完成训练或已复现论文分数的声明。', '', '训练资产：']
    lines += [f"{name}: {row['download_status']}" for name, row in training.items()]
    lines += ['', '评测数据：']
    lines += [f"{name}: {row['status']}; 样本数 {row['samples']}" +
              (f"; 视频解码 {row['media_decode_check']}" if 'media_decode_check' in row else '')
              for name, row in benchmarks.items()]
    lines += ['', '完整论文复现：尚未完成。', '原方法组件检查：' + result['method_component_checks'],
              '详细协议与公开材料缺口：METHODS_AND_PROTOCOL.txt', '原论文消融：experiment-matrix.json',
              '当前训练阶段：' + str(result['training_pipeline'].get('stage')),
              'Image 正式更新已观测：' + str(latest_image_update.get('step', 0)) + '/700（快照）',
              'Video 提前单步工程检查：' + result['early_video_training_smoke'].get('status', 'pending'),
              'Video 400 题训练内验证输入检查：' +
              result['video_training_validation_preparation']['native_input_checks'].get('status', 'pending') +
              '（本地固定抽样；完整 Bench 另行运行）',
              '数值无效输出兼容补丁：' +
              result['numerical_reward_output_compatibility_guard']['validation'].get('status', 'pending') +
              '（奖励边界行为差异另记，保留有效数值公式及首数字解析）',
              '训练 judge 原生请求诊断：' +
              result['native_training_judge_request_audit'].get('status', 'pending') +
              '（受控样例；真实轨迹格式警告另存，不表示全部训练判分有效）']
    (ROOT / 'PREPARATION_STATUS.txt').write_text('\n'.join(lines) + '\n')
    print('Refreshed reproduction readiness inventory')


if __name__ == '__main__':
    main()
