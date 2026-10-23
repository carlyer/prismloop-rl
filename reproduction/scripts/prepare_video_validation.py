"""Prepare a pinned 400-question VSI subset for native Video training validation.

The paper gives the subset size, but not the authors' exact selected IDs/seed.
This local selection is explicitly recorded and remains separate from full Bench.
"""
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'data/evaluation/vsibench.json'
TRAIN = ROOT / 'data/PyVision-Video-RL-Data/pyvision_video_rl_data.json'
MEDIA = ROOT / 'manifests/media-validation-vsibench.json'
NATIVE = ROOT.parent / 'verl_agents/verl/utils/dataset/rl_dataset_wo_mm_hint.py'
TEMPLATE = ROOT.parent / 'verl_agents/verl/utils/dataset/rl_system_prompt_template.json'
NUMERIC = ROOT.parent / 'verl_agents/verl/utils/reward_score/rule_match_verify.py'
OUTPUT = ROOT / 'data/prepared/video_val_dataset_vsibench400.json'
RAW = ROOT / 'data/prepared/video_val_dataset_vsibench400.raw.json'
RECEIPT = ROOT / 'manifests/video-training-validation400.json'
SEED = 42
NUMERICAL_TYPES = {'object_counting', 'object_size_estimation',
                   'room_size_estimation', 'object_abs_distance'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_write(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Refusing to replace a different frozen validation artifact: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def native_transfer(sample_metadata):
    tree = ast.parse(NATIVE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'transfer_to_rl_form_video')
    namespace = dict(json=json, sample_frames_from_video=sample_metadata)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NATIVE), 'exec'), namespace)
    return namespace['transfer_to_rl_form_video']


def verify_prepared():
    report = json.loads(RECEIPT.read_text())
    if report['status'] != 'prepared' or report['samples'] != 400:
        raise RuntimeError('Invalid Video validation receipt')
    for item in report['input_bindings'] + report['artifacts']:
        if digest(item['path']) != item['sha256']:
            raise RuntimeError(f'Video validation input/artifact changed: {item["path"]}')
    rows = json.loads(OUTPUT.read_text())
    if len(rows) != 400 or len({row['extra_info']['index'] for row in rows}) != 400:
        raise RuntimeError('Video validation IDs/count changed')
    if any(row['env_name'] != 'pyvision_gym_wo_video_hint' or
           row['mm_hint']['hint_type'] != 'video' or
           not Path(row['mm_hint']['hint_path']).is_file() for row in rows):
        raise RuntimeError('Video validation media/schema invalid')
    return report


def prepare():
    if RECEIPT.exists():
        report = verify_prepared()
        print(json.dumps(dict(status='verified_existing', samples=report['samples'], output=str(OUTPUT))))
        return report
    rows = json.loads(BENCH.read_text())
    training = json.loads(TRAIN.read_text())
    if len(rows) != 5130 or len(training) != 15000:
        raise RuntimeError('Unexpected official input counts')
    train_scenes = {Path(row['mm_hint']['hint_path']).stem for row in training}
    bench_scenes = {Path(row['video_path']).stem for row in rows}
    intersection = sorted(train_scenes & bench_scenes)
    if intersection:
        raise RuntimeError('Training/Bench scene IDs intersect; review before selecting validation')
    media_report = json.loads(MEDIA.read_text())
    if (media_report['status'] != 'passed' or media_report['failures'] or
            media_report['source_sha256']['vsibench'] != digest(BENCH)):
        raise RuntimeError('VSI media validation is incomplete or stale')
    metadata = {item['path']: item for item in media_report['results'] if item['status'] == 'passed'}
    spec = importlib.util.spec_from_file_location('isolated_numeric_validation', NUMERIC)
    numeric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(numeric)
    selected_positions = sorted(random.Random(SEED).sample(range(len(rows)), 400))
    selected = [rows[position] for position in selected_positions]
    raw = []
    for row in selected:
        is_numeric = row['question_type'] in NUMERICAL_TYPES
        options = row.get('options')
        if is_numeric:
            parsed = numeric._parse_number(row['ground_truth'])
            if options is not None or parsed is None or not math.isfinite(parsed):
                raise RuntimeError('Invalid numerical benchmark row')
            question = row['question']
        else:
            if not options or str(row['ground_truth']) not in {'A', 'B', 'C', 'D'}:
                raise RuntimeError('Invalid choice benchmark row')
            question = row['question']+'\n'+'\n'.join(options)
        raw.append(dict(id=str(row['id']), question=question, answer=row['ground_truth'],
                        ability='spatial_reasoning', data_source='vsi_numerical' if is_numeric else 'vsi',
                        video_path=row['video_path']))

    def cached_metadata(path, num_frames=0):
        if num_frames != 0:
            raise RuntimeError('Validation preparation must not pre-sample video frames')
        value = metadata[path]
        if Path(path).stat().st_size != value['size']:
            raise RuntimeError('Video size changed since actual decode validation')
        return dict(frames=[], video_length=value['frames'], fps=value['fps'],
                    height=value['dimensions'][0], width=value['dimensions'][1])

    prepared = native_transfer(cached_metadata)(copy.deepcopy(raw), str(TEMPLATE))
    # Retain native converter output, including its historical split="train" metadata label.
    # This file is passed only as data.val_files; it is never added to RL training examples.
    if len(prepared) != 400:
        raise RuntimeError('Native validation converter changed the count')
    immutable_write(RAW, raw)
    immutable_write(OUTPUT, prepared)
    report = dict(status='prepared', observed_utc=datetime.now(timezone.utc).isoformat(),
                  samples=400, output=str(OUTPUT), raw=str(RAW),
                  selection=dict(method='Uniform random sample without replacement, then source-order sort',
                                 seed=SEED, selected_source_positions=selected_positions,
                                 selected_ids=[row['id'] for row in selected],
                                 authors_exact_subset_recovered=False,
                                 limitation='Paper Appendix D.1 states 400 random VSI-Bench questions; '
                                            'authors exact IDs and random seed were not found in the pinned release.'),
                  overlap_audit=dict(training_questions=len(training), training_scene_ids=len(train_scenes),
                                     full_bench_questions=len(rows), full_bench_scene_ids=len(bench_scenes),
                                     scene_id_intersection=intersection,
                                     scope='Exact media basename IDs in released RL corpus versus fixed VSI snapshot. '
                                           'Does not prove absence from SFT/pretraining or perceptual duplicates.'),
                  data_sources=dict(Counter(row['data_source'] for row in prepared)),
                  question_types=dict(Counter(row['question_type'] for row in selected)),
                  unique_videos=len({row['video_path'] for row in selected}),
                  input_bindings=[dict(path=str(path), sha256=digest(path))
                                  for path in [BENCH, TRAIN, MEDIA, NATIVE, TEMPLATE, NUMERIC]],
                  artifacts=[dict(path=str(path), sha256=digest(path)) for path in [RAW, OUTPUT]],
                  preparation='Execute exact released transfer_to_rl_form_video AST with metadata from '
                              'the already passed first/last-frame decode report; dimensions, FPS, frame '
                              'counts and current media sizes bound and checked. No initial frames sampled.',
                  native_split_metadata_preserved='train, as hardcoded by released converter; '
                                                  'the entire artifact is only data.val_files',
                  score_scope='Native training monitor: vsi uses original text judge; vsi_numerical uses '
                              'released first-number MRA with malformed-prediction zero guard. '
                              'Source-level means are not final canonical eight-task VSI Bench accuracy.',
                  training_data_modified=False, full_benchmark_input_modified=False,
                  intended_validation_frequency=20, trajectories_per_question=1,
                  not_a_completed_evaluation=True)
    immutable_write(RECEIPT, report)
    verify_prepared()
    print(json.dumps(dict(status='prepared', samples=400, output=str(OUTPUT),
                          data_sources=report['data_sources'], unique_videos=report['unique_videos'])))
    return report


if __name__ == '__main__':
    prepare()
