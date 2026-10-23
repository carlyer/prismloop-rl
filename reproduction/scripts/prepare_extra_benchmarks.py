"""Build held-out inputs, preserving official options, splits and instructions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

from download_preparation_queue import ROOT, digest, write
from prepare_benchmarks import canonical, embedded_image, parquet_rows, save, unpack

REFERENCES = ROOT / 'references/extra-benchmark-scorers'


def source(name):
    receipt = ROOT / 'manifests' / (name + '-prepared-assets.json')
    record = json.loads(receipt.read_text())
    if record['status'] != 'complete':
        raise RuntimeError('Download must finish first: ' + name)
    return record


def mathvista():
    record = source('mathvista')
    folder = ROOT / 'data/benchmarks/mathvista'
    rows = []
    for row in parquet_rows(next(folder.glob('data/testmini-*.parquet'))):
        image = embedded_image(row['decoded_image'], folder / 'prepared_images' / row['pid'])
        # query is supplied by MathVista and includes answer-type instructions/options.
        rows.append(canonical(row['pid'], row['query'], row['answer'], [image],
                              row['metadata']['task'], metric='mathvista',
                              choices=row['choices'], question_type=row['question_type'],
                              answer_type=row['answer_type'], precision=row['precision'],
                              dataset_metadata=row['metadata']))
    if len(rows) != 1000:
        raise RuntimeError('Unexpected MathVista testmini count')
    save('mathvista', rows, record, 'Official normalization/equality applied to last boxed answer; no GPT answer extractor.')


def mmmu():
    record = source('mmmu')
    folder = ROOT / 'data/benchmarks/mmmu'
    rows = []
    for path in sorted(folder.glob('*/validation-*.parquet')):
        for row in parquet_rows(path):
            images = []
            mapping = {}
            for number in range(1, 8):
                value = row.get(f'image_{number}')
                if value:
                    mapping[number] = len(images)
                    images.append(embedded_image(value, folder / 'prepared_images' / row['id'] / str(number)))
            def replace(match):
                return '<IMAGE_PLACE_HOLDER_' + str(mapping[int(match.group(1))]) + '>'
            question = re.sub(r'<image\s+(\d+)>', replace, row['question'])
            for image_index in range(len(images)):
                placeholder = f'<IMAGE_PLACE_HOLDER_{image_index}>'
                if placeholder not in question:
                    question += '\n' + placeholder
            options = ast.literal_eval(row['options']) if isinstance(row['options'], str) else row['options']
            if row['question_type'] == 'multiple-choice':
                question += '\n' + '\n'.join(chr(65+i) + '. ' + str(v) for i, v in enumerate(options))
                question += '\nAnswer with the option letter.'
            rows.append(canonical(row['id'], question, row['answer'], images, path.parent.name,
                                  metric='mmmu', question_type=row['question_type'], choices=options,
                                  difficulty=row['topic_difficulty']))
    if len(rows) != 900:
        raise RuntimeError('Unexpected MMMU validation count')
    save('mmmu', rows, record, 'Official parsing and correctness helpers; deterministic per-question seed for fallback choices.')


def charxiv():
    record = source('charxiv')
    folder = ROOT / 'data/benchmarks/charxiv'
    official = REFERENCES / 'charxiv'
    sys.path.insert(0, str(official / 'src'))
    from descriptive_utils import descriptive_query_helper
    from reasoning_utils import build_reasoning_queries
    reasoning = json.loads((official / 'data/reasoning_val.json').read_text())
    descriptive = json.loads((official / 'data/descriptive_val.json').read_text())
    queries = build_reasoning_queries(reasoning, str(folder / 'prepared_images'))
    images = {}
    for row in parquet_rows(folder / 'val.parquet'):
        identifier = Path(row['figure_path']).stem
        images[identifier] = embedded_image(row['image'], folder / 'prepared_images' / identifier)
        # Confirm the two public distributions describe the same held-out QA.
        if reasoning[identifier]['answer'] != row['reasoning_a']:
            raise RuntimeError('CharXiv annotation versions disagree')
    if set(images) != set(reasoning) or len(images) != 1000:
        raise RuntimeError('CharXiv media/annotation set mismatch')
    rr, dr = [], []
    for identifier, row in reasoning.items():
        query = queries[row['figure_id']]
        rr.append(canonical(identifier, query['question'], row['answer'], [images[identifier]],
                            str(row['inst_category']), metric='charxiv-reasoning',
                            raw_question=row['query'], inst_category=row['inst_category'], figure_id=identifier))
    for identifier, row in descriptive.items():
        for index, (qid, answer) in enumerate(zip(row['qids'], row['answers'])):
            dr.append(canonical(f'{identifier}_{index}', descriptive_query_helper(qid, row['subplot_loc']),
                                answer, [images[identifier]], str(qid), metric='charxiv-descriptive',
                                charxiv_qid=qid, figure_id=identifier))
    note = 'Official question/grading prompts; user-selected local 32B judge replaces GPT-4o; scores must be labeled accordingly.'
    save('charxiv-reasoning', rr, record, note)
    save('charxiv-descriptive', dr, record, note)


def save_video(name, rows, record, note):
    identifiers = [row['question_id'] for row in rows]
    if not rows or len(set(identifiers)) != len(rows):
        raise RuntimeError('Empty/duplicate video questions: ' + name)
    media = {row['video_path'] for row in rows}
    missing = [p for p in media if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(str(missing[:5]))
    output = ROOT / 'data/evaluation' / (name + '.json')
    write(output, rows)
    write(ROOT / 'manifests' / ('benchmark-' + name + '.json'), {
        'status': 'ready', 'name': name, 'samples': len(rows), 'unique_media': len(media),
        'data_file': str(output), 'sha256': digest(output), 'source': record,
        'protocol_note': note, 'evaluated': False, 'media_decode_check': 'pending'})
    print(name, len(rows), 'questions; all media paths verified', flush=True)


def tempcompass():
    record = source('tempcompass')
    folder = ROOT / 'data/benchmarks/tempcompass'
    unpack(folder / 'tempcompass_videos.zip', folder)
    # Index only the newly extracted local dataset, never the shared NAS tree.
    videos = {p.stem: str(p.resolve()) for p in folder.rglob('*.mp4')}
    for split, suffix in [('multi-choice', 'mc'), ('yes_no', 'yn'), ('caption_matching', 'caption-matching')]:
        rows = []
        for i, row in enumerate(parquet_rows(next((folder / split).glob('*.parquet')))):
            rows.append(dict(question_id=str(i), text=row['question'], label=row['answer'],
                             video_path=videos[row['video_id']], video_id=row['video_id'],
                             category=row['dim'], metric='tempcompass-' + suffix))
        save_video('tempcompass-' + suffix, rows, record,
                   'Official rule evaluation with --disable_llm, applied to last boxed answer; captioning stored but not scored.')


def restricted_video(path, destination, start, end, frames=False, fps=None):
    """Losslessly expose only the interval permitted by MVBench annotations."""
    import cv2
    marker = destination.with_suffix('.json')
    identity = dict(source=str(path), start=start, end=end, frame_directory=frames, source_fps=fps)
    if frames:
        identity['frame_index_origin'] = 1  # Official MVBench read_frame(... first_idx=1)
    if destination.exists() and marker.exists() and json.loads(marker.read_text()).get('identity') == identity:
        return str(destination.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = None
    writer = None
    count = 0
    try:
        if frames:
            paths = sorted(path.glob('*.jpg'))
            if not paths:
                raise FileNotFoundError('No source frames: ' + str(path))
            if [p.name for p in paths] != [f'{i:05d}.jpg' for i in range(1, len(paths)+1)]:
                raise RuntimeError('TVQA frames must be consecutively numbered from 00001.jpg')
            fps = float(fps)
            first = max(1, round(float(start) * fps))
            last = min(len(paths), round(float(end) * fps))
            def read():
                for p in paths[first-1:last]:
                    image = cv2.imread(str(p))
                    if image is None:
                        raise RuntimeError('Unreadable video frame: ' + str(p))
                    yield image
        else:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError('Unreadable source video: ' + str(path))
            fps = capture.get(cv2.CAP_PROP_FPS)
            first = max(0, round(float(start) * fps))
            last = min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))-1, round(float(end) * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            def read():
                for _ in range(first, last+1):
                    success, image = capture.read()
                    if not success:
                        raise RuntimeError('Source video ended inside its permitted interval: ' + str(path))
                    yield image
        for image in read():
            if writer is None:
                height, width = image.shape[:2]
                if width % 2 or height % 2:
                    raise RuntimeError('Odd source dimensions require explicit lossless conversion validation')
                writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'FFV1'), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError('Lossless FFV1 writer unavailable')
                first_frame = image.copy()
            writer.write(image)
            last_frame = image
            count += 1
    finally:
        if writer:
            writer.release()
        if capture:
            capture.release()
    if not count:
        raise RuntimeError('MVBench annotation selects no video frames')
    from decord import VideoReader, cpu
    import numpy as np
    decoded = VideoReader(str(destination), ctx=cpu(0), num_threads=1)
    if len(decoded) != count or not np.array_equal(decoded[0].asnumpy(), first_frame[:, :, ::-1]) or not np.array_equal(decoded[-1].asnumpy(), last_frame[:, :, ::-1]):
        raise RuntimeError('Lossless interval validation failed: ' + str(destination))
    write(marker, dict(identity=identity, frames=count, fps=fps, source_first_frame=first,
                       source_last_frame=last, first_and_last_frames_match_losslessly=True))
    return str(destination.resolve())


def mvbench():
    record = source('mvbench')
    folder = ROOT / 'data/benchmarks/mvbench'
    video_root = folder / 'video'
    for archive in sorted(video_root.glob('*.zip')):
        unpack(archive, video_root)
    # Match the source mapping in the pinned official mvbench.ipynb.
    folders = {
        'action_sequence': 'star/Charades_v1_480', 'action_prediction': 'star/Charades_v1_480',
        'action_antonym': 'ssv2_video', 'fine_grained_action': 'Moments_in_Time_Raw/videos',
        'unexpected_action': 'FunQA_test/test', 'object_existence': 'clevrer/video_validation',
        'object_interaction': 'star/Charades_v1_480', 'object_shuffle': 'perception/videos',
        'moving_direction': 'clevrer/video_validation', 'action_localization': 'sta/sta_video',
        'scene_transition': 'scene_qa/video', 'action_count': 'perception/videos',
        'moving_count': 'clevrer/video_validation', 'moving_attribute': 'clevrer/video_validation',
        'state_change': 'perception/videos', 'character_order': 'perception/videos',
        'egocentric_navigation': 'vlnqa', 'episodic_reasoning': 'tvqa/frames_fps3_hq',
        'counterfactual_inference': 'clevrer/video_validation'}
    bounded = {'action_sequence', 'action_prediction', 'object_interaction', 'action_localization', 'episodic_reasoning'}
    rows = []
    for task, subfolder in folders.items():
        annotation = json.loads((folder / 'json' / (task + '.json')).read_text())
        for i, row in enumerate(annotation):
            raw = video_root / subfolder / row['video']
            if not raw.exists():
                # Some public archives have one extra wrapper directory.
                matches = list(video_root.glob('*/' + subfolder + '/' + row['video']))
                if len(matches) != 1:
                    raise FileNotFoundError(str(raw))
                raw = matches[0]
            if task in bounded:
                key = hashlib.sha256(json.dumps([str(raw), row['start'], row['end']]).encode()).hexdigest()[:24]
                video = restricted_video(raw, folder / 'permitted_intervals' / (key + '.avi'),
                                         row['start'], row['end'], frames=task == 'episodic_reasoning',
                                         fps=row.get('fps'))
            else:
                video = str(raw.resolve())
            options = row['candidates']
            answer = chr(65 + options.index(row['answer']))
            question = row['question'] + '\n' + '\n'.join(chr(65+j) + '. ' + str(t) for j, t in enumerate(options))
            question += '\nAnswer with the option letter.'
            rows.append(dict(question_id=f'{task}-{i}', text=question, label=answer, video_path=video,
                             category=task, metric='mvbench', choices=options, source_video=str(raw),
                             source_start=row.get('start'), source_end=row.get('end')))
        print('Prepared MVBench task', task, len(annotation), flush=True)
    if len(rows) != 3800:
        raise RuntimeError('Expected exactly 19 public tasks x 200 questions')
    note = ('19/20 tasks only: fine_grained_pose requires separately obtained NTU RGB+D media (official release gap). '
            'Never report this as full MVBench. Preserve annotated time intervals using verified lossless clips; '
            'frame-based TVQA encoded losslessly at source fps. Agentic video inference differs from fixed-frame notebook.')
    save_video('mvbench-public-19', rows, record, note)
    write(ROOT / 'manifests/mvbench-ntu-gap.json', dict(status='external_media_missing', excluded_task='fine_grained_pose',
          excluded_questions=200, listed_required_videos=320,
          official_notice='https://huggingface.co/datasets/OpenGVLab/MVBench#dataset-description',
          source_access='https://rose1.ntu.edu.sg/dataset/actionRecognition/', full_mvbench_evaluated=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('benchmark', choices=['mathvista', 'mmmu', 'charxiv', 'tempcompass', 'mvbench'])
    args = p.parse_args()
    globals()[args.benchmark]()
