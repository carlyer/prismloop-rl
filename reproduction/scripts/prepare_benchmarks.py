"""Adapt public benchmark releases to the official evaluator's canonical rows."""
import argparse
import base64
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import re
import zipfile

from download_preparation_queue import ROOT, digest, write


def unpack(archive, target):
    marker = archive.with_suffix(archive.suffix + '.extracted.json')
    checksum = digest(archive)
    if marker.exists() and json.loads(marker.read_text()).get('sha256') == checksum:
        return
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            path = (target / entry.filename).resolve()
            if target.resolve() not in path.parents and path != target.resolve():
                raise RuntimeError('Unsafe archive path: ' + entry.filename)
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError('Archive contains a symbolic link')
        z.extractall(target)
    write(marker, {'sha256': checksum})


def media(root, relative):
    path = root / relative
    if path.is_file():
        return str(path.resolve())
    # Public ZIPs sometimes add one enclosing directory.
    matches = list(root.glob('*/' + relative))
    if len(matches) != 1:
        raise FileNotFoundError(str(path))
    return str(matches[0].resolve())


def embedded_image(value, destination):
    from PIL import Image
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    elif isinstance(value, dict):
        if value.get('bytes'):
            data = value['bytes']
        else:
            return str(Path(value['path']).resolve())
    elif isinstance(value, str):
        data = base64.b64decode(value.split(',', 1)[-1])
    else:
        raise TypeError('Unrecognized embedded image format')
    with Image.open(io.BytesIO(data)) as image:
        suffix = '.jpg' if image.format == 'JPEG' else '.png'
        image.verify()
    destination = destination.with_suffix(suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(data)
    return str(destination.resolve())


def parquet_rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
        yield from batch.to_pylist()


def canonical(identifier, question, answer, images, category, **metadata):
    return {'question_id': str(identifier), 'text': question, 'label': str(answer),
            'image_path_list': images, 'category': category, **metadata}


def save(name, rows, source, protocol_note=None):
    identifiers = [r['question_id'] for r in rows]
    if len(set(identifiers)) != len(rows):
        raise RuntimeError(name + ': duplicate question identifiers')
    missing = [p for r in rows for p in r['image_path_list'] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(str(missing[:5]))
    output = ROOT / 'data/evaluation' / (name + '.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False) + '\n')
    report = {'status': 'ready', 'name': name, 'samples': len(rows), 'unique_media': len({p for r in rows for p in r['image_path_list']}),
              'data_file': str(output), 'sha256': digest(output), 'source': source,
              'upstream_reader_bench_type': 'vstar', 'format': 'canonical official evaluator input',
              'protocol_note': protocol_note, 'evaluated': False}
    write(ROOT / 'manifests' / ('benchmark-' + name + '.json'), report)
    print(name, len(rows), 'questions; media paths verified', flush=True)


def mathvision():
    shared = Path('/dataset_rc_mm/share/datasets/huggingface.co/MathLLMs/MathVision')
    source = json.loads((ROOT / 'references/source-inventory/mathvision.json').read_text())
    for split in ['test', 'testmini']:
        rows = []
        file = next(f for f in source['files'] if f['path'].startswith('data/' + split + '-'))
        candidates = [ROOT / 'data/benchmarks/mathvision' / file['path'], shared / file['path']]
        path = next((p for p in candidates if p.is_file() and p.stat().st_size == file['size'] and digest(p) == file['sha256']), None)
        if path is None:
            write(ROOT / 'manifests' / ('mathvision-' + split + '-missing.json'), {
                'status': 'queued_for_download', 'reason': 'Shared file differs from pinned public release',
                'file': file, 'local_path': str(shared / file['path'])})
            print('MathVision', split, 'needs pinned replacement; left shared data unchanged', flush=True)
            continue
        for item in parquet_rows(path):
            image = embedded_image(item['decoded_image'], ROOT / 'data/benchmarks/mathvision/images' / item['id'])
            opts = item['options']
            if isinstance(opts, str):
                opts = json.loads(opts)
            question = item['question'].replace('\n<image1>', '')
            if opts:
                question += '\n' + '\n'.join(f'{chr(65+i)}. {o}' for i, o in enumerate(opts))
            rows.append(canonical(item['id'], question, item['answer'], [image], item['subject'], split=split))
        save('mathvision' if split == 'test' else 'mathvision-mini', rows,
             {'repo_id': source['repo_id'], 'revision': source['revision'], 'local_source': str(shared)},
             'Both test (3040) and testmini (304) retained; exact paper split must be confirmed before numerical comparison.')


def prepare(name):
    root = ROOT / 'data/benchmarks' / name
    source = json.loads((ROOT / f'references/source-inventory/{name}.json').read_text())
    provenance = {'repo_id': source['repo_id'], 'revision': source['revision']}
    if name == 'mathvision':
        return mathvision()
    if name == 'mathverse':
        unpack(root / 'images.zip', root)
        items = json.loads((root / 'testmini.json').read_text())
        groups = defaultdict(list)
        rows = []
        for x in items:
            row = canonical(x['sample_index'], x['question'], x['answer'], [media(root, x['image'])], 'mathverse',
                            problem_index=x['problem_index'], problem_version=x['problem_version'])
            rows.append(row)
            groups[x['problem_version']].append(row)
        note = 'All public image variants retained separately; the exact paper subset is not specified in the released evaluation inputs.'
        save(name, rows, provenance, note)
        for version, part in groups.items():
            save(name + '-' + re.sub(r'[^a-z0-9]+', '-', version.lower()).strip('-'), part, provenance, note)
    elif name == 'wemath':
        unpack(root / 'testmini.zip', root)
        rows = []
        for x in json.loads((root / 'testmini.json').read_text()):
            rows.append(canonical(str(x['ID']) + '-' + str(x.get('question number', '')), x['question'] + '\n' + x['option'],
                                  x['answer'], [media(root, x['image_path'])], 'wemath', split=x['split']))
        save(name, rows, provenance, 'Report answer accuracy for paper comparison; keep official We-Math diagnostic metrics separate.')
    elif name == 'tirbench':
        unpack(root / 'data.zip', root)
        rows = []
        for x in json.loads((root / 'TIR-Bench.json').read_text()):
            images = [media(root, x[k]) for k in ['image_1', 'image_2'] if x.get(k)]
            rows.append(canonical(x['id'], x['prompt'], x['answer'], images, x['task']))
        save(name, rows, provenance)
    elif name == 'dynamath':
        rows = []
        for file in sorted((root / 'data').glob('*.parquet')):
            variant = re.search(r'sample(\d+)-', file.name).group(1)
            part = []
            for x in parquet_rows(file):
                image = embedded_image(x['decoded_image'], root / 'images' / f"variant{variant}-{x['id']}")
                part.append(canonical(variant + '-' + str(x['id']), x['question'], x['ground_truth'], [image], 'dynamath',
                                      variant=variant, seed_question_id=x['id'], answer_type=x['answer_type']))
            rows += part
            save(name + '-variant' + variant, part, provenance, 'Keep each sampled variant; exact paper variant selection is not released.')
        if len(rows) != 5010:
            raise RuntimeError('Expected all 10 public DynaMath variants')
        save(name, rows, provenance, '501 seed questions x 10 variants. Report per-variant and aggregate scores; do not assume equivalence to unreleased author-generated variants.')
    elif name == 'hrbench':
        for resolution in ['4k', '8k']:
            rows = []
            for x in parquet_rows(root / f'hr_bench_{resolution}.parquet'):
                image = embedded_image(x['image'], root / 'images' / (resolution + '-' + str(x['index'])))
                question = x['question'] + '\n' + '\n'.join(k + '. ' + x[k] for k in 'ABCD' if x.get(k))
                rows.append(canonical(x['index'], question, x['answer'], [image], 'hrbench-' + resolution,
                                      original_category=x['category'], cycle_category=x.get('cycle_category')))
            save(name + '-' + resolution, rows, provenance, 'Preserve the public 800-row circular-option release; report scoring protocol explicitly.')
    else:
        raise ValueError(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', choices=['mathvision', 'mathverse', 'wemath', 'tirbench', 'dynamath', 'hrbench'])
    prepare(parser.parse_args().benchmark)
