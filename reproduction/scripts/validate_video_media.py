"""Decode boundary frames for every distinct evaluation video, without GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from download_preparation_queue import ROOT, digest, write


def inspect(path):
    try:
        from decord import VideoReader, cpu
        reader = VideoReader(path, ctx=cpu(0), num_threads=1)
        count = len(reader)
        if count <= 0 or reader.get_avg_fps() <= 0:
            raise ValueError('Non-positive video frame count or frame rate')
        frames = reader.get_batch([0, count-1]).asnumpy()
        return dict(path=path, status='passed', frames=count, fps=reader.get_avg_fps(),
                    dimensions=list(frames.shape[1:3]), size=Path(path).stat().st_size)
    except Exception as exc:
        return dict(path=path, status='failed', error_class=type(exc).__name__, error=str(exc))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('benchmarks', nargs='+')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--wait-for-ready', action='store_true',
                   help='Wait for prepared benchmark inputs before decoding their media')
    args = p.parse_args()
    sources = {name: ROOT / 'data/evaluation' / (name + '.json') for name in args.benchmarks}
    if args.wait_for_ready:
        while True:
            ready = []
            for name, path in sources.items():
                marker = ROOT / 'manifests' / ('benchmark-' + name + '.json')
                ready.append(path.exists() and marker.exists() and
                             json.loads(marker.read_text()).get('status') == 'ready')
            if all(ready):
                break
            time.sleep(20)
    hashes = {name: digest(path) for name, path in sources.items()}
    paths = set()
    for path in sources.values():
        paths.update(row['video_path'] for row in json.loads(path.read_text()))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(inspect, sorted(paths)))
    if any(digest(path) != hashes[name] for name, path in sources.items()):
        raise RuntimeError('Evaluation inputs changed during media validation')
    failed = [row for row in results if row['status'] != 'passed']
    report = dict(status='passed' if not failed else 'failed', unique_videos=len(paths), failures=len(failed),
                  checks='First and last frame decoded; positive frame count/fps and dimensions recorded',
                  source_sha256=hashes, results=results)
    for name in args.benchmarks:
        destination = ROOT / 'manifests' / ('media-validation-' + name + '.json')
        write(destination, report)
        marker = ROOT / 'manifests' / ('benchmark-' + name + '.json')
        if marker.exists():
            record = json.loads(marker.read_text())
            if record.get('sha256') == hashes[name]:
                record.update(media_decode_check=report['status'], media_validation_report=str(destination))
                write(marker, record)
    print(json.dumps({k:v for k,v in report.items() if k!='results'}, indent=2))
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
