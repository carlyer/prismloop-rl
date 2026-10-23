"""Prepare remaining paper assets after the active Image RL downloads finish."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def fetch(task):
    manifest = ROOT / task['source_manifest']
    record = json.loads(manifest.read_text())
    target = ROOT / task['target']
    target.mkdir(parents=True, exist_ok=True)
    selected = [f for f in record['files'] if f['path'] in task['files']]
    assert len(selected) == len(task['files'])
    lines = []
    reused = []
    for item in selected:
        path = target / item['path']
        if task.get('local_source') and item.get('sha256'):
            shared = Path(task['local_source']) / item['path']
            if shared.is_file() and shared.stat().st_size == item['size'] and digest(shared) == item['sha256']:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.symlink_to(shared)
                reused.append({'path': item['path'], 'source': str(shared), 'sha256': item['sha256']})
        if (path.is_file() and path.stat().st_size == item['size'] and
                (not item.get('sha256') or digest(path) == item['sha256'])):
            continue
        if path.is_symlink():
            raise RuntimeError('Shared linked asset changed; refusing to overwrite shared data: ' + str(path))
        prefix = 'datasets/' if record['repo_type'] == 'dataset' else ''
        url = f"https://huggingface.co/{prefix}{record['repo_id']}/resolve/{record['revision']}/{quote(item['path'])}"
        lines += [url, f'  dir={path.parent}', f'  out={path.name}']
        if item.get('sha256'):
            lines.append('  checksum=sha-256=' + item['sha256'])
    if lines:
        uris = ROOT / 'manifests' / (task['name'] + '-uris.txt')
        uris.write_text('\n'.join(lines) + '\n')
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/opt/pyvision-tools/usr/lib/x86_64-linux-gnu' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        for name in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY']:
            if name in env:
                env[name.lower()] = env[name]
        command = ['/opt/pyvision-tools/usr/bin/aria2c', '--input-file=' + str(uris),
                   '--continue=true', '--max-concurrent-downloads=2', '--split=4',
                   '--max-connection-per-server=4', '--min-split-size=8M',
                   '--file-allocation=none', '--auto-file-renaming=false',
                   '--allow-overwrite=true', '--check-integrity=true', '--max-tries=8',
                   '--retry-wait=5', '--timeout=60', '--connect-timeout=15',
                   '--summary-interval=60', '--enable-color=false', '--show-console-readout=false']
        # aria2 may exhaust a proxy connection before its normal transfer retry
        # path. Resume the same pinned URIs on transient failures.
        for attempt in range(4):
            result = subprocess.run(command, env=env)
            if result.returncode == 0:
                break
            if attempt == 3:
                raise subprocess.CalledProcessError(result.returncode, command)
            print(f"Retry {task['name']} aria2 transfer after exit {result.returncode}; attempt {attempt+2}", flush=True)
            time.sleep(10)
    for item in selected:
        path = target / item['path']
        if not path.is_file() or path.stat().st_size != item['size']:
            raise RuntimeError(f'Missing/incomplete file: {path}')
        if item.get('sha256') and digest(path) != item['sha256']:
            raise RuntimeError(f'Checksum mismatch: {path}')
    write(ROOT / 'manifests' / (task['name'] + '-prepared-assets.json'), {
        'status': 'complete', 'repo_id': record['repo_id'], 'revision': record['revision'],
        'target': str(target), 'selected_files': selected, 'reused_files': reused,
        'lfs_checksums_verified': True})


def main():
    state = ROOT / 'manifests/preparation-queue-status.json'
    plan = json.loads((ROOT / 'preparation-plan.json').read_text())
    previous = json.loads(state.read_text()) if state.exists() else {}
    done = previous.get('completed', [])
    try:
        write(state, {'status': 'running', 'stage': 'waiting_for_image_downloads', 'completed': done})
        # No competing network-heavy transfers while the first training run is waiting.
        while True:
            ready = all(json.loads((ROOT / f'manifests/{n}.json').read_text()).get('status') == 'complete'
                        for n in ['image-rl-data', 'image-sft-model'])
            if ready:
                break
            pid = json.loads((ROOT / 'manifests/download-process.json').read_text())['pid']
            proc = Path(f'/proc/{pid}/stat')
            if not proc.exists() or proc.read_text().split()[2] == 'Z':
                raise RuntimeError('Primary Image asset downloader stopped before completion')
            time.sleep(30)
        for task in plan['download_queue']:
            if task['name'] in done:
                receipt = ROOT / 'manifests' / (task['name'] + '-prepared-assets.json')
                source = json.loads((ROOT / task['source_manifest']).read_text())
                if (receipt.exists() and json.loads(receipt.read_text()).get('status') == 'complete'
                        and json.loads(receipt.read_text()).get('revision') == source['revision']):
                    continue
                done.remove(task['name'])
            write(state, {'status': 'running', 'stage': task['name'], 'completed': done})
            fetch(task)
            if task.get('prepare_benchmark'):
                subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_benchmarks.py'), task['name']], check=True)
            done.append(task['name'])
            subprocess.run([sys.executable, str(ROOT / 'scripts/audit_readiness.py')], check=True)
        write(state, {'status': 'complete', 'stage': 'queued_assets_prepared', 'completed': done,
                      'paper_results_reproduced': False})
    except BaseException as exc:
        write(state, {'status': 'failed', 'error': str(exc), 'completed': done})
        raise


if __name__ == '__main__':
    with (ROOT / 'preparation-queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
