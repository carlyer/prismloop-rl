"""Persist research evidence without copying model weights or mutating baselines."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import re

from download_preparation_queue import write
from run_training import ROOT

DEST = Path('/path/to/workspace/PyVision-RL/reproduction/research/20261021-improved-rl')


def mirror():
    errors, copied = [], 0
    sources = list((ROOT/'manifests').glob('*research*.json'))
    sources += list((ROOT/'manifests').glob('*final-answer*.json'))
    sources += list((ROOT/'logs').glob('*research*.log'))
    sources += [p for p in [ROOT/'research-results.csv', ROOT/'research-results.txt', ROOT/'TODO.txt'] if p.exists()]
    for run in (ROOT/'outputs').glob('*research*'):
        if not run.is_dir():
            continue
        for sub in ['invocation.json', 'exit-status.json', 'runtime-log-archive.json', 'summary.json', 'process-exit-validation.json']:
            if (run/sub).is_file():
                sources.append(run/sub)
        sources.extend(p for p in run.glob('*-repeat*.json') if re.fullmatch(r'.+-repeat\d+(?:\.previous)?\.json',p.name))
        for sub in ['rollouts', 'source-snapshots', 'runtime-logs']:
            sources.extend(p for p in (run/sub).rglob('*') if p.is_file())
    for source in sources:
        relative = source.relative_to(ROOT)
        target = DEST/'evidence'/relative
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == source.stat().st_size and target.stat().st_mtime_ns == source.stat().st_mtime_ns:
                continue
            temporary = target.with_name(target.name+'.mirror-tmp')
            shutil.copy2(source, temporary)
            temporary.replace(target)
            copied += 1
        except OSError as exc:
            errors.append(dict(file=str(source), error=str(exc)))
    report = dict(status='passed' if not errors else 'needs_attention', files_copied=copied,
                  destination=str(DEST/'evidence'), errors=errors,
                  observed_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Logs, trajectories, source snapshots and small reports; weights stay in recorded checkpoint storage')
    write(ROOT/'manifests/research-persistent-mirror-status.json', report)


if __name__ == '__main__':
    while True:
        mirror()
        time.sleep(60)
