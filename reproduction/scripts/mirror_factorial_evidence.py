"""Persist small ablation artifacts; do not copy or alter model checkpoints."""
from datetime import datetime,timezone
from pathlib import Path
import re
import shutil
import time

from download_preparation_queue import write
from run_training import ROOT

DEST=Path('/path/to/workspace/PyVision-RL/reproduction/research/20261023-controlled-ablations')


def mirror():
    files=[]
    for pattern in ['algorithm-ablation*.json','ablation-*.json','eval-ablation*.json','all-model-results-with-ablations.json']:
        files.extend((ROOT/'manifests').glob(pattern))
    files.extend((ROOT/'logs').glob('*.log'))
    files.extend(ROOT.glob('*.csv'));files.extend(ROOT.glob('*.txt'))
    for run in (ROOT/'outputs').iterdir():
        if not run.is_dir():continue
        for name in ['invocation.json','exit-status.json','summary.json','process-exit-validation.json','runtime-log-archive.json']:
            if (run/name).is_file():files.append(run/name)
        for directory in ['rollouts','source-snapshots','runtime-logs']:
            files.extend(p for p in (run/directory).rglob('*') if p.is_file())
        files.extend(p for p in run.glob('*-repeat*.json') if re.fullmatch(r'.+-repeat\d+(?:\.previous)?\.json',p.name))
    errors=[]
    for source in files:
        target=DEST/'evidence'/source.relative_to(ROOT)
        try:
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.stat().st_size==source.stat().st_size and target.stat().st_mtime_ns==source.stat().st_mtime_ns:continue
            temporary=target.with_name(target.name+'.mirror-tmp');shutil.copy2(source,temporary);temporary.replace(target)
        except OSError as exc:errors.append(dict(file=str(source),error=str(exc)))
    write(ROOT/'manifests/algorithm-ablation-mirror-status.json',dict(status='passed' if not errors else 'needs_attention',errors=errors,observed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    while True:mirror();time.sleep(60)
