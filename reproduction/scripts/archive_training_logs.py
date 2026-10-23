"""Preserve Ray worker logs without changing or contacting the trainer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def process_identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def archive(session, run, final=False):
    session, run = Path(session).resolve(strict=True), Path(run).resolve(strict=True)
    if not (run / 'invocation.json').is_file():
        raise ValueError('Run must contain its recorded training invocation')
    target = run / 'runtime-logs' / session.name
    target.mkdir(parents=True, exist_ok=True)
    records, errors = [], []
    for source in sorted((session / 'logs').iterdir()):
        if not source.is_file() or not (
            (source.name.startswith('worker-') and source.suffix in ['.out', '.err'])
            or source.name == 'raylet.err'
        ):
            continue
        try:
            destination = target / source.name
            shutil.copyfile(source, destination)
            digest = hashlib.sha256()
            with destination.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            records.append(dict(file=source.name, bytes=destination.stat().st_size,
                                sha256=digest.hexdigest()))
        except OSError as exc:
            errors.append(dict(file=source.name, error=str(exc)))
    if not records:
        errors.append(dict(error='No worker logs found in the specified Ray session'))
    report = dict(status='failed' if errors else ('archived_after_owner_exit' if final else 'live_snapshot'),
                  observed_utc=datetime.now(timezone.utc).isoformat(),
                  source_session=str(session), archive_directory=str(target),
                  owner_exit_observed=final, training_success_proven=False,
                  note='Worker stdout/stderr and raylet stderr only. A live snapshot is incomplete; '
                       'owner exit does not prove successful training. Inspect the training receipt and checkpoints.',
                  files=records, errors=errors)
    write_json(target / 'archive.json', report)
    write_json(run / 'runtime-log-archive.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--watch-owner-pid', type=int)
    args = parser.parse_args()
    session = args.session.resolve(strict=True)
    identity = process_identity(args.watch_owner_pid) if args.watch_owner_pid else None
    if args.watch_owner_pid and identity is None:
        raise RuntimeError('Specified owner is not live; cannot start a live watcher')
    report = archive(session, args.run)
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)
    if report['errors']:
        raise RuntimeError('Initial log archive failed; see the archive receipt')
    if args.watch_owner_pid:
        while process_identity(args.watch_owner_pid) == identity:
            time.sleep(20)
        report = archive(session, args.run, final=True)
        print(json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)
        if report['errors']:
            raise RuntimeError('Final log archive failed; see the archive receipt')


if __name__ == '__main__':
    main()
