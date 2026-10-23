"""Execute the unchanged original evaluator with local result storage.
Only suppress an identical rewrite of its frozen reference template.
Original cached records and source paths retain their generation fingerprints.
"""
from pathlib import Path
import json
import runpy
import sys

ORIGINAL = Path('/path/to/workspace/PyVision-RL/reproduction')
sys.path.insert(0, str(ORIGINAL / 'scripts'))
import download_preparation_queue as storage
original_write = storage.write

def write(path, value):
    path = Path(path)
    if path.is_relative_to(ORIGINAL):
        if path.is_file() and json.loads(path.read_text()) == value:
            return
        raise RuntimeError('Refusing to mutate the quota-limited original workspace: ' + str(path))
    original_write(path, value)

storage.write = write
if __name__ == '__main__':
    try:
        runpy.run_path(str(ORIGINAL / 'scripts/evaluate_official.py'), run_name='__main__')
    except SystemExit as exc:
        # PythonExecutor queues can leave a blocked feeder thread after all
        # inference and scoring have finished. Exit only after auditing the full
        # real-inference result; generation and score code above remain unchanged.
        if exc.code in (None, 0) and '--output' in sys.argv and '--config-only' not in sys.argv:
            import hashlib
            import os
            from evaluation_evidence import audit_records
            output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
            summary_path, invocation_path = output / 'summary.json', output / 'invocation.json'
            if summary_path.exists() and invocation_path.exists():
                report = json.loads(summary_path.read_text())
                config = json.loads(invocation_path.read_text())
                if report.get('full_dataset_evaluated') and not report.get('self_test_only'):
                    evidence = audit_records(output, config, report,
                                             ORIGINAL / 'data/evaluation' / (config['benchmark'] + '.json'))
                    original_write(output / 'process-exit-validation.json', dict(
                        status='passed', evidence=evidence,
                        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        scope='Post-evaluation exit only; generation, scoring and cached results unchanged'))
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(0)
        raise
