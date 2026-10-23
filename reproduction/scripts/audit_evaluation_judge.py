"""Audit the published evaluation grader on controlled answers without changing it."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from download_preparation_queue import ROOT, write


def main():
    from openai import OpenAI
    sys.path.insert(0, str(ROOT.parent.parent / 'PyVision-RL-Eval'))
    from cal_score import answer_verify as grader
    config = json.loads((ROOT / 'judge-config.json').read_text())
    cases = [row for row in json.loads((ROOT / 'manifests/judge-calibration-cases.json').read_text())
             if row['id'].startswith('manual-')]
    if len(cases) != 32:
        raise RuntimeError('Expected the frozen 32 manual calibration cases')
    original_calibration = (ROOT / 'manifests/judge-calibration.json').read_bytes()
    client = OpenAI(api_key=config['api_key'], base_url=config['base_url'], timeout=180, max_retries=1)
    local = threading.local()
    def factory(**kwargs):
        def create(**request):
            local.request = {k: v for k, v in request.items() if k != 'messages'}
            response = client.chat.completions.create(**request)
            local.raw = response.choices[0].message.content or ''
            return response
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    def run(case):
        local.raw = None
        local.request = {}
        started = time.monotonic()
        result = dict(id=case['id'], category=case['category'], source=case['source'],
                      expected=case['expected'], question=case['question'], ground_truth=case['gold'],
                      controlled_answer=case['answer'])
        try:
            score = grader.compute_score(case['answer'], case['gold'], case['question'], config, case['source'])
            raw = (local.raw or '').strip()
            parsed = raw.split('Judgement:')[-1].strip() if 'Judgement:' in raw else raw
            accepted = any(x in parsed for x in ['0', '1']) if 'Judgement:' in raw else parsed in ['0', '1']
            result.update(score=score['score'], matches_expected=score['score'] == case['expected'],
                          raw_judge_output=local.raw, source_format_accepted=accepted,
                          strict_binary_format=bool(re.fullmatch(r'(?:Judgement:\s*)?[01]', raw)),
                          request_settings=local.request)
        except Exception as exc:
            result.update(error_class=type(exc).__name__, error=str(exc), matches_expected=False)
        result['elapsed_seconds'] = time.monotonic() - started
        return result
    with patch.object(grader, 'OpenAI', factory), ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, cases))
    if (ROOT / 'manifests/judge-calibration.json').read_bytes() != original_calibration:
        raise RuntimeError('Frozen training judge calibration changed during the separate evaluation audit')
    report = dict(status='completed_audit', observed_utc=datetime.now(timezone.utc).isoformat(),
                  cases=len(results), correct=sum(x['matches_expected'] for x in results),
                  errors=sum('error' in x for x in results),
                  source_format_accepted=sum(x.get('source_format_accepted', False) for x in results),
                  strict_binary_outputs=sum(x.get('strict_binary_format', False) for x in results),
                  false_positives=sum(x['expected'] == 0 and x.get('score') == 1 for x in results),
                  source_sha256=hashlib.sha256(open(grader.__file__, 'rb').read()).hexdigest(),
                  training_calibration_unchanged=True,
                  scope='Controlled-answer audit of original evaluation prompt, temperature and parser; no held-out score or 72B equivalence claim',
                  results=results)
    write(ROOT / 'manifests/evaluation-judge-format-audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
