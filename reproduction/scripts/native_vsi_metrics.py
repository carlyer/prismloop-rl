"""Re-score boxed VSI answers with the pinned dataset authors' metric code."""
import ast
from collections import OrderedDict
from functools import lru_cache, partial
import json
from pathlib import Path
import re
from types import SimpleNamespace

from download_preparation_queue import ROOT, digest, write

FOLDER = ROOT / 'references/extra-benchmark-scorers/vsibench'


@lru_cache(maxsize=1)
def load_scorer():
    import numpy as np
    import pandas as pd
    path = FOLDER / 'utils.py'
    source = json.loads((FOLDER / 'source.json').read_text())
    if digest(path) != source['sha256']:
        raise RuntimeError('Pinned native VSI scoring source changed')
    tree = ast.parse(path.read_text())
    constants = {'MCA_QUESTION_TYPES', 'NA_QUESTION_TYPES', 'METRICS_FOR_MCA',
                 'METRICS_FOR_NA', 'WORST_CASE_FOR_METRICS'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            nodes.append(node)
    namespace = dict(np=np, pd=pd, partial=partial, OrderedDict=OrderedDict,
                     datasets=SimpleNamespace(Dataset=object),
                     eval_logger=SimpleNamespace(info=lambda *args, **kwargs: None))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, source


def rescore(folder, report):
    scorer, source = load_scorer()
    documents, records = [], []
    for path in sorted(folder.glob('*-repeat*.json')):
        if not re.fullmatch(r'.*-repeat\d+\.json', path.name):
            continue
        result = json.loads(path.read_text())
        prediction = result.get('predicted_answer') or ['']
        doc = dict(question_type=result['category'], ground_truth=result['ground_truth'])
        scored = scorer['vsibench_process_results'](doc, [prediction[-1]])['vsibench_score']
        documents.append(scored)
        metric = 'accuracy' if doc['question_type'] in scorer['MCA_QUESTION_TYPES'] else 'MRA:.5:.95:.05'
        records.append(dict(question_id=result['question_id'], repeat=result['repeat'],
                            question_type=doc['question_type'], native_score=float(scored[metric]),
                            released_pyvision_score=result['score']))
    if len(documents) != report['samples']:
        raise RuntimeError('Native VSI evaluation is missing per-question results')
    result = dict(status='complete', samples=len(documents),
                  metrics=scorer['vsibench_aggregate_results'](documents), source=source,
                  prediction_extraction='Last boxed answer from the PyVision interaction; empty/missing output scores zero',
                  judge_used=False,
                  note='Native rule scoring is separate from released PyVision judge/strict-threshold scoring')
    write(folder / 'native-vsi-summary.json', result)
    (folder / 'native-vsi-results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    return result
