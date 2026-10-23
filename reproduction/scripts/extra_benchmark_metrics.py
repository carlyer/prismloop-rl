"""Pinned benchmark scoring helpers, with explicit local judge substitutions."""
import ast
from functools import lru_cache
import hashlib
import json
import logging
import random
import re
import sys
import threading

from download_preparation_queue import ROOT, digest

REFERENCES = ROOT / 'references/extra-benchmark-scorers'
LOCK = threading.Lock()


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


@lru_cache(maxsize=None)
def helpers(relative):
    """Load only pure functions, avoiding upstream CLI/network import side effects."""
    import numpy as np
    path = REFERENCES / relative
    tree = ast.parse(path.read_text())
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    namespace = dict(re=re, random=random.Random(42), np=np, Dict=dict,
                     logging=logging, distance=edit_distance)
    exec(compile(functions, str(path), 'exec'), namespace)
    return namespace


def protocol(benchmark):
    if benchmark == 'mathvista':
        paths = ['mathvista/evaluation/calculate_score.py']
        note = 'Official normalization/equality on boxed extraction; edit distance implementation equivalent; no GPT extraction.'
    elif benchmark == 'mmmu':
        paths = ['mmmu/mmmu/utils/eval_utils.py']
        note = 'Official parser/equality, deterministic per-question fallback seed, boxed extraction.'
    elif benchmark.startswith('charxiv-'):
        paths = ['charxiv/src/constants.py', 'charxiv/src/descriptive_utils.py', 'charxiv/src/reasoning_utils.py']
        note = 'Official grading prompts with local Qwen2.5-VL-32B instead of GPT-4o; one descriptive item per grading call.'
    elif benchmark.startswith('tempcompass-'):
        paths = ['tempcompass/eval_multi-choice.py', 'tempcompass/eval_yes_no.py', 'tempcompass/eval_caption_matching.py']
        note = 'Official rule matching, --disable_llm variant; last boxed answer with one matching outer quote pair removed.'
    elif benchmark.startswith('mvbench'):
        paths = ['mvbench/video_chat2/mvbench.ipynb']
        note = 'Official notebook check_ans rule; public 19/20 task subset excludes unavailable NTU pose media; not full MVBench.'
    else:
        return None
    return dict(note=note, source_sha256={str(REFERENCES / p): digest(REFERENCES / p) for p in paths},
                adapter_sha256=digest(ROOT / 'scripts/extra_benchmark_metrics.py'))


def score_extra(benchmark, prediction, row, judge):
    if benchmark.startswith('mvbench'):
        function = mvbench_matcher()
        if not prediction.strip():
            return dict(score=0.0)
        ground_truth = '(' + row['label'] + ') ' + row['choices'][ord(row['label'])-65]
        return dict(score=float(function(prediction, ground_truth)), metric_scope='public 19/20 tasks')
    if benchmark == 'mathvista':
        f = helpers('mathvista/evaluation/calculate_score.py')
        normalized = f['normalize_extracted_answer'](prediction, row['choices'], row['question_type'],
                                                     row['answer_type'], row['precision'], True)
        return dict(score=float(f['safe_equal'](normalized, row['label'])), normalized_answer=normalized)
    if benchmark == 'mmmu':
        f = helpers('mmmu/mmmu/utils/eval_utils.py')
        if row['question_type'] == 'multiple-choice':
            options = {chr(65+i): text for i, text in enumerate(row['choices'])}
            with LOCK:
                seed = int(hashlib.sha256(str(row['question_id']).encode()).hexdigest()[:16], 16)
                f['random'].seed(seed)
                normalized = f['parse_multi_choice_response'](prediction, list(options), options)
            correct = f['eval_multi_choice'](row['label'], normalized)
        else:
            normalized = f['parse_open_response'](prediction)
            correct = f['eval_open'](row['label'], normalized)
        return dict(score=float(correct), normalized_answer=normalized)
    if benchmark.startswith('tempcompass-'):
        raw_prediction = prediction
        prediction = prediction.replace('</s>', '').strip()
        if len(prediction) >= 2 and prediction[0] == prediction[-1] and prediction[0] in ['\"', "'"]:
            prediction = prediction[1:-1].strip()
        answer = row['label']
        if benchmark == 'tempcompass-yn':
            normalized = helpers('tempcompass/eval_yes_no.py')['extract_pred'](prediction)
            correct = normalized == answer if normalized else False
        elif benchmark == 'tempcompass-caption-matching':
            normalized = helpers('tempcompass/eval_caption_matching.py')['eval_rule'](prediction, row['text'], answer)
            correct = normalized == 1
        else:
            normalized = prediction
            if prediction == answer:
                correct = True
            elif prediction in ['A', 'B', 'C', 'D']:
                correct = prediction == answer[0]
            elif any(prediction.startswith(p) for p in ['A.', 'B.', 'C.', 'D.']):
                correct = prediction.split('.')[0] == answer[0]
            elif any(prediction.startswith(p) for p in ['A)', 'B)', 'C)', 'D)']):
                correct = prediction.split(')')[0] == answer[0]
            else:
                correct = False
        return dict(score=float(correct), normalized_answer=normalized, llm_fallback_disabled=True,
                    raw_boxed_answer=raw_prediction, unquoted_boxed_answer=prediction)
    if benchmark.startswith('charxiv-'):
        from openai import OpenAI
        sys.path.insert(0, str(REFERENCES / 'charxiv/src'))
        if benchmark == 'charxiv-reasoning':
            from reasoning_utils import build_reasoning_grading_queries
            identifier = row['figure_id']
            prompt = build_reasoning_grading_queries(
                {identifier: dict(figure_id=identifier, inst_category=row['inst_category'], answer=row['label'])},
                {identifier: dict(raw_question=row['raw_question'], response=prediction)})[identifier]['grading_query']
            score_key, answer_key = 'score', 'extracted_answer'
        else:
            from descriptive_utils import build_descriptive_grading_queries
            prompt = build_descriptive_grading_queries({row['charxiv_qid']: [(row['question_id'], prediction, row['label'])]},
                                                       nq_per_query=1)[0]['grading_query']
            score_key, answer_key = 'score_T1', 'extract_answer_T1'
        client = OpenAI(api_key=judge['api_key'], base_url=judge['base_url'], timeout=180, max_retries=2)
        response = client.chat.completions.create(model=judge['model_name'],
                    messages=[dict(role='user', content=prompt)], response_format={'type': 'json_object'},
                    max_tokens=1024, temperature=0, top_p=1, seed=42).choices[0].message.content
        grading = json.loads(response)
        if grading.get(score_key) not in [0, 1]:
            raise ValueError('CharXiv judge returned an invalid grade: ' + response)
        return dict(score=float(grading[score_key]), normalized_answer=grading.get(answer_key),
                    judge_response=response, judge_substitution='local 32B replaces official GPT-4o')
    return None


@lru_cache(maxsize=1)
def mvbench_matcher():
    path = REFERENCES / 'mvbench/video_chat2/mvbench.ipynb'
    notebook = json.loads(path.read_text())
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if 'def check_ans(' in source:
            tree = ast.parse(source)
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_ans')
            namespace = {}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
            return namespace['check_ans']
    raise RuntimeError('Official MVBench scorer is missing')
