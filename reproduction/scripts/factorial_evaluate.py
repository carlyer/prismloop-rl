"""Claim completed ablations, run frozen benchmarks, and publish factorial contrasts."""
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import sys
import time

from archive_training_logs import process_identity
from download_preparation_queue import write
from run_training import ROOT
from pipeline import evaluate_suite
from evaluate_research_queue import observe
from run_qwen_matched_controls import compare_protocol

RESEARCH=Path('/tmp/pyvision-rl-research-20261021/PyVision-RL/reproduction')
BASE=Path('/tmp/pyvision-rl-runtime-20261019/PyVision-RL/reproduction')
DEST=Path('/path/to/workspace/PyVision-RL/reproduction/research/20261023-controlled-ablations')


def read(path):return json.loads(Path(path).read_text())


def claim(gpu):
    with (ROOT/'factorial-evaluation-dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=read(ROOT/'manifests/algorithm-ablation-status.json')
        jobs=sorted(state['jobs'],key=lambda j:j['kind']!=('video' if gpu=='4' else 'image'))
        for job in jobs:
            if job['status']!='evaluation_queued':continue
            path=ROOT/'manifests'/f"{job['name']}-evaluation.json"
            if path.exists():
                old=read(path)
                if old['status'] in ['complete','needs_attention'] or process_identity(old.get('pid',-1))==old.get('start_time_ticks'):continue
            record=dict(status='running',pid=os.getpid(),start_time_ticks=process_identity(os.getpid()),
                        gpu=gpu,job=job['id'],model=job['model'])
            write(path,record)
            return job,record
    return None,None


def worker(gpu):
    # Supplementary transfer benchmarks use existing, validated models while
    # the first original-method controls are training.
    method='core' if gpu=='4' else 'ppo2'
    supplemental=ROOT/'manifests'/f'algorithm-ablation-supplementary-{method}.json'
    if not supplemental.exists():
        model=RESEARCH/'outputs'/f'research-eval-image-{method}-20261021-export/merged-model'
        name=f'supplementary-image-{method}-hrbench8-20261023'
        try:
            evaluate_suite(model,['hrbench-8k'],name,gpu=gpu,port=19100+int(gpu),update_state=False)
            row,config=observe(ROOT/'outputs'/(name+'-hrbench-8k'),'hrbench-8k',model)
            compare_protocol(config,read(BASE/'outputs/official-image-sft-control-hrbench-8k/invocation.json'))
            write(supplemental,dict(status='passed',method=method,result=row))
        except Exception as exc:
            write(supplemental,dict(status='needs_attention',error=str(exc)))
    while True:
        job,record=claim(gpu)
        if job:
            path=ROOT/'manifests'/f"{job['name']}-evaluation.json"
            try:
                for attempt in range(2):
                    try:
                        evaluate_suite(Path(job['model']),job['benchmarks'],job['eval_name'],gpu=gpu,
                                       port=19100+int(gpu),update_state=False)
                        break
                    except RuntimeError:
                        if attempt:raise
                        time.sleep(10)
                results=[]
                for benchmark in job['benchmarks']:
                    row,config=observe(ROOT/'outputs'/(job['eval_name']+'-'+benchmark),benchmark,job['model'])
                    reference=BASE/'outputs'/f"official-{job['kind']}-sft-control-{benchmark}/invocation.json"
                    compare_protocol(config,read(reference))
                    results.append(row)
                record.update(status='complete',results=results)
            except Exception as exc:record.update(status='needs_attention',error=str(exc))
            record['observed_utc']=datetime.now(timezone.utc).isoformat();write(path,record)
        time.sleep(20)


def report():
    state=read(ROOT/'manifests/algorithm-ablation-status.json')
    prior=read(RESEARCH/'manifests/research-results.json')['results']
    rows=[]
    for old in prior:
        if old['method']=='core' and old.get('evaluation_status')=='passed':
            rows.append(dict(kind=old['kind'],variant='full',benchmark=old['benchmark'],status='passed',
                             score_percent=old['score_percent'],additional_steps=40,output=old['output'],
                             features=['numeric','quota','tool'] if old['kind']=='video' else ['quota','tool']))
    for job in state['jobs']:
        sidecar=ROOT/'manifests'/f"{job['name']}-evaluation.json"
        scores=read(sidecar).get('results',[]) if sidecar.exists() else []
        for benchmark in job['benchmarks']:
            row=dict(kind=job['kind'],variant=job['variant'],benchmark=benchmark,status=job['status'],
                     additional_steps=40,features=job['features'])
            score=next((s for s in scores if s['benchmark']==benchmark),None)
            if score:row.update(score,status='passed')
            rows.append(row)
    for row in rows:
        reference=next((r for r in rows if r['kind']==row['kind'] and r['benchmark']==row['benchmark'] and r['variant']=='original' and r['status']=='passed'),None)
        if reference and row['status']=='passed':row['delta_vs_original40_pp']=row['score_percent']-reference['score_percent']
    effects=[]
    for kind,benchmark,features,size in [('video','vsibench',['numeric','quota','tool'],8),('image','hrbench-4k',['quota','tool'],4),('image','vstar',['quota','tool'],4)]:
        cells=[r for r in rows if r['kind']==kind and r['benchmark']==benchmark and r['status']=='passed']
        if len(cells)!=size:continue
        for feature in features:
            on=[r['score_percent'] for r in cells if feature in r['features']]
            off=[r['score_percent'] for r in cells if feature not in r['features']]
            effects.append(dict(kind=kind,benchmark=benchmark,feature=feature,mean_main_effect_pp=sum(on)/len(on)-sum(off)/len(off)))
    result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),results=rows,factorial_effects=effects,
                limitations=['One seed per cell; descriptive contrasts are not statistical significance','40 additional steps per cell','Existing full core is reused; common runtime fixes retained across all new cells'])
    write(ROOT/'manifests/algorithm-ablation-results.json',result)
    buf=io.StringIO();fields=['kind','variant','benchmark','status','additional_steps','score_percent','delta_vs_original40_pp','native_vsi_percent','output']
    writer=csv.DictWriter(buf,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (ROOT/'controlled-ablation-results.csv').write_text(buf.getvalue())
    lines=['Controlled 40-step ablations: pending is not zero', 'Kind | Variant | Benchmark | Status | Score (%) | Delta vs original40 (pp)']
    lines.extend(' | '.join(str(r.get(k,'pending')) for k in ['kind','variant','benchmark','status','score_percent','delta_vs_original40_pp']) for r in rows)
    (ROOT/'controlled-ablation-results.txt').write_text('\n'.join(lines)+'\n')
    # A separate combined artifact preserves the original master table while
    # adding only completed, audited controls and supplementary benchmarks.
    master=read(RESEARCH/'origin-all-model-results.json') if (RESEARCH/'origin-all-model-results.json').exists() else read(Path('/path/to/workspace/PyVision-RL/reproduction/research/20261021-improved-rl/all-model-benchmark-results.json'))
    combined=list(master['results']);seen={(r['run'],r['benchmark']) for r in combined}
    additions=[r for r in rows if r['status']=='passed' and r.get('output')]
    for p in (ROOT/'manifests').glob('algorithm-ablation-supplementary-*.json'):
        item=read(p)
        if item.get('status')=='passed':additions.append(item['result'])
    for row in additions:
        key=(Path(row['output']).name,row['benchmark'])
        if key in seen:continue
        extra=dict(row,run=key[0],status='record_audit_passed',protocol_matches_sft=True)
        combined.append(extra);seen.add(key)
    write(ROOT/'manifests/all-model-results-with-ablations.json',dict(observed_utc=result['observed_utc'],results=combined))
    buf=io.StringIO();fields=['run','benchmark','score_percent','metric','micro_percent','native_vsi_percent','records','repeat','status','protocol_matches_sft','output']
    writer=csv.DictWriter(buf,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(combined)
    (ROOT/'all-model-results-with-ablations.csv').write_text(buf.getvalue())
    return result


if __name__=='__main__':
    for key in ['NO_PROXY','no_proxy']:os.environ[key]='127.0.0.1,localhost'
    if len(sys.argv)>1:
        gpu=sys.argv[1]
        with (ROOT/f'factorial-evaluation-gpu{gpu}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);worker(gpu)
    else:
        while True:
            report();time.sleep(30)
