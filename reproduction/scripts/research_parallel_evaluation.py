"""Use free GPUs for independent benchmark evaluations and reconcile statistics."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from archive_training_logs import process_identity
from download_preparation_queue import write
from evaluate_research_queue import export_model, observe, collect, ROOT, STATE
from pipeline import evaluate_suite

PLAN = [
    ('video-ppo2', 'vsibench', '0', 19000),
    ('image-core', 'hrbench-4k', '1', 19001),
    ('image-core', 'vstar', '3', 19003),
    ('image-ppo2', 'hrbench-4k', '2', 19002),
    ('image-ppo2', 'vstar', '2', 19002),
]


def read(path):
    return json.loads(Path(path).read_text())


def status_path(job_id, benchmark):
    return ROOT/'manifests'/f'research-parallel-{job_id}-{benchmark}-status.json'


def worker(args):
    path=status_path(args.job,args.benchmark)
    name='research-eval-'+args.job+'-20261021'
    item=dict(status='running',job_id=args.job,benchmark=args.benchmark,gpu=args.gpu,
              model=args.model,prefix_path=str(ROOT/'outputs'/name),
              pid=os.getpid(),start_time_ticks=process_identity(os.getpid()))
    write(path,item)
    try:
        for attempt in range(2):
            try:
                evaluate_suite(Path(args.model),[args.benchmark],name,gpu=args.gpu,port=args.port,update_state=False)
                break
            except RuntimeError:
                if attempt:raise
                time.sleep(10)
        result,_=observe(ROOT/'outputs'/(name+'-'+args.benchmark),args.benchmark,args.model)
        item.update(status='complete',result=result)
    except Exception as exc:
        item.update(status='needs_attention',error=str(exc))
    finally:
        item['observed_utc']=datetime.now(timezone.utc).isoformat()
        write(path,item)


def reconcile():
    state=read(STATE)
    sidecars=[]
    for path in (ROOT/'manifests').glob('research-parallel-*-status.json'):
        item=read(path)
        if 'job_id' not in item:continue
        sidecars.append(item)
        method=state['methods'].setdefault(item['job_id'],{})
        method.update(model=item['model'],prefix_path=item['prefix_path'])
        method.setdefault('parallel_benchmarks',{})[item['benchmark']]=item['status']
    queue=read(ROOT/'manifests/research-experiment-queue.json')
    collect(queue,state)
    write(ROOT/'manifests/research-parallel-evaluation-status.json',dict(
        observed_utc=datetime.now(timezone.utc).isoformat(),benchmarks=sidecars,
        serial_current=state.get('current'),scope='Parallel evaluation on GPUs 0,1,2,3; existing GPU4 evaluation continues'))


def main():
    queue=read(ROOT/'manifests/research-experiment-queue.json')
    jobs={j['id']:j for j in queue['jobs']}
    models={};children=[]
    for job_id,benchmark,gpu,port in PLAN:
        path=status_path(job_id,benchmark)
        if path.exists():
            old=read(path)
            if old['status']=='complete' or process_identity(old.get('pid',-1))==old.get('start_time_ticks'):
                continue
        job=jobs[job_id]
        if job['status']!='trained':
            raise RuntimeError('Only completed and audited 40-step jobs may enter this evaluation: '+job_id)
        if job_id not in models:
            models[job_id]=export_model(job['final_checkpoint'],job['kind'],'research-eval-'+job_id+'-20261021')
        command=[sys.executable,'-u',str(Path(__file__)),'--job',job_id,'--benchmark',benchmark,
                 '--model',str(models[job_id]),'--gpu',gpu,'--port',str(port)]
        # A single GPU runs HR4K and VStar sequentially. Others start without
        # waiting for the ongoing GPU4 benchmark.
        if job_id=='image-ppo2' and benchmark=='vstar':
            for previous,child in children:
                if previous==('image-ppo2','hrbench-4k'):
                    while child.poll() is None:
                        reconcile();time.sleep(20)
        with (ROOT/'logs'/f'research-parallel-{job_id}-{benchmark}.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(((job_id,benchmark),child))
    while any(child.poll() is None for _,child in children):
        reconcile();time.sleep(20)
    # Keep reconciling after worker completion because the original serial
    # evaluator may still finish later and publish a fresh snapshot.
    while True:
        reconcile();time.sleep(20)


if __name__=='__main__':
    for variable in ['NO_PROXY','no_proxy']:
        os.environ[variable]='127.0.0.1,localhost'
    parser=argparse.ArgumentParser()
    parser.add_argument('--job');parser.add_argument('--benchmark');parser.add_argument('--model')
    parser.add_argument('--gpu');parser.add_argument('--port',type=int)
    args=parser.parse_args()
    if args.job:
        worker(args)
    else:
        with (ROOT/'research-parallel-evaluation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main()
