"""Execute algorithm controls and factorial ablations with bounded scratch storage."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from archive_training_logs import archive, process_identity
from download_preparation_queue import write, digest
from run_training import ROOT, PYTHON, assert_idle, VARIANT_FEATURES

PLAN=ROOT/'manifests/algorithm-ablation-plan.json'
STATUS=ROOT/'manifests/algorithm-ablation-status.json'


def read(path):
    return json.loads(Path(path).read_text())


def live(record):
    return bool(record) and process_identity(record['pid'])==str(record['start_time_ticks'])


def start(command,log):
    with log.open('ab') as stream:
        p=subprocess.Popen(command,cwd=ROOT.parent,stdout=stream,stderr=subprocess.STDOUT,
                           start_new_session=True,stdin=subprocess.DEVNULL)
    return p,dict(pid=p.pid,start_time_ticks=process_identity(p.pid),command=command)


def compact_new_optimizer(job):
    """Keep exact FP32 final models; remove only newly-created scratch Adam tensors."""
    expected=Path('/tmp/pyvision-rl-storage/ablation-checkpoints')/job['name']/f"global_step_{job['target_step']}"
    checkpoint=Path(job['final_checkpoint']).resolve()
    if checkpoint!=expected or not job['name'].startswith('ablation-'):
        raise ValueError('Refusing optimizer scratch cleanup outside this new ablation')
    if read(ROOT/'manifests'/f"{job['name']}-final-update-audit.json")['status']!='passed':
        raise ValueError('Final optimizer/weight update audit required')
    validation=read(ROOT/'manifests'/f"{job['eval_name']}-export-validation.json")
    if (validation['status']!='passed' or validation.get('model')!=job['model']
        or not validation.get('model_shard_sha256')
        or any(digest(Path(job['model'])/name)!=sha for name,sha in validation['model_shard_sha256'].items())):
        raise ValueError('All-parameter export validation required before cleanup')
    import torch
    summaries=[]
    paths=sorted((checkpoint/'actor').glob('optim_world_size_2_rank_*.pt'))
    if len(paths)!=2:
        raise ValueError('Expected both newly saved optimizer shards')
    if any(p.is_symlink() or p.stat().st_nlink!=1 for p in paths):
        raise ValueError('Optimizer scratch must be owned regular files, not links')
    for path in paths:
        state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        steps=[float(getattr(v['step'],'_local_tensor',v['step']).item()) for v in state['state'].values() if 'step' in v]
        sha=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(16*1024**2),b''):sha.update(chunk)
        summaries.append(dict(filename=path.name,bytes=path.stat().st_size,sha256=sha.hexdigest(),
                              parameter_states=len(state['state']),optimizer_steps=sorted(set(steps)),
                              learning_rates=[float(g['lr']) for g in state['param_groups']]))
        del state
    receipt=dict(status='validated_optimizer_scratch_summary',full_optimizer_state_retained=False,
                 resumable_with_optimizer=False,retained='FP32 model shards, BF16 export, RNG/data state, full logs and audits',
                 checkpoint=str(checkpoint),shards=summaries)
    write(checkpoint/'optimizer-retention.json',receipt)
    write(ROOT/'manifests'/f"{job['name']}-optimizer-retention.json",receipt)
    for path in paths:path.unlink()


def finish_job(job):
    import re
    log=ROOT/'logs'/f"{job['name']}.log"
    steps=re.findall(r'\bstep:(\d+) - ',log.read_text(errors='replace')) if log.exists() else []
    job['last_completed_step']=int(steps[-1]) if steps else job['initial_step']
    job['additional_steps_completed']=job['last_completed_step']-job['initial_step']
    out=ROOT/'outputs'/job['name']
    ray=Path(job['ray_root'])/'ray/session_latest'
    if ray.exists() and (out/'invocation.json').exists() and time.time()-job.get('last_archive',0)>120:
        archive(ray,out);job['last_archive']=time.time()
    if live(job['process']):return
    receipt=read(out/'exit-status.json') if (out/'exit-status.json').exists() else {}
    tracker=out/'checkpoints/latest_checkpointed_iteration.txt'
    if receipt.get('returncode')!=0 or not tracker.exists() or int(tracker.read_text())!=job['target_step']:
        raise RuntimeError('Ablation training did not complete successfully: '+str(receipt))
    job['final_checkpoint']=str(out/'checkpoints'/f"global_step_{job['target_step']}")
    audit=ROOT/'manifests'/f"{job['name']}-final-update-audit.json"
    with (ROOT/'logs'/f"{job['name']}-final-audit.log").open('ab') as stream:
        result=subprocess.run([str(PYTHON),str(ROOT/'scripts/validate_research_update.py'),
                               '--kind',job['kind'],'--checkpoint',job['final_checkpoint'],'--report',str(audit)],
                              cwd=ROOT.parent,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Final update audit failed')
    try:os.killpg(job['process']['pid'],signal.SIGTERM)
    except ProcessLookupError:pass
    job.update(status='export_queued',finished_utc=datetime.now(timezone.utc).isoformat())


def main():
    plan=read(PLAN)
    state=read(STATUS) if STATUS.exists() else dict(jobs=plan['jobs'],exports={})
    children={};preparation=None
    write(ROOT/'manifests/algorithm-ablation-controller-process.json',dict(pid=os.getpid(),start_time_ticks=process_identity(os.getpid())))
    while True:
        busy=set()
        for job in state['jobs']:
            if job['status']=='training':
                if job['id'] in children:children[job['id']].poll()
                busy.add(tuple(job['gpus']))
                try:finish_job(job)
                except Exception as exc:job.update(status='needs_attention',error=str(exc))
        # GPU4/6 evaluate completed models. Two training lanes occupy 0/2 and 1/3.
        # Exports and scratch compaction happen in a separate CPU process so a
        # completed job never blocks launch of the next ready training job.
        if preparation is not None:
            process,job_id=preparation
            if process.poll() is not None:
                job=next(j for j in state['jobs'] if j['id']==job_id)
                receipt=ROOT/'manifests'/f"{job['name']}-preparation.json"
                if process.returncode==0 and receipt.exists():
                    job.update(read(receipt));job['status']='evaluation_queued'
                else:job.update(status='needs_attention',error='CPU export/validation failed; inspect preparation log')
                preparation=None
        if preparation is None:
            ready=next((j for j in state['jobs'] if j['status']=='export_queued'),None)
            if ready:
                path=ROOT/'manifests'/f"{ready['name']}-preparation-input.json";write(path,ready)
                p,meta=start([str(PYTHON),str(Path(__file__)),'--prepare',str(path)],ROOT/'logs'/f"{ready['name']}-preparation.log")
                ready.update(status='exporting',preparation_process=meta);preparation=(p,ready['id'])
        for pair in [('0','2'),('1','3')]:
            if pair in busy:continue
            try:assert_idle(list(pair))
            except RuntimeError:continue
            ready=next((j for j in state['jobs'] if j['status']=='queued' and tuple(j['preferred_gpus'])==pair),None)
            if ready is None:ready=next((j for j in state['jobs'] if j['status']=='queued'),None)
            if ready is None:continue
            storage=os.statvfs('/tmp')
            if storage.f_bavail*storage.f_frsize<220*1024**3:
                ready['waiting_reason']='Checkpoint headroom below 220 GiB; waiting for completed optimizer scratch compaction';continue
            ready['gpus']=list(pair)
            command=[str(PYTHON),'-u',str(ROOT/'scripts/run_training.py'),'--kind',ready['kind'],'--mode','full',
                     '--gpus',','.join(pair),'--name',ready['name'],'--research-variant',ready['variant'],
                     '--resume-from',ready['initial_checkpoint'],'--steps',str(ready['target_step']),
                     '--save-freq','5','--optimizer-foreach','false','--checkpoint-dir',ready['checkpoint_root'],
                     '--ray-temp-dir',ready['ray_root']]
            p,meta=start(command,ROOT/'logs'/f"{ready['name']}.log");children[ready['id']]=p
            ready.update(status='training',process=meta,started_utc=datetime.now(timezone.utc).isoformat())
        # Independent evaluation workers keep the statistics and sidecar status.
        for job in state['jobs']:
            sidecar=ROOT/'manifests'/f"{job['name']}-evaluation.json"
            if sidecar.exists() and job['status'] in ['evaluating','evaluation_queued']:
                record=read(sidecar)
                if record['status']=='complete':job.update(status='complete',evaluation=record)
                elif record['status']=='needs_attention':job.update(status='needs_attention',error=record.get('error'))
        state['observed_utc']=datetime.now(timezone.utc).isoformat();write(STATUS,state)
        time.sleep(20)


def prepare(path):
    from evaluate_research_queue import export_model
    job=read(path)
    model=export_model(job['final_checkpoint'],job['kind'],job['eval_name'])
    job['model']=str(model)
    compact_new_optimizer(job)
    write(ROOT/'manifests'/f"{job['name']}-preparation.json",dict(model=str(model),optimizer_state_retained=False))


if __name__=='__main__':
    for key in ['NO_PROXY','no_proxy']:os.environ[key]='127.0.0.1,localhost'
    if len(sys.argv)>1 and sys.argv[1]=='--prepare':
        prepare(Path(sys.argv[2]))
    else:
        with (ROOT/'algorithm-ablation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
