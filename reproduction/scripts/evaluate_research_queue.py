"""Evaluate completed 40-step methods with the frozen evaluator and tabulate evidence."""
from collections import Counter
from datetime import datetime, timezone
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import re
import sys
import time

from download_preparation_queue import write, digest
from run_training import ROOT
from pipeline import evaluate_suite, run
from evaluate_latest_local_checkpoints import complete_inference_assets
from evaluation_evidence import audit_records
from run_qwen_matched_controls import compare_protocol
from collect_results import vsi_task_metrics
from run_research_experiments import SPEC

OLD = Path('/tmp/pyvision-rl-runtime-20261019/PyVision-RL/reproduction')
STATE = ROOT/'manifests/research-evaluation-status.json'


def read(path):
    return json.loads(Path(path).read_text())


def observe(folder, benchmark, model=None):
    config, summary = read(folder/'invocation.json'), read(folder/'summary.json')
    evidence = audit_records(folder, config, summary, ROOT/'data/evaluation'/f'{benchmark}.json')
    if model and {Path(m['root']).resolve() for m in config.get('policy_models',[])} != {Path(model).resolve()}:
        raise ValueError('Evaluated policy differs from the selected checkpoint')
    row = dict(benchmark=benchmark, score_percent=100*summary['score_mean'],
               metric='mean answer score', records=evidence['records_checked'], output=str(folder), evidence=evidence)
    if benchmark=='vsibench':
        row.update(score_percent=100*vsi_task_metrics(summary)['task_macro_mean'], metric='released PyVision eight-task macro')
        from native_vsi_metrics import rescore
        row['native_vsi_percent'] = rescore(folder,summary)['metrics']['overall']
    elapsed, turns = [], []
    for p in folder.glob('*-repeat*.json'):
        if not re.fullmatch(r'.+-repeat\d+\.json',p.name):
            continue
        record=read(p)
        if isinstance(record.get('elapsed_seconds'),(float,int)):
            elapsed.append(record['elapsed_seconds'])
        if isinstance(record.get('generation_turns'),(float,int)):
            turns.append(record['generation_turns'])
    row['mean_sample_elapsed_seconds'] = sum(elapsed)/len(elapsed) if elapsed else None
    row['mean_generation_turns'] = sum(turns)/len(turns) if turns else None
    row['model_failure_records'] = summary.get('failures')
    return row, config


def training_metrics(job):
    log = ROOT/'logs'/f"{job['run_name']}.log"
    by_step = {}
    if log.exists():
        for line in log.read_text(errors='replace').splitlines():
            if 'actor/pg_loss:' not in line:
                continue
            values={k:float(v) for k,v in re.findall(r'([\w/]+):(-?\d+(?:\.\d+)?)',line)}
            if 'step' in values:
                by_step[int(values['step'])] = values
    rows=[v for step,v in by_step.items() if job['initial_global_step']<step<=job['target_global_step']]
    total=sum(v.get('timing_s/step',0) for v in rows)
    errors=total_selected=0
    if job.get('output_root'):
        for p in (Path(job['output_root'])/'rollouts').glob('*.json'):
            if p.stem.isdigit() and job['initial_global_step']<int(p.stem)<=job['target_global_step']:
                records=read(p)
                total_selected+=len(records)
                errors+=sum(r.get('end_reason')==5 for r in records)
    return dict(logged_new_steps=len(rows), logged_step_seconds=total,
                estimated_two_gpu_hours_from_step_timers=total*2/3600,
                selected_training_trajectory_count=total_selected,
                selected_training_tool_error_fraction=errors/total_selected if total_selected else None,
                cost_scope='Step timers include generation/update/validation/checkpointing; excludes cold startup. Error fraction is selected trajectories, not the full candidate pool.')


def collect(queue, state):
    results=[]
    for job in queue['jobs']:
        for benchmark in job['evaluation_benchmarks']:
            row=dict(id=job['id'],kind=job['kind'],method=job['method'],additional_steps=40,
                     training_status=job['status'],evaluation_status='pending',benchmark=benchmark)
            item=state['methods'].get(job['id'],{})
            folder=Path(item.get('prefix_path','/nonexistent')).with_name(Path(item.get('prefix_path','/nonexistent')).name+'-'+benchmark)
            try:
                if not (folder/'summary.json').exists():
                    results.append(row)
                    continue
                observed,config=observe(folder,benchmark,item.get('model'))
                row.update(observed,evaluation_status='passed',**training_metrics(job))
                initial=state['initial'][job['kind']].get('outputs',{}).get(benchmark)
                if initial:
                    baseline,base_config=observe(Path(initial),benchmark,state['initial'][job['kind']].get('model'))
                    compare_protocol(config,base_config)
                    row.update(initial_score_percent=baseline['score_percent'],delta_vs_initial_pp=row['score_percent']-baseline['score_percent'])
                core=state['methods'].get(job['kind']+'-core',{})
                if job['method']!='core' and core.get('prefix_path'):
                    core_folder=Path(core['prefix_path']+'-'+benchmark)
                    if (core_folder/'summary.json').exists():
                        control,control_config=observe(core_folder,benchmark,core.get('model'))
                        compare_protocol(config,control_config)
                        row.update(core40_score_percent=control['score_percent'],delta_vs_core40_pp=row['score_percent']-control['score_percent'])
                if benchmark=='vsibench' and initial:
                    row['native_vsi_delta_vs_initial_pp']=row['native_vsi_percent']-baseline['native_vsi_percent']
            except Exception as exc:
                # A provenance/protocol failure must not leave an apparently
                # accepted numeric score in the CSV or headline table.
                for key in list(row):
                    if key.endswith('_percent') or key.endswith('_pp'):
                        row.pop(key)
                row.update(evaluation_status='failed_audit',error=str(exc))
            results.append(row)
    report=dict(observed_utc=datetime.now(timezone.utc).isoformat(),results=results,
                scope='Exploratory single-seed 40 additional global steps per method; signed deltas are not statistical significance.',
                effectiveness='Compare against core40 at equal rollout-step budget, plus each fixed starting checkpoint; PPO2 performs twice the optimizer updates.',
                failures='Partial, mismatched-protocol and infrastructure-failed runs are not accepted scores.')
    write(ROOT/'manifests/research-results.json',report)
    columns=['id','benchmark','training_status','evaluation_status','additional_steps','score_percent','initial_score_percent','delta_vs_initial_pp','core40_score_percent','delta_vs_core40_pp','native_vsi_percent','estimated_two_gpu_hours_from_step_timers','selected_training_tool_error_fraction','mean_sample_elapsed_seconds','mean_generation_turns','output','error']
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    (ROOT/'research-results.csv').write_text(buf.getvalue())
    lines=['40-step research results (pending is not zero; one seed; no significance claim)',
           'Method | Benchmark | Training | Evaluation | Score (%) | Delta vs core40 (pp)']
    for row in results:
        lines.append(' | '.join(str(row.get(k,'pending')) for k in ['id','benchmark','training_status','evaluation_status','score_percent','delta_vs_core40_pp']))
    (ROOT/'research-results.txt').write_text('\n'.join(lines)+'\n')
    return report


def export_model(checkpoint,kind,name):
    source=ROOT/'models'/f'PyVision-{kind.capitalize()}-7B-SFT'
    model=ROOT/'outputs'/f'{name}-export'/'merged-model'
    validation=ROOT/'manifests'/f'{name}-export-validation.json'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT.parent/'verl_agents'),OMP_NUM_THREADS='8')
    if model.exists():
        record=read(validation) if validation.exists() else {}
        if record.get('status')!='passed' or not record.get('model_shard_sha256') or any(digest(model/f)!=sha for f,sha in record['model_shard_sha256'].items()):
            raise ValueError('Existing model export lacks matching complete validation; preserve it for diagnosis')
    else:
        run([sys.executable,str(ROOT.parent/'verl_agents/scripts/model_merger.py'),'--backend','fsdp',
             '--local_dir',str(Path(checkpoint)/'actor'),'--hf_model_path',str(source),'--target_dir',str(model)],name+'-merge.log',env=env)
    assets=complete_inference_assets(model,source)
    write(ROOT/'manifests'/f'{name}-assets-validation.json',assets)
    run([sys.executable,str(ROOT/'scripts/validate_merged_model.py'),'--model',str(model),
         '--source-model',str(source),'--report',str(validation)],name+'-model-validation.log',env=env)
    return model


def main():
    if STATE.exists():
        state=read(STATE)
    else:
        initial_image_prefix=OLD/'outputs/official-local-image-checkpoint-20261020T092620Z-step20'
        state=dict(status='running',initial={'image':dict(status='complete',outputs={b:str(initial_image_prefix)+'-'+b for b in ['hrbench-4k','vstar']}),
                                            'video':dict(status='queued',outputs={})},methods={})
    state['initial']['image']['model']=read(OLD/'manifests/local-checkpoint-evaluation-selection.json')['checkpoints']['image']['model']
    state['process']=dict(pid=os.getpid(),start_time_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19])
    for b in ['hrbench-4k','vstar']:
        observe(Path(state['initial']['image']['outputs'][b]),b,state['initial']['image']['model'])
    while True:
        queue=read(ROOT/'manifests/research-experiment-queue.json')
        collect(queue,state)
        pending=None
        initial=state['initial']['video']
        if initial['status']=='queued':
            pending=dict(id='video-initial-checkpoint',kind='video',checkpoint=SPEC['video']['checkpoint'],benchmarks=['vsibench'],item=initial)
        else:
            for job in queue['jobs']:
                item=state['methods'].setdefault(job['id'],dict(status='waiting_for_training'))
                if job['status']=='trained' and item['status'] in ['waiting_for_training','queued']:
                    pending=dict(id=job['id'],kind=job['kind'],checkpoint=job['final_checkpoint'],benchmarks=job['evaluation_benchmarks'],item=item)
                    break
        if pending:
            item=pending['item'];name='research-eval-'+pending['id']+'-20261021'
            item.update(status='exporting',checkpoint=pending['checkpoint'])
            state['current']=pending['id'];write(STATE,state)
            try:
                if os.statvfs('/tmp').f_bavail*os.statvfs('/tmp').f_frsize < 120*1024**3:
                    raise RuntimeError('Insufficient export headroom; existing checkpoints are preserved')
                model=export_model(pending['checkpoint'],pending['kind'],name)
                item.update(status='evaluating',model=str(model),prefix_path=str(ROOT/'outputs'/name))
                write(STATE,state)
                for attempt in range(2):
                    try:
                        evaluate_suite(model,pending['benchmarks'],name,gpu='4',port=18974,update_state=False)
                        break
                    except RuntimeError:
                        if attempt:raise
                        time.sleep(10)
                for b in pending['benchmarks']:
                    observe(ROOT/'outputs'/(name+'-'+b),b,model)
                item.update(status='complete',outputs={b:str(ROOT/'outputs'/(name+'-'+b)) for b in pending['benchmarks']})
            except Exception as exc:
                item.update(status='needs_attention',error=str(exc))
            state['observed_utc']=datetime.now(timezone.utc).isoformat();write(STATE,state)
            collect(read(ROOT/'manifests/research-experiment-queue.json'),state)
        else:
            state['observed_utc']=datetime.now(timezone.utc).isoformat();write(STATE,state)
        time.sleep(20)


if __name__=='__main__':
    for variable in ['NO_PROXY','no_proxy']:
        os.environ[variable]='127.0.0.1,localhost'
    with (ROOT/'research-evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
