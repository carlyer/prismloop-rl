"""Adopt live trainers and execute a persistent queue of 40-step experiments."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from archive_training_logs import archive, process_identity
from download_preparation_queue import write
from run_training import ROOT, PYTHON, assert_idle
from run_research_experiments import SPEC

QUEUE = ROOT/'manifests/research-experiment-queue.json'


def read(path):
    return json.loads(Path(path).read_text())


def alive(metadata):
    return bool(metadata) and process_identity(metadata['pid']) == str(metadata['start_time_ticks'])


def target_step(initial, budget=40):
    if budget != 40:
        raise ValueError('Every method must receive exactly 40 additional global steps')
    return initial + budget


def completed(job):
    return job.get('latest_completed_global_step', job['initial_global_step']) - job['initial_global_step']


def build_command(job, registry, resume):
    record = registry[job['method']]
    if not record.get('validated') or not Path(record['test_receipt']).is_file():
        raise ValueError('Unvalidated research method cannot launch')
    root = Path(record['reproduction_root'])
    return [str(PYTHON), '-u', str(root/'scripts/run_training.py'), '--kind', job['kind'],
            '--mode', 'full', '--gpus', SPEC[job['kind']]['gpus'], '--name', job['run_name'],
            '--resume-from', str(resume), '--steps', str(target_step(job['initial_global_step'], job['additional_global_steps'])),
            '--save-freq', '5', '--optimizer-foreach', 'false',
            '--checkpoint-dir', '/tmp/pyvision-rl-storage/research-checkpoints/'+job['run_name'],
            '--ray-temp-dir', '/tmp/pvr-'+job['id'], *record['launcher_args']]


def launch(job, registry, resume):
    assert_idle(SPEC[job['kind']]['gpus'].split(','))
    command = build_command(job, registry, resume)
    out_root = Path(registry[job['method']]['reproduction_root'])
    with (ROOT/'logs'/f"{job['run_name']}.log").open('ab') as log:
        process = subprocess.Popen(command, cwd=out_root.parent, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    job.update(status='running', output_root=str(out_root/'outputs'/job['run_name']),
               active_ray_root='/tmp/pvr-'+job['id'], current_segment_target=job['target_global_step'],
               process=dict(pid=process.pid, start_time_ticks=process_identity(process.pid), command=command),
               started_utc=datetime.now(timezone.utc).isoformat())
    write(ROOT/'manifests'/f"{job['run_name']}-process.json", job['process'])
    return process


def prune_owned_intermediate_checkpoints(job):
    root = Path(job['output_root'])/'checkpoints'
    expected = Path('/tmp/pyvision-rl-storage/research-checkpoints')/job['run_name']
    if root.resolve() != expected:
        raise ValueError('Checkpoint retention outside the owned research run is forbidden')
    tracker = root/'latest_checkpointed_iteration.txt'
    if not tracker.exists():
        return
    committed = int(tracker.read_text())
    # Rotation across a 20+20 continuation is the same latest-one policy as
    # within a single run. Never remove a checkpoint still being written.
    for candidate in root.glob('global_step_*'):
        if int(candidate.name.rsplit('_', 1)[1]) < committed:
            shutil.rmtree(candidate)


def main():
    children = {}
    write(ROOT/'manifests/research-controller-process.json', dict(pid=os.getpid(), start_time_ticks=process_identity(os.getpid()), controller='persistent_40_step_queue'))
    while True:
        queue = read(QUEUE)
        registry = read(ROOT/'manifests/research-method-registry.json')
        active_kinds = set()
        for job in queue['jobs']:
            if job['status'] != 'running':
                continue
            kind = job['kind']
            active_kinds.add(kind)
            metadata = job.get('process') or job.get('adopted_process')
            job['process'] = metadata
            job.setdefault('output_root', str(ROOT/'outputs'/job['run_name']))
            command = metadata['command']
            job.setdefault('active_ray_root', command[command.index('--ray-temp-dir')+1])
            try:
                process = children.get(job['id'])
                if process is not None:
                    process.poll()
                log = ROOT/'logs'/f"{job['run_name']}.log"
                text = log.read_text(errors='replace') if log.exists() else ''
                steps = re.findall(r'\bstep:(\d+) - ', text)
                job['latest_completed_global_step'] = int(steps[-1]) if steps else job['initial_global_step']
                job['new_updates_completed'] = completed(job)
                out = Path(job['output_root'])
                ray = Path(job['active_ray_root'])/'ray/session_latest'
                if ray.exists() and (out/'invocation.json').exists() and time.time()-job.get('last_archive_unix', 0)>120:
                    archive(ray, out)
                    job['last_archive_unix'] = time.time()
                tracker = out/'checkpoints/latest_checkpointed_iteration.txt'
                if tracker.exists():
                    job['latest_saved_step'] = int(tracker.read_text())
                    prune_owned_intermediate_checkpoints(job)
                if alive(metadata):
                    continue
                receipt = read(out/'exit-status.json') if (out/'exit-status.json').exists() else {}
                segment_target = job['current_segment_target']
                if receipt.get('returncode') != 0 or job.get('latest_saved_step') != segment_target:
                    raise RuntimeError('Trainer stopped without the committed segment checkpoint: '+str(receipt))
                try:
                    os.killpg(metadata['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if segment_target < job['target_global_step']:
                    job.update(status='queued_continuation', resume_checkpoint=str(out/'checkpoints'/f'global_step_{segment_target}'))
                else:
                    job.update(status='trained', final_checkpoint=str(out/'checkpoints'/f'global_step_{segment_target}'),
                               finished_utc=datetime.now(timezone.utc).isoformat())
                    audit = ROOT/'manifests'/f"{job['run_name']}-final-update-audit.json"
                    result = subprocess.run([str(PYTHON), str(ROOT/'scripts/validate_research_update.py'), '--kind', kind,
                                             '--checkpoint', job['final_checkpoint'], '--report', str(audit)],
                                            capture_output=True, text=True)
                    job['final_update_audit'] = str(audit)
                    if result.returncode:
                        raise RuntimeError('Final checkpoint update audit failed: '+result.stderr[-1000:])
                    job['evaluation'] = 'queued'
            except Exception as exc:
                job.update(status='needs_attention', error=str(exc))
        for kind in SPEC:
            if kind in active_kinds:
                continue
            ready = [j for j in queue['jobs'] if j['kind']==kind and j['status'] in ['queued', 'queued_continuation']
                     and j['method'] in registry and registry[j['method']].get('validated')]
            if not ready:
                continue
            job = ready[0]
            try:
                assert_idle(SPEC[kind]['gpus'].split(','))
                if os.statvfs('/tmp').f_bavail*os.statvfs('/tmp').f_frsize < 220*1024**3:
                    job['waiting_reason'] = 'Less than 220 GiB checkpoint headroom; preserve existing evidence'
                    continue
                resume = job.get('resume_checkpoint', SPEC[kind]['checkpoint'])
                children[job['id']] = launch(job, registry, resume)
                job.pop('waiting_reason', None)
            except RuntimeError as exc:
                job['waiting_reason'] = str(exc)
            except Exception as exc:
                job.update(status='needs_attention', error=str(exc))
        queue['observed_utc'] = datetime.now(timezone.utc).isoformat()
        write(QUEUE, queue)
        lines = ['PyVision RL research TODO',
                 'Every method: 40 ADDITIONAL global steps from the same frozen initial checkpoint.',
                 'Image: initial 20 -> target 60; Video: initial 120 -> target 160.',
                 'Planned research methods require implementation and tests before launch.',
                 'Order within each GPU pair follows the list of validated jobs; evaluation uses GPU 4.', '']
        for i, job in enumerate(queue['jobs'],1):
            lines.append(f"{i}. [{job['status']}] {job['id']}: {job['title']} | implementation={job['implementation']} | new steps={completed(job)}/40")
            lines.append('   implement -> test -> train 40 steps -> full frozen evaluation -> compare against core40 and initial checkpoint -> summarize')
        lines += ['', 'Results: research-results.csv / research-results.txt / manifests/research-results.json',
                  'Required analysis: accuracy, per-task scores, tool failures, sample latency and training GPU-hours.',
                  '40-step single-seed results are preliminary; no automatic significance claim.',
                  'Storage: preserve original baselines and results; latest-one rotation applies only within each owned research run.']
        (ROOT/'TODO.txt').write_text('\n'.join(lines)+'\n')
        states = {kind: next((j for j in queue['jobs'] if j['kind']==kind and j['status']=='running'),
                           {'status':'waiting_for_next_validated_method'}) for kind in SPEC}
        write(ROOT/'manifests/research-training-status.json', dict(status='running', states=states,
              additional_steps_per_method=40, original_700_step_reproduction_cancelled=True,
              planned_unimplemented=sum(j['implementation']=='pending_implementation' for j in queue['jobs']),
              observed_utc=queue['observed_utc']))
        time.sleep(20)


if __name__ == '__main__':
    for variable in ['NO_PROXY','no_proxy']:
        os.environ[variable] = '127.0.0.1,localhost'
    with (ROOT/'research-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
