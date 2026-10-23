"""Archive complete native periodic validations at their printed precision."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
TAG = '##################### validation metric dict: '


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_continuity(previous, current_step, validation_steps, input_hash):
    if previous and (current_step < previous['completed_training_updates_observed'] or
                     not {row['step'] for row in previous['validations']}.issubset(validation_steps) or
                     previous['validation_input_sha256'] != input_hash):
        raise ValueError('Refusing to replace a validation archive with regressed or different-input evidence')


def parse(text):
    lines = text.splitlines(keepends=True)
    loader = re.findall(r'^Size of val dataloader: (\d+)\s*$', text, re.M)
    entries, previous = [], 0
    for end, line in enumerate(lines):
        match = re.match(r'^step:(\d+) - ', line)
        if not match:
            continue
        step = int(match[1])
        region = lines[previous:end+1]
        metrics = [ast.literal_eval(item.split(TAG, 1)[1]) for item in region if item.startswith(TAG)]
        if metrics:
            if len(metrics) != 1 or not loader:
                raise ValueError('Ambiguous validation metrics or missing loader count')
            values = metrics[0]
            if not isinstance(values, dict) or not values or any(
                    not key.startswith('val/') or not isinstance(value, (float, int)) or
                    not math.isfinite(value) for key, value in values.items()):
                raise ValueError('Invalid native validation metric dictionary')
            markers = [i for i in range(previous, end)
                       if re.match(r'^Dumped generations to .*/rollouts/'+str(step)+r'\.json\s*$', lines[i])]
            if len(markers) != 1:
                raise ValueError('Validation has no unique matching training rollout boundary')
            start = markers[0]
            metadata = [ast.literal_eval(item.split('test_gen_batch meta info: ', 1)[1])
                        for item in lines[start:end] if item.startswith('test_gen_batch meta info: ')]
            if (len(metadata) != int(loader[-1]) or
                    any(item.get('validate') is not True for item in metadata)):
                raise ValueError('Incomplete native validation batch dispatch')
            for key, value in values.items():
                rounded = re.search(r'(?: - )'+re.escape(key)+r':(-?\d+\.\d+)(?: - |\s*$)', line)
                if not rounded or abs(float(rounded[1])-value) > .0005000001:
                    raise ValueError('Full-precision dictionary differs from associated step summary')
            entries.append(dict(step=step, metrics=values, line_start=start+1, line_end=end+1,
                                generation_batches=len(metadata), batch_metadata=metadata,
                                excerpt=''.join(lines[start:end+1])))
        elif re.search(r' - val/[^ :]+:', line):
            raise ValueError('Rounded validation metrics lack their native full-precision dictionary')
        previous = end+1
    return entries


def collect(run, source):
    run, source = Path(run).resolve(strict=True), Path(source).resolve(strict=True)
    invocation_path = run / 'invocation.json'
    invocation = json.loads(invocation_path.read_text())
    validation = Path(invocation['args']['validation']).resolve(strict=True)
    input_data = validation.read_bytes()
    inputs = json.loads(input_data)
    options = dict(arg.split('=', 1) for arg in invocation['command'][3:] if '=' in arg)
    frequency = int(options['trainer.test_freq'])
    if frequency <= 0:
        raise ValueError('This run does not enable periodic validation')
    config_bytes = subprocess.check_output(['git', 'show', invocation['upstream_revision']+
        ':verl_agents/verl/trainer/config/ppo_trainer.yaml'], cwd=ROOT.parent)
    config = yaml.safe_load(config_bytes)
    configured_n = int(options.get('actor_rollout_ref.rollout.val_kwargs.n',
                                  config['actor_rollout_ref']['rollout']['val_kwargs']['n']))
    source_bytes = source.read_bytes()
    text = source_bytes.decode('utf-8')
    completed = [int(value) for value in re.findall(r'^step:(\d+) - ', text, re.M)]
    if not completed:
        raise ValueError('No completed training update in the specified driver log')
    observed = parse(text)
    expected = set(range(((min(completed)+frequency-1)//frequency)*frequency,
                         max(completed)+1, frequency))
    if {entry['step'] for entry in observed} != expected:
        raise ValueError('Completed periodic validations do not match the configured schedule')
    configured_keys = {'val/'+row['data_source'] for row in inputs}
    binding = invocation.get('validation_input')
    binding_scope = 'current_input_observation_only'
    if binding:
        if binding['sha256'] != digest(input_data):
            raise ValueError('Validation input changed since recorded launch')
        binding_scope = 'recorded_prelaunch_input_hash_matches'
    else:
        legacy = ROOT / 'manifests' / (invocation['args']['kind']+'-validation-step20.json')
        if legacy.exists():
            prior = json.loads(legacy.read_text())
            if prior['run'] == run.name:
                if prior['input_sha256'] != digest(input_data):
                    raise ValueError('Validation input differs from prior native validation observation')
                binding_scope = 'matches_prior_step20_observation_not_a_prelaunch_hash'
    target = run / 'training-validation/periodic'
    target.mkdir(parents=True, exist_ok=True)
    path = run / 'training-validation/periodic-index.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    verify_continuity(previous, max(completed), {row['step'] for row in observed}, digest(input_data))
    records = []
    for entry in observed:
        if set(entry['metrics']) != configured_keys:
            raise ValueError('Native validation metrics do not cover configured data sources')
        excerpt = entry.pop('excerpt').encode()
        log = target / f'step-{entry["step"]}.log'
        if log.exists() and log.read_bytes() != excerpt:
            raise ValueError('Refusing to overwrite a different frozen validation excerpt')
        if not log.exists():
            log.write_bytes(excerpt)
        records.append(entry | dict(raw_log=str(log), raw_log_sha256=digest(excerpt),
                                   source=str(source)))
    report = dict(status='completed_validations_archived',
                  observed_utc=datetime.now(timezone.utc).isoformat(), run=run.name,
                  completed_training_updates_observed=max(completed), validation_frequency=frequency,
                  configured_questions=len(inputs), configured_source_counts=dict(Counter(
                      row['data_source'] for row in inputs)), configured_samples_per_question=configured_n,
                  validation_input=str(validation), validation_input_sha256=digest(input_data),
                  input_identity_scope=binding_scope, invocation_sha256=digest(invocation_path.read_bytes()),
                  invocation_path=str(invocation_path),
                  judge_model=invocation.get('judge_selection', {}).get('model_id'),
                  collector_source_sha256=digest(Path(__file__).read_bytes()),
                  upstream_config_sha256=digest(config_bytes), source_log=str(source),
                  source_log_sha256_at_observation=digest(source_bytes), validations=records,
                  full_benchmark_result=False, nonzero_weight_update_proven_by_this_report=False,
                  scope='Only completed step summaries with a matching full-precision native dictionary, '
                        'matching rollout boundary and complete validation batch dispatch are archived. '
                        'An in-progress validation is not a completed result. This monitor is separate '
                        'from final paper/expanded Bench protocols and saved-weight verification.')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(path)
    print(json.dumps(dict(run=run.name, validations_archived=len(records),
                          completed_training_step=max(completed),
                          validation_steps=[row['step'] for row in records])))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--driver-log', type=Path, required=True)
    args = parser.parse_args()
    collect(args.run, args.driver_log)
