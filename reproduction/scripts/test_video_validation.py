"""CPU native media/converter, full validation-dataset and future launch checks."""
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import prepare_video_validation as preparation
import run_training


def arguments(**overrides):
    return SimpleNamespace(**(dict(kind='video', mode='full', validation=None, eval_only=False,
        checkpoint=None, name='video-validation-configuration-audit', gpus='1,4', steps=None,
        max_tool_turns=4, no_tool_reward=False, no_std_sort=False, std_normalization=False,
        optimizer_foreach='auto') | overrides))


def main():
    report = preparation.verify_prepared()
    raw = json.loads(preparation.RAW.read_text())
    prepared = json.loads(preparation.OUTPUT.read_text())
    # Verify the cached metadata preparation against real native decoder/converter calls.
    sys.path.insert(0, str(preparation.ROOT.parent / 'verl_agents'))
    from verl.utils.dataset.rl_dataset_wo_mm_hint import (
        RLHF_wo_mm_hint_Dataset, transfer_to_rl_form_video)
    samples = [next(index for index, row in enumerate(raw) if row['data_source'] == source)
               for source in ['vsi', 'vsi_numerical']]
    for index in samples:
        fresh = transfer_to_rl_form_video([copy.deepcopy(raw[index])], str(preparation.TEMPLATE))
        assert fresh[0] == prepared[index], index

    from transformers import AutoProcessor
    from omegaconf import OmegaConf
    source_model = preparation.ROOT / 'models/PyVision-Video-7B-SFT'
    processor = AutoProcessor.from_pretrained(str(source_model), min_pixels=3136, max_pixels=2000000,
                                            local_files_only=True)
    dataset = RLHF_wo_mm_hint_Dataset([str(preparation.OUTPUT)], processor.tokenizer,
        OmegaConf.create(dict(max_prompt_length=32000, return_raw_chat=True,
                              filter_overlong_prompts=False, prompt_template_path=str(preparation.TEMPLATE))),
        processor=processor)
    lengths, sources = [], Counter()
    for index in range(len(dataset)):
        item = dataset[index]
        assert item['input_ids'].shape == (32000,)
        assert item['attention_mask'].shape == (32000,)
        assert len(item['raw_prompt_ids']) < 32000
        assert item['origin_multi_modal_data'] == {'video': [raw[index]['video_path']]}
        assert item['reward_model']['ground_truth'] == raw[index]['answer']
        assert item['data_source'] == raw[index]['data_source']
        assert item['extra_info']['index'] == int(raw[index]['id'])
        lengths.append(len(item['raw_prompt_ids']))
        sources[item['data_source']] += 1
    assert len(lengths) == 400 and sources == Counter(report['data_sources'])

    args = arguments()
    validation = run_training.resolve_validation(args)
    assert args.validation == str(preparation.OUTPUT) and validation['preparation_receipt']
    command = run_training.build(args)[0]
    opts = dict(argument.split('=', 1) for argument in command[3:])
    assert opts['trainer.total_training_steps'] == '700'
    assert opts['trainer.test_freq'] == '20'
    assert opts['data.val_files'] == '['+str(preparation.OUTPUT)+']'
    assert opts['data.train_files'].endswith('/data/prepared/video_train.json]')
    assert opts['actor_rollout_ref.rollout.agent.tool_using_cumulative_reward_per_turn'] == '0.1'
    assert opts['actor_rollout_ref.rollout.n'] == '8'
    for other in [arguments(mode='smoke'), arguments(kind='image')]:
        assert run_training.resolve_validation(other) is None
        assert other.validation is None
    explicit = arguments(validation=str(preparation.OUTPUT))
    assert run_training.resolve_validation(explicit)['origin'] == 'explicit'
    evaluation = arguments(eval_only=True)
    assert run_training.resolve_validation(evaluation) is None
    try:
        run_training.build(evaluation)
    except ValueError:
        pass
    else:
        raise AssertionError('Eval-only still requires explicit validation')
    with tempfile.TemporaryDirectory(prefix='pyvision-validation-gate-') as temporary:
        stale = copy.deepcopy(report)
        stale['artifacts'][0]['sha256'] = '0'*64
        receipt = Path(temporary) / 'stale-receipt.json'
        receipt.write_text(json.dumps(stale))
        with patch.object(preparation, 'RECEIPT', receipt):
            try:
                preparation.verify_prepared()
            except RuntimeError:
                pass
            else:
                raise AssertionError('Stale validation artifact must be rejected')
    preparation.verify_prepared()
    result = dict(status='passed', observed_utc=datetime.now(timezone.utc).isoformat(),
                  validation_receipt=str(preparation.RECEIPT),
                  validation_receipt_sha256=preparation.digest(preparation.RECEIPT),
                  source_model=str(source_model), model_weights_loaded=False,
                  native_converter_fresh_decode_equivalence_cases=len(samples),
                  actual_native_dataset_items_checked=len(lengths),
                  actual_prompt_tokens=dict(min=min(lengths), max=max(lengths), mean=sum(lengths)/len(lengths)),
                  data_sources=dict(sources), automatic_video_full_validation_verified=True,
                  image_smoke_and_explicit_validation_preserved=True, stale_receipt_rejected=True,
                  future_video_full_command=command,
                  source_sha256={str(path): preparation.digest(path) for path in
                                 [Path(run_training.__file__), Path(preparation.__file__)]},
                  scope='CPU only: actual decoder, native dataset and local processor/tokenizer; '
                        'pure launch configuration. No model weights, training, GPU allocation, '
                        'real inference or benchmark scores.')
    path = preparation.ROOT / 'manifests/video-training-validation400-checks.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='passed', checked_dataset_items=400,
                         fresh_decode_equivalent=len(samples), max_prompt_tokens=max(lengths))))


if __name__ == '__main__':
    main()
