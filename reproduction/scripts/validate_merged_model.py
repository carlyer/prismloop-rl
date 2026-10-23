"""Load merged weights on CPU and verify the complete parameter set."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from download_preparation_queue import digest, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--after-exit-record', type=Path)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    report = dict(status='running', model=str(args.model), source_model=str(args.source_model),
                  scope='CPU weight load and finite-parameter check; no model accuracy measurement')
    try:
        if args.after_exit_record:
            while not args.after_exit_record.exists():
                time.sleep(20)
            exit_record = json.loads(args.after_exit_record.read_text())
            if exit_record['returncode'] != 0:
                raise RuntimeError('Checkpoint merger failed: ' + str(exit_record))
        write(args.report, report)
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForImageTextToText

        index = json.loads((args.model / 'model.safetensors.index.json').read_text())
        saved_dtypes = Counter()
        for filename in sorted(set(index['weight_map'].values())):
            with safe_open(args.model / filename, framework='pt', device='cpu') as shard:
                saved_dtypes.update(shard.get_slice(key).get_dtype() for key in shard.keys())
        exported_config = AutoConfig.from_pretrained(args.model)
        declared_dtype = exported_config.torch_dtype
        declared_sections = {
            name: str(getattr(exported_config, name).torch_dtype)
            for name in ['text_config', 'vision_config'] if getattr(exported_config, name, None) is not None}
        report.update(saved_tensor_dtype_counts=dict(saved_dtypes),
                      declared_torch_dtype=str(declared_dtype), declared_section_dtypes=declared_sections,
                      auto_dtype_load_required=True)
        if (declared_dtype != torch.bfloat16 or set(saved_dtypes) != {'BF16'}
                or any(dtype != 'torch.bfloat16' for dtype in declared_sections.values())):
            raise RuntimeError('Merged BF16 tensor precision and config.torch_dtype do not agree')
        config = AutoConfig.from_pretrained(args.source_model)
        with torch.device('meta'):
            reference = AutoModelForImageTextToText.from_config(config, torch_dtype=torch.bfloat16)
        expected = {k: tuple(v.shape) for k, v in reference.named_parameters()}
        del reference
        model, loading = AutoModelForImageTextToText.from_pretrained(
            args.model, torch_dtype='auto', device_map='cpu',
            low_cpu_mem_usage=True, output_loading_info=True)
        if any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
            raise RuntimeError('Incomplete or mismatched weight load: ' + str(loading))
        actual = {k: tuple(v.shape) for k, v in model.named_parameters()}
        if expected != actual:
            raise RuntimeError('Merged model parameter names/shapes differ from the official SFT architecture')
        loaded_dtypes = Counter(str(p.dtype) for p in model.parameters())
        report['parameter_dtype_counts'] = dict(loaded_dtypes)
        if set(loaded_dtypes) != {'torch.bfloat16'}:
            raise RuntimeError('Automatic dtype loading did not preserve the saved BF16 parameter precision')
        invalid = [name for name, value in model.named_parameters() if not torch.isfinite(value).all().item()]
        if invalid:
            raise RuntimeError('Non-finite model parameters: ' + str(invalid))
        report.update(status='passed', model_class=type(model).__name__,
                      parameter_tensors=len(actual), parameter_elements=sum(p.numel() for p in model.parameters()),
                      parameter_dtype_counts=dict(Counter(str(p.dtype) for p in model.parameters())),
                      all_parameters_finite=True, official_architecture_shapes_match=True,
                      loading_info=loading, model_index_sha256=digest(args.model / 'model.safetensors.index.json'),
                      config_sha256=digest(args.model / 'config.json'))
        index = json.loads((args.model / 'model.safetensors.index.json').read_text())
        report['model_shard_sha256'] = {
            name: digest(args.model / name) for name in sorted(set(index['weight_map'].values()))}
    except Exception as exc:
        report.update(status='failed', error_class=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['observed_utc'] = datetime.now(timezone.utc).isoformat()
        write(args.report, report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
