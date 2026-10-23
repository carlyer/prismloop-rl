"""Use the supported Transformers auto class in the official model merger."""
from download_preparation_queue import ROOT, digest, write


def main():
    target = ROOT.parent / 'verl_agents/scripts/model_merger.py'
    backup = ROOT / 'references/model_merger.original.py'
    old, new = 'AutoModelForVision2Seq', 'AutoModelForImageTextToText'
    source = target.read_text()
    if old in source:
        if source.count(old) != 3:
            raise RuntimeError('Upstream merger auto-model references changed')
        if not backup.exists():
            backup.write_text(source)
        target.write_text(source.replace(old, new))
    elif source.count(new) != 3 or not backup.exists():
        raise RuntimeError('Unexpected checkpoint merger source')
    write(ROOT / 'manifests/checkpoint-merger-patch.json', {
        'target': str(target), 'original_sha256': digest(backup), 'patched_sha256': digest(target),
        'reason': 'Transformers 4.54 deprecated Vision2Seq.from_config omits keyword arguments',
        'replacement': 'AutoModelForImageTextToText accepts torch_dtype and maps to the same Qwen model class',
        'scope': 'Checkpoint conversion only; training and scoring unchanged',
        'validation': 'manifests/checkpoint-merger-constructor-validation.json'})
    print('Updated official checkpoint merger auto-model factory')


if __name__ == '__main__':
    main()
