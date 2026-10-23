"""Allow an explicit AdamW execution mode for memory-limited engineering runs."""
from download_preparation_queue import ROOT, digest, write


def main():
    target = ROOT.parent / 'verl_agents/verl/workers/fsdp_workers.py'
    backup = ROOT / 'references/fsdp_workers.original.py'
    old = '                weight_decay=optim_config.get("weight_decay", 1e-2),\n'
    new = old + '                foreach=optim_config.get("foreach", None),  # Local optional execution setting\n'
    source = target.read_text()
    if new not in source:
        if source.count(old) != 1:
            raise RuntimeError('Upstream optimizer construction changed')
        if not backup.exists():
            backup.write_text(source)
        target.write_text(source.replace(old, new))
    write(ROOT / 'manifests/optimizer-configuration-patch.json', {
        'target': str(target), 'original_sha256': digest(backup), 'patched_sha256': digest(target),
        'default': 'None preserves the upstream AdamW default',
        'engineering_override': 'foreach=False reduces temporary tensor-list storage',
        'algorithm': 'AdamW; learning rate, betas, epsilon and weight decay unchanged',
        'validation': 'manifests/adamw-execution-equivalence.json'})
    print('Enabled optional AdamW foreach configuration')


if __name__ == '__main__':
    main()
