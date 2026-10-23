"""Avoid scalar-by-scalar CUDA transfers when padding already-tensor rows."""
from pathlib import Path

from download_preparation_queue import ROOT, digest, write

REPO = ROOT.parent
TARGET = REPO / 'verl_agents/verl/utils/torch_functional.py'
ANCHOR = '    padded_response = [tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list)) for sub_list in response]\n'
FAST = '''    # Local reproduction: copy tensor rows once, avoiding per-token device sync.
    # Keep the original inference path for other types/dtypes.
    tensor_rows = all(isinstance(row, torch.Tensor) and row.ndim == 1 for row in response)
    if tensor_rows:
        dtype = response[0].dtype
        compatible_pad = isinstance(pad_token_id, int) or (dtype == torch.float32 and isinstance(pad_token_id, float))
        if dtype in (torch.int64, torch.float32) and compatible_pad and all(row.dtype == dtype for row in response):
            return torch.stack([
                F.pad(row.detach().to(device="cpu"), (0, target_length - len(row)), value=pad_token_id)
                for row in response
            ])
'''


def main():
    backup = ROOT / 'references/torch_functional.original.py'
    text = TARGET.read_text()
    if FAST not in text:
        if text.count(ANCHOR) != 1:
            raise RuntimeError('Upstream tensor-padding source changed')
        if not backup.exists():
            backup.write_text(text)
        TARGET.write_text(text.replace(ANCHOR, FAST + ANCHOR))
    write(ROOT / 'manifests/tensor-padding-patch.json', {
        'target': str(TARGET), 'original_sha256': digest(backup), 'patched_sha256': digest(TARGET),
        'purpose': 'Equivalent tensor padding without per-scalar host/device conversion',
        'rl_algorithm_modified': False, 'equality_evidence': 'manifests/tensor-padding-validation.json'})
    print('Applied tensor padding transfer optimization')


if __name__ == '__main__':
    main()
