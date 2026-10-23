"""Headers used by Triton in the isolated Python 3.10 environment."""
from pathlib import Path


def configure_runtime(env):
    root = Path('/opt/pyvision-tools/usr/include')
    if (root / 'python3.10/Python.h').exists():
        paths = [str(root / 'python3.10'), str(root)]
        if env.get('CPATH'):
            paths.append(env['CPATH'])
        env['CPATH'] = ':'.join(paths)
    return env
