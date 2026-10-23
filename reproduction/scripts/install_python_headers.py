"""Extract checksum-pinned Python headers without changing system packages."""
import hashlib
from pathlib import Path
import subprocess
import urllib.request

DEST = Path('/opt/pyvision-tools')
PACKAGE = 'libpython3.10-dev_3.10.12-1~22.04.18_amd64.deb'
SHA256 = 'c058675011ec4b64f5ad803d74b5fe7de0c720c3700683040da68d1fb5b44ed9'

if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / PACKAGE
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        urllib.request.urlretrieve('https://archive.ubuntu.com/ubuntu/pool/main/p/python3.10/' + PACKAGE, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError('Python header package checksum mismatch')
    subprocess.run(['dpkg-deb', '-x', str(archive), str(DEST)], check=True)
