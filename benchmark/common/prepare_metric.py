"""Run on a CPU compute node with the native runtime, never on login nodes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Requires a Slurm compute node')
root = Path('/metric-assets')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target',
                str(root / 'python'), 'lpips==0.1.4'], check=True)
os.environ['TORCH_HOME'] = str(root / 'torch')
sys.path.insert(0, str(root / 'python'))
import torch
torch.set_num_threads(1)
import lpips
model = lpips.LPIPS(net='alex', version='0.1').eval()
with torch.inference_mode():
    x = torch.zeros(1,3,64,64)
    identity = model(x,x).item()
    different = model(x, torch.ones_like(x)).item()
assert abs(identity) < 1e-6 and different > 0
assets = {}
for path in [root / 'torch/hub/checkpoints/alexnet-owt-7be5be79.pth',
             root / 'python/lpips/weights/v0.1/alex.pth']:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1048576), b''):
            h.update(chunk)
    assets[str(path.relative_to(root))] = h.hexdigest()
(root / 'manifest.json').write_text(json.dumps(dict(lpips='0.1.4', net='alex', version='0.1',
    hashes=assets, identity=identity, different=different), indent=2))
print('Offline LPIPS assets and CPU identity/difference tests passed', flush=True)
