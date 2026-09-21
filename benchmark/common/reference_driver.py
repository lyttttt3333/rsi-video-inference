"""Runs only within an owner Slurm allocation. No agent-trial budget is changed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--mode', required=True)
p.add_argument('--split', required=True)
p.add_argument('--weights', required=True)
a = p.parse_args()
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Requires Slurm compute allocation')
out = Path('/run-output')
(out / 'allocation.json').write_text(json.dumps({k: os.environ.get(k) for k in
    ('SLURM_JOB_ID', 'SLURM_JOB_NODELIST', 'SLURM_GPUS', 'SLURM_JOB_PARTITION')}, indent=2))
cases = json.loads(Path('/cases.json').read_text())
common = ['--model', a.model, '--submission', '/submission', '--weights', a.weights,
          '--output', '/run-output/results']
if a.mode == 'smoke':
    (out / 'cases.json').write_text(json.dumps(cases[:1]))
    command = [sys.executable, '/runner/measure.py', *common, '--cases', '/run-output/cases.json',
               '--warmup', '1', '--repeats', '1', '--save-tensors']
else:
    command = [sys.executable, '/runner/verify.py', *common, '--cases', '/cases.json',
               '--reference', f'/reference-output/references/{a.split}', '--calibrate']
with (out / 'run.log').open('w') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
if (result.returncode == 0 and a.mode == 'calibrate' and a.split == 'all'
        and a.model != 'sana_video2'):
    import shutil
    source = Path('/reference-output/references/all')
    manifest = json.loads((source / 'manifest.json').read_text())
    public_count = 3 if a.model == 'minimax_h3' else 4
    for split, selected in [('public', cases[:public_count]), ('heldout', cases[public_count:])]:
        target = source.parent / split
        target.mkdir(exist_ok=False)
        subset = dict(manifest, cases={r['id']: manifest['cases'][r['id']] for r in selected})
        for row in subset['cases'].values():
            # Same filesystem: avoid duplicating multi-GB tensor storage.
            os.link(source / row['tensor'], target / row['tensor'])
        (target / 'manifest.json').write_text(json.dumps(subset, indent=2))
print((out / 'run.log').read_text()[-5000:], flush=True)
raise SystemExit(result.returncode)
