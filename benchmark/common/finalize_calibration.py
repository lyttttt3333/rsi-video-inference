"""Lightweight owner-side publication of measured public baseline statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('--model', choices=['minimax_h3','sana_video2'], required=True)
p.add_argument('--run', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
slug = {'minimax_h3':'minimax-h3-hot-inference','sana_video2':'sana-video2-hot-inference'}[a.model]
task = root / 'benchmark/tasks' / slug
result = json.loads((a.run / 'results/reward.json').read_text())
if result['invalid'] or not result.get('calibration_complete'):
    raise SystemExit('Calibration did not pass')
ids = {r['id'] for r in json.loads((task / 'environment/validation/cases.json').read_text())}
rows = [r for r in result['cases'] if r['id'] in ids]
if len(rows) != len(ids):
    raise SystemExit('Public result set mismatch')
target = task / 'environment/baseline/baseline_val_reward.json'
target.write_text(json.dumps(dict(version=1, split='validation', reward=dict(
    direction='higher_better', mean=statistics.mean(r['reward'] for r in rows),
    sample_std=None, runs=1)), indent=2) + '\n')
manifest = task / 'task.toml'
calibration = ('ONE_SHARED_WARMUP_ONE_TIMING_PER_CASE' if a.model == 'minimax_h3'
               else 'ONE_FULL_RUN_THREE_TIMINGS_PER_CASE')
manifest.write_text(manifest.read_text().replace('baseline_calibration = "NOT_MEASURED"',
    f'baseline_calibration = "{calibration}"').replace(
    'draft-unallocated-uncalibrated', 'draft-unallocated-local-calibrated'))
paths = ['environment/baseline/baseline.sh', 'environment/baseline/baseline_val_reward.json',
         'environment/validation/val.sh', 'tests/test.sh']
(task / 'checksums.sha256').write_text(''.join(
    f'{hashlib.sha256((task / name).read_bytes()).hexdigest()}  {name}\n' for name in paths))
print(target)
