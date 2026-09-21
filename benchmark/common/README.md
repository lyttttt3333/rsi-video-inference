# Benchmark tooling

This directory is the canonical implementation of the hot-inference timing and
quality contract shared by the two task packages.

- `measure.py`: load once, warm up, synchronize CUDA and time decoded requests.
- `verify.py`: validate output shapes and calculate LPIPS plus Excess-tLP.
- `score.py`: apply spatial/temporal gates and aggregate quality-adjusted speedup.
- `adapters/`: model-specific candidate adapters.
- `prepare_metric.py`: prepare LPIPS assets on a Slurm CPU node.
- `stage_weights.py`: copy and hash an explicit weight manifest on a CPU node.
- `launch_reference.sh`: public reference smoke/calibration on one H100.
- `campaign.py`: shared 12-GPU-hour budget bookkeeping and Harbor launcher.

The task packages live in `../tasks`. MiniMax-H3 private evaluation assets are
not part of this repository. `sync_verifier_contract.py` updates public task
copies only and never creates a held-out split.

Run the standard-library contract tests with:

```bash
python -m unittest discover -s benchmark/common -p 'test_*.py' -v
```

The task metadata remains draft/unallocated/uncalibrated. These tools do not by
themselves provide adversarial worker isolation or official Harbor deployment.

