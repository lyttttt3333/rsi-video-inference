# RSI video inference

Standalone, single-H100 PyTorch inference for MiniMax-H3 and SANA-Video 2.0,
plus one RSI Bench task for SANA-Video-2.0. The model definitions are expanded
under `inference/*/model`; neither runtime imports Diffusers or SGLang.

This repository contains source code only. Model weights, Enroot images,
teacher tensors, videos, metric weights, run logs and private evaluation cases
are intentionally excluded.

## Layout

- `inference/minimax_h3`: native PyTorch T2VA inference, Dockerfile and Slurm launcher.
- `inference/sana_video2`: native PyTorch T2V inference, Dockerfile and Slurm launcher.
- `tasks/sana-video2-hot-inference`: the self-contained Harbor/Modal RSI Bench
  package. All eight SANA cases are visible test cases.
- `benchmark/common`: legacy local experiment and analysis utilities; it is not
  part of the submitted Harbor task.
- `autoresearch`: reproducible campaign drivers and source snapshots; generated
  journals, run records and metric artifacts are kept outside the public tree.
- `tests`: lightweight source/layout checks. GPU acceptance tests must run in
  a Slurm GPU allocation.

## Public evaluation contract

The benchmark workload uses one H100, batch size one, 960x544 output, 121
frames, and 50 sampling steps. Models load once. The first case runs once as a
shared untimed warmup, then all eight cases are timed once. Timing covers text
processing, denoising, and decoded RGB output.

The quality-adjusted case reward is:

```text
speedup * max(0, 1 - LPIPS) * max(0, 1 - Excess-tLPx100)
```

The complete split scores zero unless its spatial and temporal gates pass.
Current limits are documented in each task's `task.toml` and `instruction.md`.
The task metadata remains draft until its three-run Modal calibration is
recorded.

## Run the benchmark task

The task images are built remotely by Harbor's Modal backend, so no local
Docker daemon or Enroot image is required:

```bash
harbor run -p "$PWD/tasks/sana-video2-hot-inference" \
  -a codex -m gpt-5.6-sol --ak reasoning_effort=high -e modal -y
```

Pinned public checkpoints are downloaded during the image build and inference
runs offline.

## Build and run

Build a runtime with Docker or another OCI builder from either inference
directory. On an Enroot-enabled Slurm cluster, convert that image to `.sqsh`
outside this repository and set the launcher variables described below.

MiniMax-H3 requires `H3_IMAGE` plus either `MODEL_ROOT`, or both
`H3_DIFFUSERS` and `H3_FL2VA`. SANA requires `SANA_IMAGE`,
`SANA_CHECKPOINT_ROOT`, `SANA_VAE` and `SANA_GEMMA_REPO`. Both launchers also
require `SLURM_ACCOUNT`; outputs default to a local ignored directory.

```bash
SLURM_ACCOUNT=your_account \
H3_IMAGE=/path/to/minimax-h3.sqsh \
MODEL_ROOT=/path/to/models \
bash inference/minimax_h3/launch_shell.sh /bin/bash
```

Inside the image, invoke `python infer.py --help`. All weights are consumed
from read-only mounts and Hugging Face network access is disabled.

## Lightweight checks

These checks do not load models or use a GPU:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m unittest discover -s benchmark/common -p 'test_*.py' -v
```

Do not run model loading, image builds, bulk hashing or benchmarks on a cluster
login node. Use explicitly sized Slurm CPU/GPU allocations.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and license
information.
