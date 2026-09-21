# RSI video inference

Standalone, single-H100 PyTorch inference and RSI benchmark scaffolding for
MiniMax-H3 and SANA-Video 2.0. The model definitions are expanded under
`inference/*/model`; neither runtime imports Diffusers or SGLang.

This repository contains source code only. Model weights, Enroot images,
teacher tensors, videos, metric weights, run logs and private evaluation cases
are intentionally excluded.

## Layout

- `inference/minimax_h3`: native PyTorch T2VA inference, Dockerfile and Slurm launcher.
- `inference/sana_video2`: native PyTorch T2V inference, Dockerfile and Slurm launcher.
- `benchmark/common`: hot-latency measurement, LPIPS/Excess-tLP verification,
  scoring, public-reference preparation and campaign budget controls.
- `benchmark/tasks`: the two self-contained RSI task packages. MiniMax-H3
  contains only its three public validation cases; its four final cases are a
  private evaluator overlay. All eight SANA cases are visible test cases.
- `autoresearch`: reproducible campaign drivers, best submitted candidates,
  journals and temporal-flicker analysis.
- `tests`: lightweight source/layout checks. GPU acceptance tests must run in
  a Slurm GPU allocation.

## Public evaluation contract

Both workloads use one H100, batch size one, 960x544 output and 50 sampling
steps. MiniMax-H3 uses 124 frames and SANA uses 121 frames. Models load once.
MiniMax-H3 runs the first case once as the single untimed warmup, then times
each case once. Timing covers text processing, denoising and decoded output;
H3 audio decoding is also included.

The quality-adjusted case reward is:

```text
speedup * max(0, 1 - LPIPS) * max(0, 1 - Excess-tLPx100)
```

The complete split scores zero unless its spatial and temporal gates pass.
Current limits are documented in each task's `task.toml` and `instruction.md`.
The task metadata remains explicitly marked draft/unallocated/uncalibrated.

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

## Private evaluator boundary

The MiniMax-H3 final prompts, enable token, teacher tensors and reference
manifests are not present in this public repository. A benchmark operator must
mount those assets into the separate verifier environment. Deleting a secret
after committing it does not remove it from Git history, so the private
overlay is ignored at the repository root and by the H3 task package.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and license
information.

