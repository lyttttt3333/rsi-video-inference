# SANA 50-round campaign result

The campaign completed exactly 50 sequential single-H100 rounds. The final and
global winner is round 50, commit
`a2c24818d54fe2bab0556e34720b6b775f5abb27`.

## Winning all-eight metrics

- Mean hot latency: 9.674338 seconds.
- Mean speedup over the 26.407829-second GPU-resident baseline: 2.729684x.
- LPIPS 01-08: 0.16765490, 0.24871833, 0.25976051, 0.32833425,
  0.14275821, 0.15651094, 0.35932734, 0.25370256.
- Mean LPIPS: 0.23959588 (limit 0.30).
- Worst LPIPS: 0.35932734 (limit 0.50).
- Reward: 2.07559659; global quality gate passed.

## Winning implementation

The winner combines GPU-resident Gemma/denoiser/VAE execution with forced Flash
SDPA, max-autotune `torch.compile` with CUDA graphs for the denoiser, full-frame
VAE decode plus fixed-shape cuDNN benchmarking, order-2 DPM-Solver++ with 45
internal steps and flow shift 13.125, and request-local alternating denoiser
output reuse while protecting the first three and final call. The cross-step
cache is reset for every request and never stores or returns completed videos.

## Explored techniques

- DPM-Solver step/order reduction and flow-shift/schedule refinement.
- Periodic, adaptive-window, threshold, parity, scaled-output, and extrapolated
  cross-step diffusion caching.
- Flash SDPA, `torch.compile` modes, CUDA graphs, cuDNN fixed-shape algorithm
  search, TF32/matmul settings, VAE compilation/tiling, and text-embedding or
  text-encoder compilation/caching.
- Three rounds failed and were consumed without retry: r1 harness workdir, r27
  CUDA visibility, and r39 incompatible CFG history shapes. Forty-seven rounds
  completed. Reserved budget was 42,000 GPU-seconds (11.667 H100-hours, below
  12 hours); observed Slurm job time summed to 7.229 hours.

## Artifacts

- Winning source: `workspace/candidate.py`.
- Complete research journal: `journal.md`.
- Generated measurements, ledgers and videos are intentionally excluded from Git.
