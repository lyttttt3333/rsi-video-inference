# SANA GPU-resident all-eight autoresearch policy

- Run exactly 50 sequential SANA-Video 2.0 optimization rounds.
- The immutable starting point is the task's GPU-resident baseline: Gemma,
  denoiser, and VAE remain resident on one H100 after build.
- All eight prompts, `sana_video2-01` through `sana_video2-08`, are visible test
  cases and are evaluated every round. There is no held-out split.
- Each submission gets one warmup generation of case 01, followed by one formal
  timed generation of all eight cases. Loading, compilation, and warmup are
  excluded from latency but consume the research allocation.
- Raw reward is the mean across eight cases of
  `baseline_seconds / candidate_seconds * (1 - LPIPS)`.
- The entire round scores zero unless mean LPIPS across all eight cases is at
  most 0.30 and the worst-case LPIPS is at most 0.50. Both limits are inclusive.
- Use at most one H100. Fifty reservations of 14 minutes cap the campaign below
  the 12 GPU-hour budget. A failed or timed-out job still consumes its round and
  reservation; do not retry it as the same round.
- Methods are unrestricted within the output contract. Actively explore both:
  - lossy methods: cross-step/diffusion/FirstBlock cache, adaptive cache
    thresholds, step skipping, approximate computation, and schedule changes;
  - systems/kernel methods: fusion, SDPA/FlashAttention, torch.compile,
    CUDA graphs, precision/layout changes, and text/VAE/runtime optimization.
- Internet research is allowed. Record URLs plus the concrete mechanism used in
  `journal.md`; prefer primary papers, official docs, and upstream source.
- Never change weights, prompt text, seed, resolution, frame count, requested
  step count, or output schema. Never return a cached completed output or use
  prompt/seed-specific completed-output lookup.
- Work only in `workspace/`. Keep it self-contained. Before starting a round,
  commit the proposed source state and keep the worktree clean.
- For every round: write a hypothesis, call `supervisor.py start`, run the
  emitted command, call `supervisor.py finish`, and record speed, per-case
  LPIPS, mean/worst LPIPS, reward, and conclusion in `journal.md`.
- Keep improvements. Revert regressions in the dedicated workspace before the
  next experiment. Preserve every trial in Git history and every measurement
  under `runs/`.

