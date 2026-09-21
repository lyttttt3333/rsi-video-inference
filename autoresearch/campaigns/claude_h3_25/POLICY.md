# Claude Opus MiniMax-H3 25-round autoresearch policy

- Run exactly 25 sequential MiniMax-H3 optimization rounds on one H100.
- The immutable starting point is the standalone native-PyTorch `dense +
  no-cache` implementation copied into `workspace/`: all 49 denoising
  iterations execute the full transformer stack. Persistent fused-QKV weights,
  exact AdaLN schedule precomputation, and TF32-high matmul dispatch are part of
  this baseline.
- Evaluate exactly the three visible cases `minimax_h3-02` through
  `minimax_h3-04` every round. Case 01 was removed by the task owner.
- Cases `minimax_h3-05` through `minimax_h3-08` are held out. They are not
  mounted into round jobs and must not be read, inferred, reconstructed, or
  evaluated during this campaign.
- Each submission gets one untimed warmup generation of case 02, followed by
  exactly one timed generation each of cases 02, 03, and 04. Do not warm up
  cases 03 or 04. Model loading, compilation, and warmup are excluded from the
  reported inference latency but consume the GPU allocation.
- Raw reward is the mean across the three visible cases of
  `baseline_seconds / candidate_seconds * (1 - LPIPS)`.
- The entire round scores zero unless mean LPIPS across all three cases is at
  most 0.25 and worst-case LPIPS is at most 0.30. Both limits are inclusive.
- Use exactly one H100 per round. All 25 rounds have a 28-minute limit. The
  maximum combined reservation is 11.67 H100-hours, inside the 12 GPU-hour
  campaign budget.
  Failed or timed-out jobs consume their round; do not retry them under the
  same round number.
- Methods are unrestricted within the output contract. Actively explore both:
  - lossy methods: cross-step/diffusion/FirstBlock cache, adaptive thresholds,
    step skipping, approximate computation, and schedule changes;
  - exact or systems methods: operator/kernel fusion, SDPA/FlashAttention,
    Triton, `torch.compile`, CUDA graphs, precision/layout changes, memory
    residency/offload, and text/VAE/runtime optimization.
- Internet research is allowed. Record URLs and the concrete mechanism used in
  `journal.md`; prefer primary papers, official documentation, and upstream
  source.
- Never alter weights, prompts, seeds, resolution, frame count, requested step
  count, or output schema. Never return a cached completed output or add
  prompt-, seed-, or case-specific completed-output lookup.
- Work only in `workspace/`. Keep it self-contained. Before starting a round,
  commit the proposed source state and keep the worktree clean.
- This is an independent model comparison. Do not inspect or copy source,
  journals, state, or results from sibling Codex/SANA/H3 campaign directories.
- For every round: write a hypothesis, call `supervisor.py start`, run its
  emitted command, call `supervisor.py finish`, then record speed, all three
  LPIPS values, mean/worst LPIPS, reward, and conclusion in `journal.md`.
- Keep improvements and revert regressions before the next proposal. Preserve
  every trial in Git history and every measurement under `runs/`.
