# Continuous MiniMax-H3 25-round assignment

Read `POLICY.md` completely, then independently execute all 25 rounds without
waiting for user confirmation between rounds. Begin from the committed native
PyTorch dense/no-cache baseline in `workspace/`.

Use web research whenever useful. Do not limit the search to conservative or
lossless changes: the LPIPS gates intentionally permit approximate diffusion
methods. Prioritize cross-step caching and adaptive diffusion reuse as well as
kernel/runtime improvements. Measure every proposal on all three visible cases;
a single-case result is not a valid round.

The supervisor is the authoritative round ledger. A failed experiment counts.
Maintain `journal.md`, keep the best reproducible source state, and finish with
`FINAL.md` identifying the best commit, per-case latency and LPIPS, mean/worst
LPIPS, reward, techniques attempted, actual Slurm GPU time, and absolute
artifact paths. Never access the held-out prompts or outputs.

