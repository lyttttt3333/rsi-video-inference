# Continuous 50-round assignment

Read `POLICY.md` completely, then independently execute all 50 rounds without
waiting for user confirmation between rounds. Begin from the committed
GPU-resident baseline in `workspace/`.

Use web research whenever it helps. Do not limit the search to conservative or
lossless changes: the LPIPS gates intentionally permit approximate diffusion
methods. Combine cross-step caching or step reduction with kernel/runtime work
when promising. Measure every proposal on all eight visible cases; a single
case experiment is not a valid round.

The supervisor is the authoritative round ledger. A failed experiment counts.
Maintain `journal.md`, keep the best reproducible source state, and finish with
a concise `FINAL.md` identifying the best commit, all-eight metrics, techniques
attempted, and absolute artifact paths.

