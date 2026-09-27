# Continuous Claude Opus MiniMax-H3 25-round assignment

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

The cluster login node is control-plane only: source edits, short inspections,
Slurm queries, and job submission are allowed there. Never run model loading,
builds, benchmarks, bulk hashing, compression, or parallel CPU work on the
login node. All GPU execution must go through `run_round.sh`, which requests
exactly one H100. For a round longer than the Bash tool timeout, launch the
script in the background, poll it without busy-waiting, capture the `srun` job
ID, and finish the ledger only after Slurm reports a terminal state.

Do not inspect any sibling campaign directory. Your only mutable source is
`workspace/`; the campaign harness and reference are authoritative.
