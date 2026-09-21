# Continuous Claude Opus SANA 50-round assignment

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

The cluster login node is control-plane only: source edits, short inspections,
Slurm queries, and job submission are allowed there. Never run model loading,
builds, benchmarks, bulk hashing, compression, or parallel CPU work on the
login node. All GPU execution must go through `run_round.sh`, which requests
exactly one H100. For a round longer than the Bash tool timeout, launch the
script in the background, poll it without busy-waiting, capture the `srun` job
ID, and finish the ledger only after Slurm reports a terminal state.

Do not inspect any sibling campaign directory. Your only mutable source is
`workspace/`; the campaign harness and reference are authoritative.
