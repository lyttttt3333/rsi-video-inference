# Autoresearch supervisor policy

- Exactly 40 rounds in two independent sequential campaigns: first SANA rounds
  1-20, then MiniMax-H3 rounds 1-20. H3 cannot start before SANA round 20 is
  terminal and recorded.
- Only public cases 01-04 may be used. The case rotates across a model's turns.
- Forbidden: either task's `tests/`, reference-spec records/cases 05-08,
  teacher-videos 05-08, any `references/heldout`, and any held-out enable token.
- Do not run or build a held-out verifier. Do not infer held-out prompts from
  owner artifacts. Stop and report if a command would traverse a forbidden path.
- One H100 maximum. SANA receives 3 GPU-hours total (9 minutes per round);
  H3 receives 9 GPU-hours total (27 minutes per round). No retries: a failed
  round is still one of that model's 20. Maximum reserved time is exactly the
  shared 12 GPU-hour cap; jobs ending early do not authorize extra rounds.
- Each round: write hypothesis, edit only that model's submission workspace,
  run `supervisor.py start --model MODEL --round N`, call the emitted command,
  then `supervisor.py finish --model MODEL --round N --job-id JOB`.
- Keep a change only if public score improves or it is a documented enabling
  experiment. Revert regressions in the next source-only action, not by retrying.
- Web research is allowed. Prefer official framework docs, source repositories,
  and primary papers. Log URLs and the concrete mechanism in `journal.md`.
- Scope: PyTorch/CUDA kernels, attention/SDPA/FlashAttention, offload/transfer,
  compile/graphs, fusion, VAE/text path, diffusion caching, and step schedules.
  Changing weights, prompts, seeds, shapes, output contracts, or returning a
  cached completed output is forbidden.
- Public evaluation uses one warmup and one timed generation for exploration.
  The final public winner must later receive a four-case verification; this is
  outside the 20 inner rounds and does not unlock held-out.
