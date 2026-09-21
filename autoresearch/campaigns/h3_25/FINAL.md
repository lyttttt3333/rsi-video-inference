# MiniMax-H3 25-round campaign — final report

## Outcome

- Completed exactly 25 sequential public-only rounds on one H100 per round.
- Best measured round: **Round 19**.
- Best measured source commit: `e818e1f5a899fd59c2862863d41f9bf57cf8a7af`.
- Final workspace commit: `bc8b1c31737f75a81d55396cefc2daa7f6d2c553`.
  Its source tree is identical to the Round 19 commit; it is the explicit
  post-campaign restoration commit.
- Reward: **1.3197285768**.
- Quality gate: pass; mean LPIPS **0.0893242409** <= 0.25 and worst LPIPS
  **0.1312085212** <= 0.30.
- Mean timed latency: **127.0463 s**, versus dense/no-cache baseline mean
  **183.5817 s**.

## Best public results

| Public case | Baseline (s) | Candidate (s) | Speedup | LPIPS | Case reward |
|---|---:|---:|---:|---:|---:|
| minimax_h3-02 | 185.2635 | 117.3263 | 1.57905x | 0.098414 | 1.423645 |
| minimax_h3-03 | 183.0606 | 131.8006 | 1.38892x | 0.131209 | 1.206682 |
| minimax_h3-04 | 182.4210 | 132.0120 | 1.38185x | 0.038350 | 1.328859 |

The selected configuration keeps the full requested 49 model evaluations and
uses:

- FirstBlockCache threshold `0.032`;
- standalone native-PyTorch Sol sparse attention with Triton backend,
  `tau=0.0`, first 20 denoising steps dense, and one protected early layer;
- the baseline persistent fused-QKV checkpoint, exact AdaLN schedule table,
  BF16 model path, and TF32-high matmul policy;
- both required allocator flushes around offloaded component residency.

## Techniques explored

- Cross-step FirstBlockCache across thresholds 0.02, 0.03, 0.032, 0.033,
  0.035, 0.04, 0.05, 0.08, 0.12, and 0.20. Aggressive thresholds were much
  faster but failed or approached the global quality limits.
- Approximate schedule reduction by removing one denoising grid point. It
  passed the gate but scored below the retained full schedule.
- Standalone Triton Sol sparse attention: tau 1.0, 0.0, and -0.5; dense
  prefixes 10, 15, 18, 20, 22, and 25; protected dense layers 0, 1, 2, and 5.
- Combined FBC + sparse-attention policies. The best was threshold 0.032,
  tau 0.0, prefix 20, and one protected layer.
- Parallel checkpoint shard reads. The round timed out and was rejected.
- Per-request CUDA allocator flush removal. Removing both caused VAE-decode
  OOM; removing only the front flush was memory-safe but did not improve score.
- Existing exact kernel/runtime foundation was retained and ablated around:
  fused QKV, exact AdaLN precompute, BF16, TF32-high, SDPA/Flash-compatible
  attention paths, CPU component offload, and the released SM90 Triton sparse
  backend. No SGLang runtime was introduced.

Primary mechanism references and round-by-round hypotheses, URLs, job IDs,
measurements, failures, and decisions are recorded in `journal.md`.

## GPU accounting

- Actual summed Slurm elapsed time: **26,370 seconds = 7.325 H100-hours**.
- Supervisor reservation ledger: **42,420 seconds = 11.783 H100-hours**.
- Campaign cap: **12 H100-hours**.
- Round 1 reserved 35 minutes; Rounds 2–25 reserved 28 minutes each.
- Failed and timed-out rounds were counted once and never retried.

## Artifacts

- Campaign source: this directory.
- Restored best source: `workspace/candidate.py`.
- Complete experiment record: `journal.md`.
- Generated run outputs, ledgers and videos are intentionally excluded from Git.

Held-out cases 05–08 were never mounted, read, or evaluated during this
campaign.
