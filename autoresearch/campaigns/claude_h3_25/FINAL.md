# FINAL — MiniMax-H3, 25-round campaign (claude-opus)

## Best result

| | |
|---|---|
| **Best round** | **24** |
| **Best commit** | `6262ce3d3dd950139c83d45d9d1315668766cb70` (`workspace/`, message `round 24: tail_steps 1 -> 0`) |
| **Submitted state** | `9ca2ac7` — source byte-identical to `6262ce3`; the only later commit reverts round 25 and updates `summary.md` |
| **Reward** | **3.1222522774638084** |
| **Mean LPIPS** | 0.2113287419841815 (limit 0.25) |
| **Worst LPIPS** | 0.2661462160608461 (limit 0.30) |
| **Quality gate** | passed |

### Per case

| case | candidate s | baseline s | speedup | LPIPS | case reward |
|---|---|---|---|---|---|
| `minimax_h3-02` | 41.815449215937406 | 185.263501688838 | 4.430503681357733× | 0.23741404315637005 | 3.3786398891474114 |
| `minimax_h3-03` | 49.06033482309431 | 183.06059535592794 | 3.7313360378811624× | 0.2661462160608461 | 2.738255070547621 |
| `minimax_h3-04` | 48.810874921735376 | 182.42104215733707 | 3.7373032638696952× | 0.13042596673532839 | 3.249861872696392 |

Metric: LPIPS-alex, package 0.1.4, mean over frames, mean over repeats.
Peak device memory 72.16 GiB. Only the three visible cases were ever read,
generated or evaluated; `minimax_h3-05`..`-08` were never touched.

## GPU time

25 rounds started, 25 finished, **28,188 s = 7.83 H100-hours** actually consumed
(`sacct -X`, one H100 per job, agreeing exactly with the supervisor ledger's
`elapsed_seconds`). Reservation 42,000 s = 11.67 h; budget 12 h.

Job IDs in round order: 19007217, 19008383, 19009550, 19010393, 19011611,
19012563, 19013308, 19014104, 19015224, 19016699, 19018271, 19018972, 19019722,
19020995, 19022197, 19023163, 19024027, 19025208, 19026129, 19026783, 19027900,
19029171, 19029734, 19030445, 19031331.

## Ledger

| r | reward | outcome |
|---|---|---|
| 1 | 1.1485 | host-resident weights, lm_head dropped, phase timing |
| 2 | 0.0 | timed out |
| 3 | 0.0 | gate |
| 4 | 1.4763 | FirstBlockCache introduced |
| 5 | 0.0 | gate — threshold 0.055 too loose |
| 6 | 1.4557 | regression, reverted |
| 7 | 0.0 | OOM |
| 8 | 1.3631 | regression, reverted |
| 9 | 0.0 | timed out |
| 10 | 0.0 | gate |
| 11 | 1.4995 | conditioner truncated to 51 decoder layers |
| 12 | 1.8768 | late-region reuse structure |
| 13 | 0.0 | gate |
| 14 | 2.1439 | fused QKV |
| 15 | 2.3837 | AdaLN precomputation |
| 16 | 2.6554 | attention backend calibration (cuDNN/value) |
| 17 | 2.8100 | threaded pinned-bounce-buffer staging |
| 18 | 2.8969 | inductor-compiled VAE decoder |
| 19 | 0.0 | gate — deep-split reuse (raw 2.139) |
| 20 | 2.9076 | late-region knobs |
| 21 | 0.0 | gate — depth-gated kernels (raw 2.820) |
| 22 | 0.0 | FAILED — pipelined-staging plan list held 24 GiB alive, OOM |
| 23 | 3.0452 | `tail_steps` 3 → 1 |
| 24 | **3.1223** | `tail_steps` 1 → 0 — **best** |
| 25 | 0.0 | gate — threshold 0.036 → 0.044 (raw 3.2169), reverted |

## Techniques attempted

**Exact / systems.** Host-master weight residency with no device→host copies;
dropping the unused `lm_head`; threaded page-locked bounce-buffer H2D staging
(4 workers × 96 MiB × per-worker streams, 19–21 GiB/s); pipelined
staging-under-compute (round 22, measured negative: ~8 GiB/s concurrent vs 20
idle, break-even 7.9 GiB/s); fused QKV projections; AdaLN precomputation into
per-step tables (24.23 → 0.883 GiB, disk-cached); building only the 51 Qwen3-VL
decoder layers that can reach `hidden_states[50]` (60.68 → 48.88 GiB);
attention-backend and layout calibration at run time (cuDNN/value at 17.87 ms vs
flash 29.72 ms, 1.72×, identical float32 slice error); `torch.compile` on the
VAE decoder; step- and depth-gated application of the calibrated backend.

**Lossy / diffusion.** FirstBlockCache with a relative-L1 drift gate on block
0's residual, plus warmup, tail, max-consecutive-reuse, late-fraction,
late-max-run and predictor guards; threshold sweeps; late-region reuse runs;
deep-split (blocks 25–49 only) reuse; per-case adaptive operating points
(rejected on evidence, see below). Round 24's configuration computes 12 of 49
forwards on case 02 and 15 on cases 03/04.

## What the campaign established

1. **Damage is decided by *which* steps are approximated**, not how many and not
   in what pattern. Reusing step 4 costs case 02 +0.154 LPIPS; reusing steps
   36–45 costs +0.0007 in total; steps 46/47/48 cost 0.009/0.009/0.017 each.
2. **Position in the pipeline decides what a kernel difference costs.** A
   2.9e-4 per-call divergence compounds to 0.21 LPIPS across 50 blocks × 27
   steps, while the same class of change at the end of the pipeline (compiled
   VAE decoder) moves LPIPS by 1.1e-7.
3. **Depth makes approximation worse, not better** — two independent mechanisms
   (rounds 19 and 21): reusing a deep segment's output costs 4× what reusing the
   whole stack costs, and injecting a kernel difference into the deep half
   carries 71% of the cost of injecting it everywhere.
4. **No in-flight signal orders the cases by damage** — five independent
   failures: the raw drift trace, its shape, tail-residual magnitude, per-case
   `hold` prediction error, and finally a matched pair of drift values agreeing
   to 2% where the same two reused steps cost case 03 +0.0667 and case 04
   +0.0215. The operating point must therefore be global and set by the case
   that suffers most.
5. **An untested default is an unmeasured one.** `tail_steps = 3` was a round-4
   guess carried through nineteen rounds while neighbouring knobs were tuned to
   exhaustion; removing it (rounds 23–24) was worth +14% of reward.

## Artifacts

- Submitted source: `workspace/candidate.py`.
- Method summary: `workspace/summary.md`.
- All 25 hypotheses and results: `journal.md`.
- Generated measurements, ledgers and videos are intentionally excluded from Git.
