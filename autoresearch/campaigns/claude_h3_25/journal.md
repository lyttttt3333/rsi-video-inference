# Claude Opus MiniMax-H3 25-round journal

Independent baseline: native PyTorch dense/no-cache. Visible cases 02–04;
case 02 is warmed once per submission. Gate: mean LPIPS <= 0.25 and worst LPIPS
<= 0.30. Heldout cases 05–08 are forbidden.

## Starting-point analysis (no GPU time)

Read from the committed baseline and the harness, before round 1:

* Geometry of all three visible cases is identical — 544x960, 124 frames,
  50 requested steps. The VAE compresses 16x spatially and packs (1,2,2)
  patches, so the packed sequence is 37 latent frames x 17 x 30 = 18 870 video
  rows, 2 x 207 = 414 audio rows and the prompt's text rows (~10^2). The
  schedule's terminal sigma has no model evaluation, so 50 requested steps drive
  49 transformer forwards.
* The transformer is 50 blocks, hidden 5376, 56 x 128 attention heads
  (inner 7168), SwiGLU ffn 14336. ~19.3 B parameters outside the AdaLN
  projections and ~13.0 B inside them; `precompute_adaln` replaces the AdaLN
  projections with a per-step table on the first request and frees the rest.
* Reference latency: 185.26 / 183.06 / 182.42 s for cases 02 / 03 / 04.
* Rough roofline for one forward: ~5.3e14 FLOP of attention and ~7.4e14 FLOP of
  GEMM, i.e. ~2.1 s of ideal H100 bf16 work against a ~3.7 s/step budget.
* The baseline moves whole models across PCIe inside every timed request:
  `text_encoder.to('cuda')`, `.to('cpu')`, `transformer.to('cuda')`,
  `.to('cpu')`. That is ~200 GiB per request, half of it device->host copies of
  read-only weights.

Levers the policy admits, ranked by expected value: cross-step caching
(FirstBlockCache is already wired but disabled), fp8 GEMM, attention sparsity
(`sol_attn` is wired but disabled), transfer/residency, VAE decode.

## Round 1 — exact weight residency + instrumentation

Hypothesis: the four per-request whole-model device transfers can be cut to two
host->device stagings with no numerical change, because the weights are
read-only and the host already holds them. Bundled with phase instrumentation
(`H3_PHASES`) and a warmup-only kernel table for one transformer forward, so
that rounds 2-25 target measured hot spots rather than the roofline estimate
above.

Change: `_HostResident` keeps the authoritative copy on the host and stages a
device copy into `Parameter.data`; unloading restores the host tensor instead of
copying back. `lm_head` (never evaluated — MiniMax-H3 reads
`hidden_states[50]` off `text_encoder.model`) is replaced by `Identity`.
Commit `6e6a7f4`.

Result: **reward 1.1485**, job 19007217 COMPLETED in 890 s.

| case | seconds | baseline | speedup | LPIPS |
|------|---------|----------|---------|-------|
| minimax_h3-02 | 153.99 | 185.26 | 1.203 | 0.000 |
| minimax_h3-03 | 176.39 | 183.06 | 1.038 | 0.000 |
| minimax_h3-04 | 151.44 | 182.42 | 1.205 | 0.000 |

mean LPIPS 0.000, worst 0.000, gate passed, raw reward 1.1485.

Conclusions:

1. **The pipeline is bit-exact reproducible.** LPIPS is exactly 0.0 on all three
   cases, so every LPIPS from here on is purely my own approximation error. No
   run-to-run noise floor to subtract.
2. Phase split of a timed request (case 02): text staging 6.88 s, text encode
   0.06 s, transformer staging 4.30 s, prepare 0.06 s, **denoise 138.3 s**,
   video decode 4.24 s, audio decode 0.05 s. The denoise loop is 90% of the
   request; the conditioner's *compute* is 0.06 s against 6.9 s of staging.
3. Case 03 lost 24 s to slow staging (3.53 / 2.07 GiB/s against 8.8 / 9.9 GiB/s
   on the other two), which is the whole of its worse speedup. Pageable staging
   bandwidth is not reproducible — hence page-locking in round 2.
4. Kernel table of one transformer forward (2.808 s of device time, 49 of them
   per request):

   | kernel | ms | share |
   |--------|----|-------|
   | `pytorch_flash::flash_fwd_kernel` (50) | 1671 | 59.5% |
   | `nvjet_tst_192x208_...` i.e. `aten::mm` (200) | 916 | 32.6% |
   | `_qknorm_partial_rope_kernel` (100) | 103 | 3.7% |
   | `aten::copy_` — the `.contiguous()` in the rope fusion (100) | 37 | 1.3% |
   | `_swiglu_kernel` (50) | 28 | 1.0% |
   | `_residual_gate_rmsnorm_modulate_kernel` (50) | 16 | 0.6% |
   | `_rmsnorm_modulate_kernel` (50) | 9 | 0.3% |

   The 200 GEMMs do 7.5e14 FLOP in 916 ms = **820 TFLOPS**, i.e. ~83% of the
   H100's bf16 peak: there is nothing left in bf16 GEMM, only fp8. Attention
   does 5.4e14 FLOP in 1671 ms = **326 TFLOPS**, a third of peak, which is far
   below what FlashAttention-class kernels reach on this shape. Attention is
   where the exact work is.

## Round 2 — measured attention backend + page-locked staging

Hypothesis: the hard-coded `SDPBackend.FLASH_ATTENTION` and the as-produced
query/key/value layout are not the fastest pair available. Benchmark
{cuDNN, flash, efficient} x {as-produced, contiguous value, all contiguous} on
the real packed-sequence shape and stride pattern during the untimed warmup and
pin the winner (`model/attention_backend.py`), with a fallback inside the hot
path so a backend that rejects the token refiner's 203-row sequence cannot fail
the round. Page-lock the host masters at build time (AdaLN projections excluded:
they are staged exactly once). Commit `2cdadb7`.

Result: **reward 0.0 — TIMEOUT.** Job 19008383 FAILED after 1642 s; the harness
killed `measure.py` at its 1560 s limit during the warmup generation. No
attention measurement was obtained. The round is spent.

Cause, from the build log:

```
H3_PIN text_gib=60.68 transformer_gib=37.49 seconds=577.3
Build completed: 1550.740s excluded
```

Two separate costs, both outside the scored latency but inside the 26-minute
measurement window:

1. **Page-locking 98 GiB cost 577 s.** Against a per-request staging bill of
   ~11 s that can never pay back inside four generations, even if it had made
   staging free. Reverted; `H3_PIN_BUDGET_GIB` keeps the code path reachable
   but defaults to 0.
2. **Weight loading alone took 973 s** (1550.7 − 577.3), against **155 s for
   byte-identical code in round 1**. Nothing in my diff touches loading, so this
   is contention: the sibling campaigns read the same 62 GiB transformer and
   63 GiB conditioner off the same Lustre filesystem, and round 3 was scheduled
   onto the same node as one of them. Build time is an externally driven
   6x-variable cost that I have to leave headroom for — it is not a constant
   I can plan against.

Consequences carried forward: never add build-time work that is not paid back
within four generations; treat ~1000 s of build as a plausible worst case, so
the four generations have to fit in ~550 s; and gate optional warmup
diagnostics on the remaining wall-clock budget (done — `H3_PROFILE skipped`
prints instead).

## Round 3 — round 2's hypothesis, without the page-locking

Identical to round 2 except that host page-locking is off and the warmup
profile is skipped when under 700 s of the measurement window remains.
Commit `0ab361b`.

Result: **reward 0.0 — quality gate FAILED.** Job 19009550 COMPLETED in 704 s.
Build was 125.9 s this time (vs 973 s in round 2 for the same loading code),
confirming that round 2's build blow-up was external contention, not my diff.

| case | seconds | speedup | LPIPS |
|------|---------|---------|-------|
| minimax_h3-02 | 128.65 | 1.4400 | **0.5079** |
| minimax_h3-03 | 127.80 | 1.4324 | 0.1041 |
| minimax_h3-04 | 128.31 | 1.4217 | 0.0074 |

mean LPIPS 0.2065 (under the 0.25 mean gate), worst 0.5079 (**over** the 0.30
worst gate) → the whole round scores zero. Had it passed it would have been
worth ~1.13; had all three cases behaved like case 04 it would have been 1.42.

The attention search worked exactly as designed:

```
H3_ATTENTION_SELECT seq=19487 heads=56 head_dim=128 chosen=cudnn/none
  cudnn/none=17.63ms cudnn/value=18.41ms cudnn/all=18.55ms
  flash/value=30.73ms flash/none=30.86ms flash/all=31.10ms
  efficient/value=59.87ms efficient/none=59.97ms efficient/all=60.38ms
```

cuDNN is **1.75x** FlashAttention on this shape; layout is worth under 2%, so
the `.contiguous()` question is settled and closed. One forward fell from
2.808 s to 2.229 s and the kernel table re-balanced to attention 1.018 s
(45.7%) / GEMM 980.7 ms (44.0%) / qknorm_rope 110 ms (4.9%).

Conclusions, and they reshape the rest of the campaign:

1. **Speed was never the problem; case 02 is.** The identical numerical change
   produced LPIPS 0.508 / 0.104 / 0.007 across the three cases — a **70x
   spread**. Case 02 is the only case with a real prompt (203 text tokens
   against 19 and 23), and it is the fragile one. Every future proposal has to
   be judged on case 02's LPIPS; the mean is nearly uninformative, and the
   worst-case gate is what actually binds.
2. A 1.4x kernel that is *exactly* as correct as the reference to within
   bf16 rounding still moved case 02 by 0.51 LPIPS. Either cuDNN is genuinely
   coarser, or the sampler is chaotic enough that reordering-sized differences
   compound over 49 steps. Round 4 carries a warmup diagnostic
   (`H3_ATTENTION_ERROR`) that compares each backend to a float32 reference to
   separate the two, because the responses differ completely: a coarser kernel
   can be fixed, a chaotic sampler means *no* exact-but-different kernel is ever
   safe and the whole systems half of the lever list is dead.
3. Reverted to flash. The cuDNN plumbing stays, gated behind
   `H3_ATTENTION_BACKENDS` (default `flash`), so a later round can re-enable it
   for a subset of steps once the mechanism is known.
4. Per-request phases are now: text staging 9.17 s, text encode 0.06 s,
   transformer staging 5.06 s, prepare 0.05 s, **denoise 110.03 s (86%)**,
   video decode 4.16 s, audio decode 0.05 s, peak 72.30 GiB.

## Round 4 — FirstBlockCache with drift guards, on exact attention

Hypothesis: the denoise loop is 86% of the request and 49 forwards of it are
near-duplicates of one another, so skipping the block stack on steps where the
first block's residual barely changes buys far more than any kernel, and does it
where the LPIPS budget was designed to be spent. The relative-L1 threshold alone
is unguarded — one bad reuse early in the schedule compounds for 48 steps — so
this round bounds it three ways: the first 3 and last 2 steps of the schedule
are always dense, and no more than 2 reuses may follow one another.

Configuration: threshold 0.032 (the vendor's default), warmup 3, tail 2,
max_run 2, attention pinned to flash so that all of this round's LPIPS is
attributable to the cache and nothing else. Every step logs its relative-L1 and
its decision (`H3_FBC_TRACE step:rel_l1:reason`), which turns one round into the
entire threshold-vs-reuse curve: the trace says exactly what any other threshold
would have done, so rounds 5+ can pick one without spending a round to find it.
The warmup also emits `H3_ATTENTION_ERROR`, settling round 3's open question.
Commit `80511e0`.

Result: **reward 1.4763 — new best, +28.5% over round 1.** Job 19010393
COMPLETED in 1501 s (build was 972 s again: same node as a sibling campaign).

| case | seconds | speedup | LPIPS | reuse |
|------|---------|---------|-------|-------|
| minimax_h3-02 | 108.61 | 1.7057 | 0.0944 | 17/49 (34.7%) |
| minimax_h3-03 | 120.66 | 1.5171 | 0.0827 | 12/49 (24.5%) |
| minimax_h3-04 | 120.34 | 1.5158 | 0.0153 | 12/49 (24.5%) |

mean LPIPS 0.0642 (gate 0.25), worst 0.0944 (gate 0.30) — **passed with three
times the headroom I spent.** Denoise fell from 138.3 s to 92.0 s on case 02;
138.3 x (1 - 0.347) = 90.3, so a reused step costs essentially nothing beyond
the one block that always runs.

The per-step trace is the most valuable artifact of the campaign so far. Case 02
(the leading `inf` is the cold step):

```
0:inf:cold 1:.0585:warmup 2:.0456:warmup 3:.0405 4:.0360 5:.0379 6:.0320:reuse
7:.0627 8:.0290:reuse 9:.0545 10:.0286:reuse ... 20:.0195:reuse 21:.0401
... 38:.0308:reuse 39:.0644 40:.0367 41:.0405 ... 46:.0818 47:.1083:tail
48:.1939:tail
```

Five things fall out of it:

1. **The relative-L1 curve is U-shaped.** It starts at 0.059, sags to 0.0195
   around step 20, and climbs to 0.19 at the last step. The schedule's two ends
   are where the sequence actually changes; the middle third is nearly static.
   A single scalar threshold therefore already behaves like a schedule-aware
   rule, and the `warmup`/`tail` guards mostly duplicate what the threshold
   would have done anyway (steps 47-48 read 0.108 and 0.194, well above 0.032).
2. **Reuse doubles the next step's reading.** `previous` is only refreshed on a
   computed step, so after one reuse the metric measures two steps of drift:
   0.0320 reuse -> 0.0627, 0.0290 -> 0.0545, 0.0212 -> 0.0409. The pattern is an
   exact compute/reuse alternation and **`max_consecutive_reuse=2` never fired
   once** — it is inert at this threshold. The mid-schedule reuse rate is
   therefore hard-capped at 50% for any threshold below 2x the single-step
   value.
3. **Which means the threshold has to clear 2x the mid-schedule reading to buy
   anything more.** Case 02 sits at 0.020-0.022 mid-schedule, cases 03/04 at
   0.024-0.030 (they are also the cases with 19 and 23 text tokens against case
   02's 203 — less conditioning, faster-moving latents, fewer reuses). Crossing
   0.055 should turn the middle into a 2-reuse-per-3-steps pattern on all three
   cases, at which point `max_run` starts binding and becomes the real control.
4. **The LPIPS budget is barely touched.** Worst case 0.094 against a 0.30 gate.
   Reward is `speedup x (1 - LPIPS)`, and at a fixed ~16.6 s of staging + decode
   overhead, pushing reuse from 35% to 60% takes case 02 to ~72 s (speedup 2.6);
   that pays for an LPIPS of up to 0.25 and still beats today's 1.54.
5. **Round 3's cuDNN result is now a contradiction, not an explanation.** The
   warmup diagnostic says cuDNN and flash are *equally* accurate against a
   float32 reference: `cudnn=mean2.236e-03/max2.971e-02` vs
   `flash=mean2.234e-03/max2.955e-02`. So cuDNN is not a coarser kernel. But if
   the sampler were chaotic enough to turn a bf16-rounding difference into LPIPS
   0.508, then FirstBlockCache — which deletes 49 of 50 blocks on 17 steps —
   could not possibly have come in at 0.094. Both cannot be true. The diagnostic
   was run at sequence length 4096, and the model runs at 19487: I tested a
   shape the model never sees. Round 5 re-runs it at the real length.

## Round 5 — cross the 50% reuse cap

Hypothesis: round 4 spent a third of the quality budget and bought 1.48; the
trace says the remaining two thirds are reachable by one number. A threshold
above twice the mid-schedule relative-L1 reading (0.020-0.030 across the three
cases) lets the flat middle of the schedule hold a two-step reuse run instead of
alternating, which is the difference between a 50% and a 66% reuse rate exactly
where the sequence is not moving. Threshold 0.055; run cap 3 (it now actually
binds, so it is the safety net rather than dead code); warmup 4 and tail 3
because the U-shaped curve says both ends of the schedule are where the
sequence is still changing fast. Projection: ~50% reuse, ~87 s on case 02,
speedup ~2.1, and the round still clears the gate at any LPIPS below 0.30.

Bundled: the round-4 attention diagnostic re-run at the real sequence length.
A float32 reference over all 19487 rows would need a 1.5 TB attention matrix, so
it is taken over 192-row slices at the head, middle and tail of the sequence
against the full key/value. 19487 is not a multiple of any kernel tile, and a
backend that mishandles the ragged last tile would show up in the tail slice
alone — which a 4096-row square test could never have seen.
Commit `621ac5b`.

Result: **reward 0.0 — quality gate FAILED.** Job 19011611 COMPLETED in 1362 s
(build 952 s).

| case | seconds | speedup | LPIPS | reuse |
|------|---------|---------|-------|-------|
| minimax_h3-02 | 80.22 | 2.3094 | 0.2713 | 26/49 (53.1%) |
| minimax_h3-03 | 87.83 | 2.0843 | **0.3852** | 23/49 (46.9%) |
| minimax_h3-04 | 88.93 | 2.0512 | 0.0208 | 23/49 (46.9%) |

mean LPIPS 0.2257 (inside the 0.25 mean gate), worst 0.3852 (over the 0.30 worst
gate). Speed did exactly what the trace predicted — 108.6 s -> 80.2 s on case 02,
2.31x the reference — and had the gate held the round would have scored 1.658.

**The important result of this round is not the failure, it is what cases 03 and
04 did.** Their traces are the same to within 2% at every one of the 49 steps,
they took the same 23 reuses at the same steps, and their LPIPS came out
**0.3852 and 0.0208** — a factor of 18.5. The first block's relative-L1 is
therefore a fine predictor of *how much the latent sequence is moving* and no
predictor at all of *how much damage skipping the stack does to the decoded
video*. Confirming it from the other direction: case 02 went 0.0944 -> 0.2713
for +9 reuses while case 04 went 0.0153 -> 0.0208 for +11.

Per-case reward is also not monotone in the same direction, which is what makes
this hard. Fitting the two measured points per case:

| case | LPIPS per extra reuse | seconds per extra reuse | more reuse is |
|------|----------------------|--------------------------|---------------|
| minimax_h3-02 | +0.0197 | -3.15 | reward-positive (1.545 -> 1.683) |
| minimax_h3-03 | +0.0275 | -2.99 | reward-**negative** (1.392 -> 1.282) |
| minimax_h3-04 | +0.0005 | -2.86 | strongly positive (1.493 -> 2.009) |

Case 03 pays more quality per reuse than any other case and buys the least, and
it is also the case that trips the worst-case gate. Tuning one global threshold
just slides along a frontier where one case is already past its own optimum: the
gain from 0.032 to a safe 0.038 is worth about +0.03 of reward, and the cliff at
0.30 costs everything. **Sliding the threshold is close to exhausted; the
frontier itself has to move**, i.e. a reused step has to be made more accurate
rather than rarer.

Secondary result: the attention diagnostic re-run at the real sequence length
found nothing. Head, middle and tail slices of a 19487-row sequence all give
`cudnn=2.27e-03`, `flash=2.27e-03`, `efficient=2.27e-03` mean relative error
against float32 — no ragged-tile bug, no coarser kernel. But it uses Gaussian
inputs, whose softmax is nearly uniform, so it is weak evidence about real
activations. The decisive argument is the one that comes free from rounds 4-5:
if the sampler were chaotic enough to turn bf16-rounding differences into LPIPS
0.508, then deleting 49 of 50 blocks on half the steps could not have come in at
0.02-0.09. It did. So the sampler is not chaotic, and cuDNN must be
systematically different on real activations rather than merely differently
rounded. cuDNN stays off and I am not spending more rounds on it.

## Round 6 — make a reused step more accurate, and measure which predictor works

Hypothesis: FirstBlockCache currently reuses the stale tail residual verbatim.
The first block is evaluated on every step, so its residual at the *current*
step is known even when the tail is being skipped — the cache is throwing away
the one piece of current information it has. Rescaling the cached tail residual
by the ratio of first-block residual magnitudes should track the part of the
drift that is pure magnitude, at the cost of two reductions per step.

Because the right predictor is an empirical question and each round costs 25
minutes, this round also *measures* three predictors in flight without acting on
them. On every computed step the true tail residual is available, and so are the
cached quantities the predictors would have used, so each predictor's relative
L1 error against the truth can be scored for free:

* `hold` — reuse the last computed tail residual verbatim (today's behaviour);
* `scale` — rescale it by the first-block residual magnitude ratio (applied);
* `linear` — extrapolate linearly from the last two computed tail residuals.

That turns one round into a direct ranking of the three, with no dependence on
LPIPS to tell them apart. Threshold moves only from 0.032 to 0.038, which the
traces say keeps case 02 below the run-of-two boundary (its drift doubles to
0.041 after one reuse) and so stays well clear of the cliff.
Commit `6f6de1b`.

Result: **reward 1.4557, gate passed** — below round 4's 1.4763. Job 19012563
COMPLETED in 1462 s (build 949 s).

| case | seconds | speedup | LPIPS | reuse |
|------|---------|---------|-------|-------|
| minimax_h3-02 | 101.52 | 1.8249 | 0.2770 | 19/49 |
| minimax_h3-03 | 114.21 | 1.6028 | 0.0764 | 14/49 |
| minimax_h3-04 | 114.17 | 1.5979 | 0.0190 | 14/49 |

The predictor scores, averaged over every computed step (relative L1 against the
tail residual that step actually produced):

| case | hold | scale | linear |
|------|------|-------|--------|
| minimax_h3-02 | **0.2137** | 0.2157 | 0.2924 |
| minimax_h3-03 | **0.1781** | 0.1817 | 0.2317 |
| minimax_h3-04 | **0.1438** | 0.1463 | 0.1733 |

Three things, and the third is the one worth keeping:

1. **Both predictors lose to doing nothing.** `scale` is 1% worse on all three
   cases and `linear` is 20-37% worse. The tail residual moves ~15-21% between
   computed steps and that movement is neither a pure magnitude change nor
   smooth enough in the step index to extrapolate through. The idea that the
   first block's current residual carries usable information about the rest of
   the stack is dead — as a *correction*; it remains a usable change detector.
2. **The applied `scale` was worse than its 1% score suggests.** Case 02 reached
   LPIPS 0.277 on 19 reuses; round 5 needed 26 reuses to reach 0.271 with
   `hold`. A 1% edge in per-step relative L1 turned into roughly a 7-reuse
   penalty, because `scale`'s error is *biased*: the first-block residual grows
   through the back half of the schedule, so the ratio sits above 1 on most
   reused steps and the correction inflates the tail residual the same way every
   time. Errors that share a sign accumulate; errors that do not, cancel.
   A per-step error metric that ignores sign cannot see this, which is exactly
   why the round measured a ranking and still had to be checked against LPIPS.
3. **`hold` error does not track damage either.** Case 02 has the *highest* hold
   error (0.214) and case 03 the middle (0.178), yet case 03 is the case whose
   LPIPS explodes. Together with round 5's 03-vs-04 result, three different
   in-flight signals now fail to predict which case a reuse will hurt. I am
   going to stop looking for one.

Reverted to round 4's `hold` at threshold 0.032. Best so far remains commit
`80511e0` at 1.4763.

## Round 7 — spend the clock instead of the quality budget

Hypothesis: three rounds of trying to buy speed with quality have produced one
usable configuration and a 0.30 cliff. Meanwhile 12 s of every ~100 s request is
PCIe staging, which is exactly lossless and has nothing to do with the gate.
`Tensor.to(device)` on pageable memory already bounces through a page-locked
buffer inside the driver, but serially and on one core: rounds 1-6 measured
6.6-9.9 GiB/s on a link that runs at about 25. Four worker threads, each with a
96 MiB page-locked buffer and its own CUDA stream, overlap one chunk's host
memcpy with another's transfer and give the memcpy more than one core. Round 2
established page-locking the models is unaffordable at 5.9 s/GiB; 384 MiB of
*reusable* buffers is ~2 s, paid once.

Bundled with it, and also exactly lossless: the conditioner is truncated to
`text_encoder_layer + 1 = 51` decoder layers. MiniMax-H3 reads
`hidden_states[50]`, so layers 51-63 of the 64-layer Qwen3-VL body cannot reach
any output. Keeping 51 leaves index 50 addressable whichever convention
transformers' output capture uses (the input of layer 50, or its output), and
removes ~12 GiB from both the build's read and every request's staging. The
chunk/view arithmetic the stager depends on was checked for byte-exactness on
small tensors before the round, and the warmup staging byte-compares a sample of
twelve staged tensors against their host masters (`H3_STAGE_VERIFY`) so that a
chunking bug reports itself instead of arriving as an inexplicable LPIPS.

Also added: per-phase build instrumentation (`H3_PHASES build`). Four of the six
rounds so far have spent 950-975 s in build against a 1560 s measurement window,
and I still do not know which model's load that is.
Commit `dd1c275`.

Result: **reward 0.0 — FAILED, CUDA out of memory.** Job 19013308 FAILED after
1031 s. All three cases came back `invalid`; the warmup generation died staging
the transformer.

The two lossless changes both did what they were supposed to:

* `H3_CONDITIONER_TRUNCATED kept=51 dropped=13`, and the conditioner's host
  master fell from 60.68 GiB to **48.88 GiB**.
* `H3_STAGER workers=4 buffer_mib=96 pin_seconds=0.18` — page-locking 384 MiB
  took 0.18 s, not the ~2 s I budgeted from round 2's 5.9 s/GiB.

And the instrumentation answered the build question outright:

```
H3_PHASES build total=940.655 tokenizer=0.514 text_encoder=928.760
                 transformer=0.288 vae=10.912 stager=0.181
```

**98.7% of the build is one call**: HF's `from_pretrained` for the 64-layer
Qwen3-VL conditioner. The transformer's 61.73 GiB costs 0.288 s because
`runtime.py` loads it through a memory-mapped safetensors file whose pages are
not faulted in until the first staging. The conditioner is the entire timeout
risk, and 13 of the 64 layers it reads cannot reach `hidden_states[50]`.

The failure itself was mine, and the interesting part is not the bug but its
blast radius:

```
H3_STAGER_ERROR RuntimeError: Inplace update to inference tensor outside
InferenceMode is not allowed.
```

`generate` runs under `torch.inference_mode()`, so the device tensors it
allocates are inference tensors — and inference mode is **thread-local**, so the
stager's worker threads were not in it and could not write to them. That much
was a one-line bug. What turned it into a dead round is that I let the worker's
exception propagate into `load`, where I caught it and ran the serial fallback.
The fallback staged the conditioner correctly (`H3_STAGE_VERIFY text_encoder
checked=12 mismatched=0`), but the exception's traceback formed a reference
cycle holding the *frame* that held those 48.88 GiB of device tensors. Python's
cyclic collector had not run by the time `unload()` was called, so `unload`
dropped `.data` references that were no longer the only ones, `empty_cache()`
freed nothing, and staging the 61.73 GiB transformer on top of a conditioner
that was supposed to be gone hit 78.98 GiB of 79.11 GiB.

Three lessons, in order of how much they cost:

1. **An error path that runs while 50 GiB is live is not a nicety.** A `try`
   around a fallback looks like defensive programming and was the thing that
   killed the round. `_Stager.copy` now drains every future with
   `Future.exception()`, which waits without raising, keeps the exception as a
   formatted string, and lets nothing escape.
2. **`torch.inference_mode()` is thread-local.** Any worker thread touching
   tensors that `generate` allocated has to enter it.
3. **A phase timer that does not report memory cannot see a leak.** The request
   line now carries `alloc_after_text_unstage_gib`, which would have said 48.88
   instead of ~0 and named the problem in one line.

## Round 8 — the same lossless changes, with the error path fixed

Identical intent to round 7 and no new quality risk: FirstBlockCache stays at
round 4's settings (`hold`, threshold 0.032, warmup 3, tail 2), so the three
LPIPS values should reproduce 0.094 / 0.083 / 0.015 and everything the round
measures is clock.

* The stager's workers now enter `torch.inference_mode()`, `record_stream` is
  gone (nothing is ever outstanding on a side stream when `copy` returns, since
  each worker synchronises its own stream per chunk), and `copy` cannot raise.
* The conditioner is now *built* with `conditioning_layer + 1 = 51` decoder
  layers, from a `Qwen3VLConfig` edited before `from_pretrained`, so the 13
  unreachable layers' checkpoint keys match no parameter and safetensors never
  faults their pages off Lustre. On round 7's numbers that should take the
  build from ~929 s to ~740 s, which is the difference between 230 s and 420 s
  of margin against the 1560 s measurement timeout. Falls back to the full load
  on any exception.
* Every request reports allocated device memory after the conditioner is
  unstaged.

Result: **reward 1.3631, gate passed.** Job 19014104 COMPLETED in 1521 s.
Commit `5ed43fe`.

| case | seconds | speedup | LPIPS | text_stage | transformer_stage |
|------|---------|---------|-------|-----------|-------------------|
| minimax_h3-02 | 137.29 | 1.3494 | 0.0944 | 40.06 s @ 1.22 GiB/s | 1.87 s @ 20.09 GiB/s |
| minimax_h3-03 | 132.52 | 1.3814 | 0.0827 | 22.55 s @ 2.17 GiB/s | 1.91 s @ 19.67 GiB/s |
| minimax_h3-04 | 112.25 | 1.6251 | 0.0153 | **2.37 s @ 20.59 GiB/s** | 1.94 s @ 19.37 GiB/s |

Everything about this round is in those two columns.

**The three LPIPS values are bit-identical to round 4's**, which is the check
that mattered: the conditioner truncation and the threaded stager are exactly
lossless, and `H3_STAGE_VERIFY` reported `mismatched=0` for both models.

**The stager works.** Where the host pages are resident it moves weights at
19.4-20.6 GiB/s against the 6.6-9.9 GiB/s of `Tensor.to`. Case 04 staged both
models in 4.31 s where round 4 needed 12.03 s, and that is the whole of its
improvement from 120.34 s to 112.25 s. Page-locking cost 0.17 s.

**The conditioner's host pages are not resident.** The same 48.88 GiB, staged by
the same code, took 40.06 s on request 1, 22.55 s on request 2 and 2.37 s on
request 3. Safetensors hands back a file mapping; every request reads the
conditioner's 48.88 GiB and then the transformer's 37.49 GiB on top of it, and
the kernel reclaims whichever it needs to. The transformer, read last, stays;
the conditioner, read first, is gone by the next request. Rounds 1-6 hid this
because `Tensor.to`'s own bounce was slow enough (9.2 GiB/s) that a cold read at
1.2-2.2 GiB/s only looked like ordinary variance.

Build instrumentation: **759.1 s total, 748.9 s of it the conditioner** — the
51-layer construction took 180 s off round 7's 928.8 s, as predicted. But the
transformer's *first* staging cost **201.4 s at 0.31 GiB/s** against round 5's
34.8 s at 1.77 GiB/s through the serial path. Four threads faulting 96 MiB
chunks out of one Lustre mapping is far worse than one sequential reader; the
stager is a win on resident memory and a disaster on cold pages. That 201 s is
untimed but it is 13% of the measurement window, and round 8 finished with only
39 s to spare.

Kernel table unchanged from round 1 (flash attention 59.7%, 200 GEMMs 32.4%,
qknorm_rope 3.7%), which is expected — nothing this round touched the forward.

## Round 9 — make the host masters unreclaimable

Hypothesis: both of round 8's staging problems are the same problem, that the
host masters are file-backed. Copying them into anonymous memory once at build
time fixes the reclaim (anonymous pages cannot be evicted) and moves the cold
Lustre read out of the measurement window and into build, where it is one
sequential reader instead of four competing ones. `_HostResident.materialize`
does it one tensor at a time, installing the replacement and dropping the
original before moving on, so host memory never holds two copies. 110.6 GiB of
anonymous memory against the job's 240 GiB, dropping to 86.4 GiB once the AdaLN
masters are released after the first request.

Everything else is held fixed — FirstBlockCache stays at round 4's `hold` at
0.032 — so the three LPIPS values should come back 0.094 / 0.083 / 0.015 for the
third round running and the only thing that moves is the clock. Projection: all
four generations stage in ~4.3 s instead of 12.0-42.0, case 02 around 101 s.

Result: **TIMEOUT, reward 0.0. Job 19015224 FAILED after 1583 s.** The round is
spent. The conditioner materialised exactly as projected -- `H3_MATERIALIZE text
gib=48.88 seconds=87.5`, and the 51-layer lazy construction worked again
(`H3_CONDITIONER_LAZY kept=51 dropped=13`) -- and that line is the last thing
the process ever printed. The harness killed it 1560 s in, inside the
transformer's materialize.

So the transformer's 61.73 GiB cold read took more than 700 s that day, against
the 201 s round 8 paid for the same bytes through the stager and the 34.8 s
round 5 paid through the serial path. Under 0.09 GiB/s. **Lustre read bandwidth
on this file is not a constant and I had been treating it as one:** 1.77, 0.31
and now <0.09 GiB/s for the same 61.73 GiB on three different days. Every
projection I have made that put a cold read inside a fixed budget was a
projection against a number that moves by 20x.

Two conclusions, both structural rather than about this experiment:

1. The conditioner materialize is cheap and safe *because its pages are already
   warm* -- `from_pretrained` has just read them. The transformer materialize is
   the opposite: nothing has read those pages yet, so it is a cold read that I
   was performing eagerly for no reason, when the warmup generation is going to
   perform it anyway. Round 9 paid a cold read at build *in addition to* the one
   it was trying to avoid.
2. Anything whose cost is a cold Lustre read needs a deadline, not an estimate.

## Round 10 -- serial first staging, and the reuse-run A/B

Hypothesis, systems half: keep only the half of round 9 that worked.
`H3_MATERIALIZE` defaults to `text`, so the conditioner (warm, 87.5 s, bounded)
is copied into anonymous memory and the transformer stays mapped. The cold read
of the transformer happens exactly once, in the untimed warmup, and
`_HostResident.load` now routes a residency's *first* staging through the serial
path (`if not (self.anonymous or self.loads): stager = None`) because four
threads faulting one Lustre mapping measured 0.31 GiB/s against the sequential
reader's 1.77. `materialize` also takes a deadline now
(`H3_MATERIALIZE_BUDGET_S`, 240 s) and reports `complete=`; a partial copy
leaves `anonymous` false, which is exactly the condition that keeps the serial
path for that residency's first load. Round 9 cannot repeat.

Hypothesis, quality half -- this is the round's real question. Round 5 raised
the threshold 0.032 -> 0.055 and failed the gate at LPIPS 0.271 / 0.385 / 0.021.
Two mechanisms are confounded in that result and I have never separated them:
the *number* of reuses rose (16 -> 23 on case 03), and so did the *length of
consecutive runs* of them (max 2 -> max 3, with several runs of 3). If damage is
roughly linear in reuse count, run length is irrelevant and only the total
matters. If damage is superlinear in run length -- which is what the drift
metric's behaviour suggests, since it roughly doubles after each reuse and so
the second and third steps of a run extrapolate from an increasingly stale tail
-- then the runs are doing most of the harm and the count is nearly free.

Round 10 holds threshold at **0.055**, warmup 4 and tail 3 -- round 5's exact
settings -- and changes **max consecutive reuse from 3 to 1**, and nothing else.
That forces strict compute/reuse alternation: every reused step extrapolates
from a tail residual computed one step earlier, never two or three. Expected
~20-21 reuses on case 03 against round 5's 23, so if count is what matters LPIPS
lands near 0.34 and the gate fails again; if run length is what matters it
should fall well under 0.25 while keeping most of round 5's 2.31x on case 02.

Projection if the gate holds: case 02 ~90.6 s (2.05x), 03 ~94.5 s (1.94x), 04
~94.5 s (1.93x), reward ~1.7 against round 4's 1.476. Budget: build ~660 s +
88 s materialize, warmup ~140 s including the serial cold read, three timed
requests ~285 s, ~1175 s of 1560.

Result: **gate failed, reward 0.0. Job 19016699 COMPLETED in 1446 s.** Raw
reward 1.583. LPIPS **0.2476 / 0.3851 / 0.0168**, mean 0.2165 (passes the 0.25
mean gate), worst 0.3851 (fails the 0.30 worst gate, on case 03 as always).
Speedups 2.073x / 1.999x / 1.994x at 89.36 / 91.56 / 91.49 s.

The A/B answers its question, and the answer is **no**:

| | threshold | max run | case 03 reuses | case 03 LPIPS |
|---|---|---|---|---|
| round 5 | 0.055 | 3 | 23 | 0.38515 |
| round 10 | 0.055 | 1 | 20 | 0.38512 |

Three fewer reuses, and strict compute/reuse alternation instead of runs of up
to three, moved case 03's LPIPS by **0.00003**. Run length is not the mechanism,
and at this operating point neither is the reuse count. Round 4, at threshold
0.032, took 12 reuses on case 03 and scored 0.0827. So between 12 and 20 reuses
case 03 goes 0.083 -> 0.385 and then saturates: the curve is a cliff with a flat
top, not a slope, and the thing that moves along it is **which** steps are
admitted, i.e. how much per-step drift the threshold tolerates -- not how many
are admitted or in what pattern.

The predictor scoring killed the last in-flight signal I had. Per case the
`hold` prediction error was 0.2262 (02) / 0.1914 (03) / 0.1650 (04) while LPIPS
was 0.248 / 0.385 / 0.017. Case 03 has the *second lowest* measured prediction
error and by far the worst LPIPS; case 02 has the worst prediction error and
1.5x better LPIPS. That is the fourth in-flight signal to fail to order the
cases, after the raw drift trace, the drift trace's shape, and the tail-residual
magnitude. Case 03 is simply more LPIPS-fragile per unit of latent error, and
nothing the transformer can see during the forward says so. I am done looking
for an adaptive per-case signal; the operating point has to be global and has to
be chosen so that the most fragile case clears 0.30.

Systems, which all worked:

- The conditioner materialize under a deadline **saved the round**. It ran out
  of budget at 242.7 s having copied 39.55 of 48.88 GiB, left `anonymous` false,
  and the first staging correctly fell back to the serial path. Round 9's
  failure mode is closed.
- That partial copy was still enough: text staging cost 10.9 s in warmup and
  **2.31 / 2.83 / 2.67 s** in the three timed requests, at 17-21 GiB/s, against
  round 8's 40.06 / 22.55 / 2.37. The page-reclaim problem is solved.
- The transformer's cold read through the serial path cost 37.9 s at 1.63 GiB/s,
  against round 8's 201 s at 0.31 through the stager. Confirmed.
- Per-request overhead outside denoise is now 8.3 s (2.5 staging, 4.3 video
  decode, the rest small), so latency is denoise and nothing else.

**And one number that does not fit.** The materialize loop read the
conditioner's mapping at 0.16 GiB/s while, minutes later on the same node, the
staging path read the transformer's at 1.63. `copy_` is a memcpy once a page is
resident, so materialize is bandwidth-bound on faulting the mapping, and 10x is
too big for noise. Two candidate causes, and they compound: the loop walks
`self.host` in *module* order, which is not file order, so it faults a 48.88 GiB
mapping in scattered order; and the conditioner lives in the container image
(`/workspace/MiniMax-H3/weights`), while the transformer is a bind mount of
plain files, so they may not even be the same kind of storage -- a compressed
squashfs punishes a random-order walk far worse than 10x.

## Round 11 -- walk the mapping in address order, and step the threshold

Hypothesis, systems: sort the materialize groups by `data_ptr` so the copy walks
the mapping front to back. If readahead is the explanation, 48.88 GiB completes
in ~30-40 s instead of running out of a 180 s budget at 39.55, and the whole
conditioner becomes anonymous, which also restores the guarantee that no request
can be caught by a reclaim. `_Marks` now reports CPU time beside wall time for
every phase, which settles the other open question for free: the conditioner's
`from_pretrained` has cost 745-929 s in four rounds and I do not know whether
that is a Lustre read or module construction, and those have nothing in common
as targets. An `H3_INVENTORY` line names the filesystem behind each weight path
and lists the fused-attention libraries present in the image -- FlashAttention
is 59.7% of a forward at 326 TFLOPS, a third of this H100's bf16 peak, so
whether an FA3 build exists here is worth knowing before spending a round on it.

Hypothesis, quality: go back to round 4's structure -- warmup 3, tail 2, max_run
2, all known-good -- and step the threshold 0.032 -> **0.036**. This is a
deliberate small step onto the cliff rather than a jump across it: round 4's
case 03 sat at 0.0827 with 12 reuses and a 0.30 budget, which is an enormous
margin, and the entire 0.032-0.055 interval is unmeasured. Everything else about
the round is round 10's, and round 10 showed that the same quality settings now
cost ~15 s less per request than they did in round 4, so even round 4's exact
settings would score ~1.57 today against its recorded 1.476.

Projection: case 02 ~19 reuses at LPIPS ~0.13, case 03 ~14 at ~0.15, case 04 ~16
at ~0.017; requests ~97 / ~106 / ~102 s; reward ~1.60. Window: build ~790 s if
the address-order walk works (1000 if it does not and the budget binds), warmup
~150, three requests ~305, so 1245-1455 of 1560.

Result: **reward 1.4995 -- a new best, gate passed. Job 19018271 COMPLETED in
1053 s.** LPIPS 0.2479 / 0.1247 / 0.0170, mean 0.1299 (limit 0.25), worst 0.2479
(limit 0.30). Latencies 98.22 / 110.35 / 110.15 s at 1.886x / 1.659x / 1.656x.

Three separate findings, and the third is the biggest of the campaign so far.

**1. The address-order walk works, and the build was never IO-bound.** The
materialize loop went from 0.16 GiB/s on 39.55 GiB to **0.57 GiB/s on the full
48.88 GiB in 85.4 s, `complete=1`**. Sorting the copy by `data_ptr` was worth
3.6x. The new CPU-vs-wall line then contradicted my whole model of the build:

    H3_PHASES     build total=283.0 text_encoder=188.7 vae=8.2 materialize=85.4
    H3_PHASES_CPU build total=281.2 text_encoder=212.8 vae=7.6 materialize=60.1

CPU time exceeds wall time for the conditioner (it is threaded), so
`from_pretrained` is **CPU-bound module construction, not a Lustre read**. The
745-929 s I have been attributing to storage in rounds 7-10 was contention for
this node's eight cores. And `H3_INVENTORY` shows why storage was never the
issue for the conditioner at all: `/workspace/MiniMax-H3/weights/text_encoder`
is on **ext4** (the container image, node-local), while the transformer's bind
mount is `lustre`. I had the two exactly backwards. The build is now 283 s
instead of 999, which frees ~700 s of the 1560 s measurement window.

**2. There is no FlashAttention-3 in this image**, and no xformers, no
Transformer Engine, no `flash_attn` at all: `torch 2.9.1+cu128` and `triton
3.5.1`. So the 326 TFLOPS the SDPA flash backend reaches -- a third of this
H100's bf16 peak, which is the well-known ceiling for FA2 on Hopper -- cannot be
lifted by installing a better library. The only two routes to Hopper-class
attention here are cuDNN's fused kernel (measured at 1.75x in round 3, and
0.508 LPIPS) and writing one in Triton.

**3. One step is responsible for case 02's entire quality loss.** The reuse sets
are exactly nested:

| | case 02 reuse steps | LPIPS |
|---|---|---|
| round 4 | 6,8,...,38 (17) | 0.0944 |
| round 11 | **4**,6,8,...,38 (18) | 0.2479 |
| round 10 | **4**,6,...,44 (21) | 0.2476 |

Reusing step 4 costs **+0.154 LPIPS**. Adding steps 40, 42 and 44 on top of it
costs **-0.0003**. That is the cliff-then-plateau shape rounds 5 and 10 saw from
the other side, and it is not about how many steps are reused at all: it is
about *which*. The early steps are where the trajectory is decided and the late
ones are where it is already decided.

Which makes the relative-L1 metric worse than useless over the back half of the
schedule. It is U-shaped -- large at both ends -- so a flat threshold spends its
budget protecting the cheap late steps and then, one small step up in threshold,
buys the single most expensive early step. Case 02 at threshold 0.036 computed
steps 40-46 because their drift was 0.037-0.082, and reused step 4 because its
drift was 0.0360. Both decisions were backwards.

## Round 12 -- protect the early steps, stop metering the late ones

Hypothesis: split the schedule. Raise `warmup_steps` 3 -> **6**, which protects
steps 4 and 5 outright -- the measurement says that alone is worth +0.15 LPIPS
on case 02 -- and add a `late_fraction` past which the threshold is simply
dropped and reuse is limited only by the tail guard and a consecutive-reuse cap.
At 0.72 of 49 steps the late region starts at step 36; `tail_steps` rises 2 ->
3 so steps 46-48 stay dense, which keeps every newly reused step at or below
step 45 and so within one step of ground that rounds 4 and 10 already measured.
Threshold stays at round 11's 0.036 and `max_run` at 2 for the metered region,
so the early and middle of the schedule behave exactly as they did in round 4.

Predicted reuse sets, read off round 11's traces: case 02 {6,8,...,34} +
{36,37,39,40,42,43,45} = 22 reuses against round 4's 17; case 03 {11,13,...,35}
+ {36,38,39,41,42,44,45} = 20 against round 11's 13. Predicted latency 86.7 /
92.3 / 92.3 s (2.14x / 1.98x / 1.98x) and, if late reuse is as nearly free for
cases 03 and 04 as it measurably is for case 02, LPIPS around 0.11 / 0.15 /
0.02 for a reward near **1.84**. The window is no longer a constraint: build
283 s + warmup ~130 + three requests ~271 = ~684 s of 1560.

Result: **reward 1.8768, a new best by a wide margin. Job 19018972 COMPLETED.**
LPIPS 0.0951 / 0.1263 / 0.0231, mean 0.0815 (limit 0.25), worst 0.1263 (limit
0.30). Latencies 86.57 / 91.84 / 91.28 s at 2.140x / 1.993x / 1.999x. Reuse
counts 22 / 20 / 20 against round 11's 18 / 13 / 13.

The prediction was exact where it could be checked. Case 02 reused 22 steps and
came back at **0.0951** against round 4's **0.0944 on 17** -- the five extra
reuses, all of them in the late region, cost **0.0007 LPIPS between them**, and
5.7 s per request. Reusing step 4 costs 0.154; reusing steps 36 through 45 costs
0.0007. That is a factor of two hundred between two steps of the same schedule,
and the relative-L1 metric ranks them the other way round.

Case 03 confirms it from the other direction: 13 reuses at 0.1247 in round 11,
20 reuses at 0.1263 in round 12, all seven additions late. Case 04 moved 0.0170
-> 0.0231, still 13x inside its budget.

The quality gate is no longer close. Worst-case LPIPS is 0.126 against a 0.30
limit and the mean is 0.0815 against 0.25 -- there is more than twice the
damage budget left unspent, and rounds 5 and 10 show what the ceiling looks
like when it is spent badly.

## Round 13 -- decide the attention backend on real activations

Hypothesis: the single largest remaining lever is not the cache. One computed
forward is 59.7% FlashAttention at **326 TFLOPS, a third of this H100's bf16
peak**, and round 11 established there is no FlashAttention-3, no xformers and
no Transformer Engine in this image to replace it with. The one Hopper-class
attention kernel that *is* here is cuDNN's, measured in round 3 at **1.75x**.
Round 3 also measured it at LPIPS 0.508 on case 02 and I have treated it as
closed ever since.

But both accuracy checks that cleared cuDNN -- round 4's at length 4096, round
11's at the real 19487 -- ran on `torch.randn`. Random keys give a softmax that
is nearly uniform over nineteen thousand rows; real ones are dominated by a
handful of entries. A kernel that mishandles a peaked distribution, or a ragged
last tile, has nothing to show on random input, which is exactly the pattern of
"equally accurate on the bench, 0.508 in the pipeline". So the check has been
measuring the wrong thing, not answering the wrong question.

`calibrate` re-runs the decision on the **model's own block-0 query, key and
value**, taken from the first forward of the untimed warmup generation. Each
backend and layout is measured two ways: worst relative error against a float32
reference over the head, middle and tail rows of the real sequence, and mean
relative divergence from FlashAttention's own output over the *whole* tensor,
which sees every row and every head. A backend is eligible only if both are
within 1.25x of flash's own error against float32, and the fastest eligible one
wins. A calibrated plan is trusted only on sequences at least half the packed
length, so the token refiner's ~200-row call keeps flash unconditionally --
there are no real activations from it to calibrate on, and a kernel that is
silently wrong on a short ragged shape would corrupt the round invisibly.

This is a one-sided bet. If cuDNN is broken on real activations, the gate
rejects it, the run reproduces round 12 exactly, and the round costs one of the
thirteen remaining. If it is not, 1.75x on 59.7% of every computed forward is
1.43x on denoise, and round 12's 2.14x becomes about 2.9x at unchanged LPIPS,
for a reward near 2.6. The cache configuration is held at round 12's values
precisely so that any LPIPS movement is attributable to the kernel and nothing
else.

Result: **gate failed by 0.003, reward 0.0, raw reward 2.083. Job 19019722
COMPLETED.** LPIPS 0.3030 / 0.1990 / 0.0257, mean 0.1759 (passes), worst 0.3030
against a limit of 0.30. Latencies **70.13 / 73.56 / 73.58 s at 2.642x / 2.489x
/ 2.479x** -- the fastest the pipeline has ever run, and three one-thousandths
of LPIPS from a reward of 2.08.

The calibration did its job and then some:

    H3_ATTENTION_CALIBRATE seq=19487 rows=96 tolerance=1.25 flash_err=1.079e-01
      chosen=cudnn/none
      cudnn/none=17.62ms/err1.079e-01/div2.892e-04
      flash/none=30.23ms/err1.079e-01/div0.000e+00

On the model's own block-0 activations cuDNN is **1.72x** FlashAttention, its
error against the float32 reference is identical to flash's to four digits, and
its mean divergence from flash's full output is **2.9e-4**. Round 3's LPIPS of
0.508 was never a broken kernel.

And yet case 02 moved 0.0951 -> 0.3030. A mean relative difference of 2.9e-4 in
one attention output, compounded over 50 blocks and 27 computed steps, is worth
0.21 LPIPS. That is the clearest measurement in this campaign of how much this
sampler amplifies a small perturbation -- and it retires the idea that "within
kernel noise" means "safe". It also explains the round 3/4/11 contradiction
completely: the accuracy checks were right, cuDNN *is* as accurate as flash, and
being as accurate as flash is not the same as being flash.

## Round 14 -- give the attention kernel the same schedule split as the cache

Hypothesis: if 2.9e-4 of kernel difference matters only through the trajectory,
then it must matter where the trajectory is decided, and the cache rounds have
already located that. Reusing step 4 costs case 02 0.154 LPIPS; reusing steps 36
through 45 costs it 0.0007. So run FlashAttention -- the baseline's own kernel,
bit-for-bit the reference trajectory -- for the first `H3_ATTENTION_DENSE_STEPS`
denoising steps, and cuDNN for the rest. `set_adaln_step` now tells the
attention backend which step it is on, and the calibrated plan applies only from
step 12 onward (and, as before, only to sequences at least half the packed
length, so the token refiner is untouched).

Steps 0-11 contain 9 computed forwards on case 02 and 11 on case 03, at
50 blocks x 12.6 ms of extra attention each, so the cost is 5.7 s and 6.9 s per
request: 75.8 / 80.5 / 80.5 s, or 2.44x / 2.27x / 2.27x. The quality question is
which model of the damage is right. If it is uniform in the number of cuDNN
steps, case 02 lands at 0.095 + 0.208 x 37/49 = 0.25 and the gate passes with
0.05 to spare. If it is concentrated early, the way every cache measurement has
been, it lands far below that. Either way the round answers whether the
early-step principle generalises from "skip the whole stack" to "use a different
kernel", which decides how the remaining rounds are spent. The cache stays at
round 12's configuration so that, once again, only one thing moves.

Result: **reward 2.1439 -- a new best by 0.27 over round 12. Job 19020995
COMPLETED.** LPIPS 0.1159 / 0.0804 / 0.0235, mean 0.0733 (limit 0.25), worst
0.1159 (limit 0.30). Latencies 76.31 / 80.86 / 80.84 s at 2.428x / 2.264x /
2.256x.

The early-step principle generalises, and by more than the arithmetic allowed
for. Running FlashAttention for the first twelve steps and cuDNN for the other
thirty-seven:

| case | all-flash (r12) | all-cuDNN (r13) | flash first 12 (r14) |
|---|---|---|---|
| 02 | 0.0951 | 0.3030 | **0.1159** |
| 03 | 0.1263 | 0.1990 | **0.0804** |
| 04 | 0.0231 | 0.0257 | **0.0235** |

A uniform-damage model predicted 0.25 for case 02; it came in at 0.116. Twelve
steps out of forty-nine -- and only nine of them actually computed -- carry
**90% of the damage** cuDNN does over the whole schedule. Case 03 went one
better and came out *below* its all-flash value, which I will not try to
explain beyond noting that at this amplification the trajectory is not
monotone in the perturbation.

Two principles now have three independent confirmations each: approximate late,
never early; and a per-call difference of 2.9e-4 is not small once it is
compounded 50 blocks x 27 steps deep.

## Round 15 -- spend the quality budget where it is cheapest

The gate is wide open again: worst-case 0.1159 against 0.30, mean 0.0733
against 0.25. And the campaign has now measured the price of the two ways of
spending it. A late reuse costs on the order of 0.0001 LPIPS and saves a whole
computed forward, 2.28 s at cuDNN speed; a dense step costs 0.63 s of extra
attention and buys back about a tenth of cuDNN's damage. The late reuses are
several orders of magnitude the better deal, so the budget goes there.

Hypothesis: move `late_fraction` 0.72 -> **0.50**, which starts the unmetered
region at step 25 instead of 36, and raise `late_max_run` 2 -> **3**. The
metered region (steps 6-24, threshold 0.036, run cap 2) and the dense-attention
prefix (steps 0-11) are both untouched, so everything this round changes lies in
the back half of the schedule, which is where every measurement says damage is
cheap. Predicted reuse counts 26 / 23 / 23 against round 14's 22 / 20 / 20, for
67.8 / 75.8 / 75.8 s -- 2.73x / 2.42x / 2.41x -- and, if the late region is as
cheap at a run of three as it has been at a run of two, a reward near **2.26**.

The one genuinely new thing being tested is a run of three consecutive reuses in
the middle-late region, steps 25-35, which no round has run. Round 6 established
that runs are not free in principle -- three-in-a-row was part of what round 5
was doing when it scored 0.385 -- but round 10 then showed that at a fixed step
set the run length was worth 0.00003. This round separates those by holding the
early half fixed and varying only the back.

Result: **reward 2.3837 -- a new best. Job 19022197 COMPLETED.** LPIPS 0.1167 /
0.0845 / 0.0302, mean 0.0772 (limit 0.25), worst 0.1167 (limit 0.30). Latencies
66.17 / 73.69 / 73.59 s at **2.800x / 2.484x / 2.479x**. Reuse counts 26 / 23 /
23, exactly as predicted.

Against round 14, four extra reuses per case and a consecutive-reuse cap of
three instead of two, all of it in the back half of the schedule, cost:

| case | round 14 | round 15 | delta |
|---|---|---|---|
| 02 | 0.1159 | 0.1167 | +0.0008 |
| 03 | 0.0804 | 0.0845 | +0.0041 |
| 04 | 0.0235 | 0.0302 | +0.0067 |

...and bought 10.1 / 7.2 / 7.3 seconds. That is a fifth of a second of LPIPS
budget for a tenth of the request. The run of three in steps 25-35, the one
genuinely untested thing in the round, is not visible in the result.

Worst-case LPIPS is still 0.117 against 0.30 and the mean is 0.077 against 0.25.
Three rounds running, the late region has refused to charge for anything.

## Round 16 -- find where the late region stops being free

Hypothesis: it has to stop somewhere, and the only way to find the edge is to
walk over it. `late_fraction` moves 0.50 -> **0.35**, which starts the unmetered
region at step 18 instead of 25, and `late_max_run` moves 3 -> **4**. Steps 0-5
stay dense, steps 6-17 stay metered at threshold 0.036 with a run cap of 2, and
the first twelve steps keep FlashAttention. Everything that has been measured as
expensive is therefore untouched; what changes is that two thirds of the
schedule, rather than half, becomes unmetered, and reuses may now run four deep.

Predicted for case 02: metered reuses {6,8,10,12,14,16} and a late region of 28
steps at four reuses per computed step, for 29 reuses and 20 computed forwards
against round 15's 26 and 23. Nine of those twenty are the FlashAttention
prefix, so denoise falls to about 52.8 s and the request to **61.2 s, 3.03x**.
Cases 03 and 04 should land near 2.7x. If the late region charges for this the
way it has charged for everything so far -- a few thousandths -- the reward is
near **2.6**; if four-deep reuse from step 18 is where it finally breaks, the
round costs one of the ten that remain and round 15's 2.3837 stands.

Result: **reward 2.6554 -- a new best. Job 19023163 COMPLETED.** LPIPS 0.0997 /
0.1177 / 0.0444, mean 0.0873 (limit 0.25), worst 0.1177 (limit 0.30). Latencies
59.68 / 65.01 / 64.87 s at **3.104x / 2.816x / 2.812x**. Reuse counts 29 / 26 /
26, again exactly as predicted.

The edge was not found. Moving the unmetered region from half the schedule to
0.35 of it and allowing four-deep reuse bought another 6.5 / 8.7 / 8.7 seconds,
and case 02's LPIPS went **down**, 0.1167 -> 0.0997. Cases 03 and 04 paid
+0.033 and +0.014. Twenty-nine of forty-nine steps now skip forty-nine of fifty
blocks and the worst case is still at 39% of its budget.

Case 04 is the one showing a trend -- 0.0235, 0.0302, 0.0444 over rounds 14, 15,
16, roughly doubling each time -- but from a base so low it would need three
more doublings to matter. Case 03 is the one to watch: 0.0804, 0.0845, 0.1177.

## Round 17 -- one more step, and then the returns flatten

Hypothesis: `late_fraction` 0.35 -> **0.25** (unmetered from step 13) and
`late_max_run` 4 -> **6**. Predicted for case 02: metered reuses {6,8,10,12} and
a late region of 33 steps at six reuses per computed forward, for 33 reuses and
16 computed forwards, of which nine are the FlashAttention prefix. Denoise falls
to about 43.9 s and the request to **52.3 s, 3.54x**.

I expect this to be near the end of the useful range even if it passes, and the
arithmetic says why. The four extra reuses are worth +14% on speed; if case 02's
LPIPS goes 0.0997 -> 0.18 the quality factor drops 9%, and the product gains 5%.
Two rounds ago the same trade was worth 14% for nothing. The lever is running
out, and the fixed overhead that the cache cannot touch -- 2.3 s of conditioner
staging, 1.9 s of transformer staging, 4.3 s of video decode, 8.4 s in all -- is
now **16% of a request** and growing as a share. That is where the next round
goes, and unlike everything since round 12 it is exactly lossless.

Result: **reward 2.8100 -- a new best. Job 19024027 COMPLETED.** LPIPS 0.1771 /
0.1712 / 0.0650, mean 0.1377 (limit 0.25), worst 0.1771 (limit 0.30).
Latencies 51.27 / 59.12 / 59.02 s at **3.613x / 3.097x / 3.091x**. Reuse counts
33 / 30 / 30.

The lever did flatten, exactly as the arithmetic said it would. Speed gained
16%, the quality factor lost 8.6%, and the reward gained 5.8% -- against round
15 -> 16, which gained 11% for nothing. Case 02's LPIPS went 0.0997 -> 0.1771
and case 03's 0.1177 -> 0.1712. There is one more step of this size left before
the worst case reaches 0.26, and it would be worth perhaps +0.1.

What the round really did was move the bottleneck. Case 02's denoise is now
39.4 s, and **26.2 s of it is nine forwards** -- the six dense warmup steps and
three more inside the FlashAttention prefix, all at 2.91 s each. Sixteen
computed forwards, nine of them the expensive kind. Meanwhile the fixed
overhead the cache cannot touch has become 8.6 s of a 51.3 s request, **17%**:

    text_stage=2.28  transformer_stage=1.87  video_decode=4.25  audio_decode=0.05

## Round 18 -- the decode

Hypothesis: the video decode is now the single largest thing outside the
denoise loop, and it is the most favourable place in this pipeline to touch.
Everything since round 12 has been an approximation *inside* the sampler, where
rounds 13 and 14 measured that 2.9e-4 per call compounds into 0.21 LPIPS. The
decode is the last operation: whatever it does goes straight to the output with
nothing left to amplify it. It is also the only hot path that is still a plain
module stack rather than hand-fused Triton, and it runs the same decoder once
per 256x256 spatial tile and per latent chunk, which is a long sequence of small
launches -- the shape inductor is built for.

So `vae.decoder` is wrapped in `torch.compile(dynamic=False)`. Compilation
happens on its first call, inside the untimed warmup generation, and the round
has roughly 600 s of unused measurement window to pay for it. The same round
turns on the decode kernel profile, which has existed since round 1 and never
been run: if the compile is a wash, the table says what the 4.25 s actually is,
and round 19 can target it directly. Tiling itself is not negotiable -- the
released frames are the blended-tile ones, so the tile geometry is part of the
output contract.

Everything else is held at round 17's configuration, so any movement in LPIPS
is the decoder's and nothing else's. Expected: 47-49 s on case 02 if inductor
finds a third of the decode, for a reward near **2.9**; no loss if it finds
nothing.

Result: **reward 2.8969 -- a new best. Job 19025208 COMPLETED.** LPIPS 0.177077 /
0.171167 / 0.064969, mean 0.1377 (limit 0.25), worst 0.1771 (limit 0.30).
Latencies 49.89 / 57.17 / 57.23 s at **3.714x / 3.202x / 3.188x**.

The decode fell 4.25 -> 2.78 s, 35%, and **LPIPS moved by 1.1e-7**:

    case      round 17            round 18            delta
    02        0.17707671183       0.17707679216       +8.0e-8
    03        0.17116675694       0.17116686654       +1.1e-7
    04        0.06497192449       0.06496923086       -2.7e-8

That is the round's real finding, and it is the counterweight to round 13. The
same class of change -- swap a numerically equivalent kernel for a faster one --
cost 0.21 LPIPS when it was made *inside* the sampler and costs 1e-7 when it is
made at the end of the pipeline. Position in the pipeline, not the size of the
numerical difference, is what decides whether an approximation is affordable.
Reward went up 3.1% for free.

The decode profile that came with it says the 2.78 s is not launch overhead
hiding behind a compile: 105 compiled decoder calls, 44.8% of device time in
`aten::mm` (1.19 s over 19110 calls) and 17.0% in FlashAttention (0.45 s over
3780). The VAE decoder is GEMM- and attention-bound at its tile geometry, and
that geometry is part of the output contract. Decode is close to its floor; it
is no longer the place to look.

## Round 19 -- a computed step does not have to be a whole step

Hypothesis. Case 02's request is 49.89 s, of which denoise is 42.52 s. Sixteen
of the forty-nine forwards are computed and nine of those run dense
FlashAttention at ~2.91 s each -- roughly 26 s, **52% of the whole request**.
Six of those nine are the warmup steps 0-5, which are dense by construction.

They are dense because round 12 measured what happens when they are not:
reusing step 4 -- skipping 49 of 50 blocks -- moved case 02 from 0.0944 to
0.2479 LPIPS, while the identical skip applied at steps 40, 42 and 44 moved it
by -0.0003. Every round since has treated that as a binary: an early step is
either computed whole or skipped whole.

It is not a binary. The cache reconstructs a skipped step as
`block_0(x) + tail`, where `tail` is the summed contribution of blocks 1..49
captured on the last computed step. Nothing forces the split to sit after block
0. On a **partial** step the first `K` blocks are computed and only the
contribution of blocks `K..49` is reused, from a cache refreshed on the
immediately preceding full step. The cost is `K/50` of a forward instead of
1/50, and the reused quantity is the deep tail -- the part of the stack whose
residual should be both smaller and slower-moving than the whole.

So: `K = 25`, applied to steps 1, 3 and 5, alternating with full steps 0, 2 and
4 so no partial step ever reads a cache older than one step. Step 0 always runs
whole to seed it. A partial step counts as having computed block 0, so the
change detector's `previous_head_residual` is refreshed honestly, but it does
**not** refresh the full tail residual, which it never computed.

Arithmetic: three steps at half price saves 3 x 1.45 = 4.4 s on case 02, taking
it to ~45.5 s and 4.07x. If the quality factor holds, reward ~3.1. If it does
not, the trace still says exactly what a half-stack reuse costs at the most
sensitive point in the schedule, which is the number every remaining round
needs. Everything else is held at round 18's configuration.

Result: **gate failed, reward 0. Job 19026129 COMPLETED in 19:46.** LPIPS
0.6335 / 0.3924 / 0.1765, mean 0.4008 (limit 0.25), worst 0.6335 (limit 0.30).
Raw reward before the gate 2.139.

The speed arrived exactly as predicted -- 45.84 / 53.17 / 52.81 s, 4.042x /
3.443x / 3.454x, 4.05 / 4.00 / 4.41 s saved from three half-price steps. The
quality did not. Case 02 went 0.177 -> 0.634. **Three partial steps at K = 25
did more than four times the damage of the single full skip at step 4 that
round 12 measured at +0.154.** Reusing half the stack for three steps is worse
than reusing 98% of it for one.

That inverts the intuition the round was built on, and the reason is worth
stating precisely. What the cache reuses is not "a residual", it is the value of
a function at a stale argument. For the shallow split that argument is the
output of block 0, which moves by 3-6% of its own magnitude between adjacent
steps -- that is exactly what the relative-L1 detector measures. For the K = 25
split the argument is the state after twenty-five blocks, which is the step-to-
step difference *amplified* through twenty-five residual blocks. The deep
segment's own contribution is smaller, but the argument it is evaluated at is
far staler, and staleness is what the cache pays for. So the split after block 0
is not an arbitrary implementation choice inherited from FirstBlockCache; it is
the split that minimises what reuse costs, and every deeper split in this family
is dominated by it. The mechanism is reverted and the finding is recorded in the
source next to the split.

Ledger: nineteen rounds started, best still round 18 at 2.8969.

## Round 20 -- spend the quality budget where it is cheapest

Hypothesis. The gate allows mean 0.25 and worst 0.30; round 18 sits at 0.1377
and 0.1771. Roughly 40% of the quality budget is unspent, and rounds 15, 16 and
17 each converted a slice of it into speed at a good exchange rate by widening
the unmetered late region. Round 19 was an attempt to buy speed somewhere else
entirely and it failed, so the budget goes back into the lever that works.

Where exactly it goes matters. Case 02's trace shows sixteen computed steps:

    warmup     0 1 2 3 4 5        dense FlashAttention, 2.91 s each
    threshold  7 9 11             dense FlashAttention, 2.91 s each
    run        18 25 32 39        calibrated cuDNN, 2.28 s each
    tail       46 47 48           calibrated cuDNN, 2.28 s each

The four `run` steps exist for one reason: the consecutive-reuse cap of six
forces a recompute every seventh step in a region where the threshold has been
switched off because it is anti-correlated with damage there. They are not
chosen by any measurement of damage. Raising the cap to nine turns four of them
into two -- computes at 22 and 32 and 42 instead of 18, 25, 32 and 39 -- and
removes two full forwards, **4.6 s**, without touching a single step before 13.

This is deliberately the opposite of round 19's bet: rather than making an
expensive early step cheaper, it deletes a cheap late step entirely. Round 10
established that run length is not itself the damage mechanism (23 reuses in
runs of 3 versus 20 in runs of 1 moved case 03 by 3e-5), and rounds 15-17
established that late-region reuse is nearly free. Expected 45.3 / 52.6 / 52.6 s,
4.09x / 3.48x / 3.47x, LPIPS somewhere near 0.21 / 0.20 / 0.08 for a reward
around **3.1**. Everything else is held at round 18's configuration.

Result: **reward 2.9076 -- a new best, but only just. Job 19026783 COMPLETED.**
LPIPS 0.2015 / 0.2156 / 0.0800, mean 0.1657 (limit 0.25), worst 0.2156 (limit
0.30). Latencies 47.79 / 55.32 / 55.36 s at **3.876x / 3.309x / 3.295x**.

The cap change removed exactly one computed step, not two -- with the late
region starting at 13 and a cap of nine the recomputes land on 21, 31 and 41
instead of 18, 25, 32 and 39 -- so 2.1 s, +4.4% speed for -4.0% quality and
+0.37% reward. **The exchange rate on this lever has reached 1:1 and the lever
is finished.** Rounds 15, 16, 17, 20 traded quality for speed at 1:0.5, 1:0.25,
1:0.55 and now 1:0.9; there is nothing left in the late region worth buying.

Case 02's fifteen computed steps and what they cost:

    warmup     0 1 2 3 4 5        FlashAttention   2.91 s   17.5 s
    threshold  7 9 11             FlashAttention   2.91 s    8.7 s
    run        21 31 41           calibrated cuDNN 2.28 s    6.8 s
    tail       46 47 48           calibrated cuDNN 2.28 s    6.8 s
    34 reuses                                      0.05 s    1.7 s

## Round 21 -- is the kernel's sensitivity spread through the stack?

Hypothesis. Nine of those fifteen forwards run FlashAttention, and they do so
because round 13 measured that swapping in the 1.72x-faster calibrated kernel
over the whole schedule takes case 02 from 0.0951 to 0.3030 LPIPS, and round 14
measured that restricting it to steps >= 12 recovers 0.116 of that. The
protection costs 9 x 0.63 = **5.7 s of a 47.8 s request**, which is now the
single largest remaining item in the budget -- larger than the decode, larger
than either staging phase.

`_PLAN_MIN_STEP` treats a denoising step as indivisible: on a protected step all
fifty blocks run the slow kernel. Nothing forces that. The question is whether
the sensitivity the protection buys is spread evenly down the stack or
concentrated near its input. So a second gate, `_PLAN_MIN_LAYER = 25`: on the
nine protected steps, blocks 0-24 keep FlashAttention and blocks 25-49 run the
calibrated kernel anyway. Cost 9 x 0.63 / 2 = **2.8 s**, to 45.0 s and 4.12x.

This deliberately re-asks round 19's question about a different quantity, and
the two should not be assumed to answer alike. Round 19 reused a *stale* deep
tail and the damage was enormous, because the argument a deep segment is
evaluated at drifts further between steps than block 0's output does -- that is
a statement about propagation. A kernel difference is *injected* at the layer it
happens at and has fewer remaining layers to be amplified through the deeper it
sits, which argues the other way. If the two effects really do point in opposite
directions, this is 2.8 s at a fraction of round 13's cost; if depth is simply
uniform, case 02 lands near 0.29 and this is the last of the quality budget.
Either answer sets the value of `_PLAN_MIN_LAYER` for the remaining rounds.
Everything else is held at round 20's configuration.

Result: **gate failed, reward 0. Job 19027900 COMPLETED in 21:09.** LPIPS
0.3333 / 0.2764 / 0.0803, mean 0.2300 (limit 0.25), worst 0.3333 (limit 0.30).
Raw reward before the gate 2.820. Latencies 45.71 / 52.21 / 52.10 s at 4.053x /
3.507x / 3.501x -- the 2.08 s arrived as predicted.

The answer is the opposite of the one the round argued for, and it is the same
answer round 19 gave. Case 02 went 0.2015 -> 0.3333, **+0.132**. Rounds 13 and
14 bracket what the whole stack does over these steps at +0.187. So half the
layers -- the deeper half -- carry **71% of the damage the whole stack does**.
Depth is not protective against an injected kernel difference either; it is
worse than proportional.

Two independent mechanisms now point the same way. Reuse a deep segment's
output (round 19) and the damage is four times what reusing the whole stack
costs. Inject a 2.9e-4 kernel difference into the deep half (round 21) and it
carries 71% of what injecting it everywhere costs. The shared explanation is
that the deep blocks are where this model's output is actually determined: a
perturbation there has few layers left to be averaged against but every
remaining one to be amplified through, and the residual stream's later
contributions are the ones the output head reads most directly. Whatever is
done to this model cheaply has to be done at the front of the stack or not at
all -- and the front of the stack is where the cache already lives.

That closes the approximation space at this operating point. The late-region
reuse lever reached a 1:1 exchange rate in round 20; the kernel lever is worse
than 1:1 in both the step and the depth direction; partial recompute is worse
still. What is left is exact work. Reverted; best remains round 20 at 2.9076.

## Round 22 -- stage the transformer behind the forward that needs it

Hypothesis. Case 02's 47.79 s contains 1.85 s in which nothing happens except
37.49 GiB crossing PCIe, plus 2.54 s of the same for the conditioner. The
conditioner's is irreducible -- 48.88 GiB for a 42 ms forward, bandwidth-bound
end to end. The transformer's is not, because **the consumer does not need all
of it at once.** It needs block 0, then block 1, and so on, and one block is
0.75 GiB: 37 ms of link time against 58 ms of compute for the forward that
block is about to run. Staging, once started, outruns execution on its own.

So the transformer is staged in execution order, from a background thread, with
one event per block; the forward waits on block `i`'s event immediately before
running block `i`. Everything that is not a block -- the two patch projections,
the context embedder and token refiner, the time embedder, the output norm and
the two heads -- is group 0 and is staged synchronously, because the forward
touches all of it before block 0 and after block 49. The expected wait is one
block, ~40 ms, against 1.85 s today: **1.8 s per request, on all three cases,
at exactly zero quality cost.**

Three things this has to get right, each of which has already cost this campaign
a round. The staging thread must not call `torch.cuda.synchronize`, which would
wait on the main thread's queued compute -- the very compute the staging is
hiding behind -- so `_Stager.copy` grows a `device_sync` flag and the pipelined
path passes False; the per-chunk `stream.synchronize()` each worker already does
is what makes the bytes visible. The threaded stager must not be used for a
residency's *cold* first staging, where round 8 measured four threads at 201 s
against one sequential reader's 34.8 -- so the untimed warmup still takes the
blocking path. And a worker's exception must not escape, because round 7 died of
an OOM caused by a traceback holding a reference cycle back to 48 GiB of device
tensors; the thread records the failure, releases every event, and the next
`gate()` raises it on the main thread.

Which leaves one real risk: the pipelined path would otherwise run for the first
time on a *timed* request, where a chunking or ordering bug shows up as a ruined
LPIPS and costs the round rather than explaining itself. So the warmup stages
the transformer a second time, through the pipelined path, joined immediately
and byte-verified against the host masters -- everything the timed path does
except the overlap. The gate's own logic (group 0 is not a block, block `i` is
group `i+1`, a failure raises rather than letting the forward run on
`torch.empty`) is checked on the login node in `tools/check_gate.py`.

Expected 46.0 / 53.5 / 53.5 s at unchanged LPIPS, reward ~**3.02**.

Result: **FAILED, reward 0. Job 19029171, srun exit 1 after 5:05.** The warmup
generation died of an out-of-memory and took case 02 with it; cases 03 and 04
completed. Two separate things came out of this round and they have to be kept
apart.

**The bug.** `stage_async` builds a `plan` list of `(host, device)` pairs and
the warmup's verification pass kept it alive. That list holds a hard reference
to every staged device tensor -- including the 24.23 GiB of AdaLN projections
that `precompute_adaln` replaces and frees a few lines later. The free became a
no-op, peak went from 72.16 to 78.42 GiB, and the transformer profile and then
the warmup itself ran out of memory. Fixed: the plan is dropped as soon as it
has been verified. The verification pass was added precisely because a
first-use-on-a-timed-request bug would cost a round -- it cost one anyway, by
being the bug. It did its job: `H3_STAGE_VERIFY transformer checked=8
mismatched=0` says the pipelined staging moves the right bytes.

**The measurement, which is clean and is a negative result.** Requests 1 and 2
ran the pipelined path end to end:

    transformer_stage  1.85 s -> 0.198 / 0.212 s
    stage_wait (inside denoise)       1.681 / 1.868 s

The phase moved and the wait moved with it. The bytes are not arriving any
faster than the consumer needs them: concurrent with the denoise loop the link
delivers roughly 8 GiB/s against the 20 GiB/s it manages when the main thread is
idle, so 37.49 GiB take about 4.7 s while one forward offers only 2.9 s to hide
them in. **The break-even rate for this overlap is 37.49 / (1.85 + 2.9) = 7.9
GiB/s and this machine sits exactly on it.** Nothing about bundling or thread
count changes that; the constraint is that the only compute available to hide
the staging behind is the single forward that needs the weights, and the copy
engine does not get a free ride while the SMs are saturated. Reverted to
blocking, the path kept behind `H3_STAGE_PIPELINE=0` because it is correct
rather than because it is faster. Best remains round 20 at 2.9076.

That also closes the exact-systems space: the conditioner's 2.5 s is
bandwidth-bound for a 42 ms forward, the transformer's 1.85 s cannot be
overlapped, and the decode has been compiled and profiled to a GEMM-and-
attention floor.

## Round 23 -- the one guard that has never been measured

Hypothesis. `tail_steps` has been 3 since round 4 and has never been varied. It
forces the last three denoising steps to be computed whole, and on case 02 those
are three calibrated-cuDNN forwards at 2.28 s -- **6.8 s of a 47.8 s request**
spent on a guard that was a guess.

It is the one remaining knob with a large allowance. The reward is
`speedup x (1 - LPIPS)`, so a change is worth making when the fractional speed
gain exceeds `dLPIPS / (1 - LPIPS)`. Dropping `tail_steps` to 1 removes two of
those three forwards, 4.56 s, **9.5%** -- which pays for up to **+0.075 LPIPS**
before it is a loss, and the gate allows +0.084 before it is a zero. Every
lever measured since round 15 has been fought over one to four percent; this one
has six times the room.

Against it: these are the steps that write the final detail, and LPIPS is a
perceptual metric with an appetite for texture. For it: round 12 measured
reusing steps 36-45 at +0.0007 LPIPS, rounds 15-17 moved the whole back half of
the schedule into free reuse for very little, and round 10 established that the
length of a reuse run is not itself the damage mechanism -- which is what a
six-step run ending at step 47 is asking to be true.

Case 02's computed steps go from fifteen to thirteen: 0-5, 7, 9, 11, 21, 31, 41
and 48. Expected 43.2 / 50.8 / 50.8 s, 4.29x / 3.60x / 3.59x. If LPIPS holds
anywhere under 0.27 this is a reward near **3.1**; if the tail turns out to be
where the detail lives, the trace says so and round 24 sets `tail_steps` to 2.
The round-22 staging path is reverted to blocking, so this measures the guard
and nothing else.

Result: **reward 3.0452 -- a new best, and the largest single gain since round
16. Job 19029734 COMPLETED in 16:14.** LPIPS 0.2202 / 0.2447 / 0.1173, mean
0.1941 (limit 0.25), worst 0.2447 (limit 0.30). Latencies 43.99 / 51.28 / 51.03
s at **4.211x / 3.570x / 3.575x**.

The guard was over-conservative by exactly the amount the arithmetic said it
could afford to be. Speed +8.6%, quality -3.5%, reward **+4.7%** -- against a
lever that had been returning 1:1 three rounds ago. Case 02's computed steps are
now thirteen:

    0 1 2 3 4 5 | 7 9 11 | 21 31 41 | 48

and the two forwards that came out, steps 46 and 47, cost **+0.0187 LPIPS
between them -- 0.009 each**. That is the cheapest per-forward price measured
anywhere in this campaign, twenty times cheaper than the +0.154 a single reused
step 4 costs and cheaper than any late-region step measured since round 15.

The lesson is not about the tail specifically. `tail_steps = 3` was written in
round 4 as a guess and then carried unexamined through nineteen rounds while
every neighbouring knob was tuned to exhaustion -- including four rounds spent
grinding `late_fraction` and `late_max_run` down to a 1:1 exchange rate. An
untested default is not a safe default; it is an unmeasured one, and this one
was holding 6.8 s.

## Round 24 -- the last forward

Hypothesis. `tail_steps` is now 1, which forces step 48 -- the final denoising
step -- to be computed whole, one calibrated-cuDNN forward at 2.28 s, **5.2% of
case 02's request**. Setting it to 0 removes the guard entirely and lets step 48
be reconstructed like any other late step.

The estimate comes from this round's own measurement rather than from the
general principle: steps 46 and 47 cost 0.009 LPIPS each, and step 48 is their
immediate neighbour. At that price 5.2% of speed costs 1.1% of quality, for
about **+4%** of reward. Break-even is +0.041 LPIPS and the gate is +0.055 away
on case 03, so there is four times the room the estimate needs.

The case against is that step 48 is not merely a late step, it is the *last*
one: its velocity prediction is the one the scheduler integrates into the
latents that are handed to the VAE, with nothing after it to correct a mistake.
Set against that, the reuse is not a skipped step -- the scheduler still
integrates, on a first block that is genuinely evaluated plus a tail residual
from step 41 -- and round 12 measured reusing steps 36-45 at +0.0007 LPIPS
total. If the last step really is special, this is the round that says so, and
it says so with a number rather than an intuition. Everything else is held at
round 23's configuration.

Result: **reward 3.1223 -- a new best. Job 19030445 COMPLETED in 7:25.** LPIPS
0.2374 / 0.2661 / 0.1304, mean 0.2113 (limit 0.25), worst 0.2661 (limit 0.30).
Latencies 41.82 / 49.06 / 48.81 s at **4.431x / 3.731x / 3.737x**.

The last step is not special, it is merely last. Removing it cost +0.0172 on
case 02 against the +0.009 its two neighbours cost, so it is about twice as
expensive as step 47 and still seven times cheaper than the cheapest thing the
late region was selling three rounds ago. Speed +5.2%, quality -2.2%, reward
+2.5%. Case 02 now computes twelve of forty-nine forwards.

## Round 25 -- the threshold, and where the other two cases actually spend

Hypothesis. Every round since 15 has been tuned by watching case 02, and case
02 is not the case that is losing. Reading all three traces side by side:

    case 02   warmup 0-5 | threshold 7 9 11        | run 21 31 41      12 computes
    case 03   warmup 0-5 | threshold 6 7 8 9 10 12 | run 22 32 42      15 computes
    case 04   warmup 0-5 | threshold 6 7 8 9 10 12 | run 22 32 42      15 computes

Cases 03 and 04 spend **six** forwards in the metered region 6-12 where case 02
spends three, because their first-block residual drifts faster there -- 0.046 to
0.037 against case 02's 0.032 to 0.023 -- and a threshold of 0.036 admits almost
none of it. That is 8.7 s per request that case 02 never pays, and it is why
case 03 sits at 3.73x while case 02 sits at 4.43x.

Raising the threshold to 0.044 selects the two lowest-drift steps of that region
on both cases -- steps 8 (0.0415-0.0425) and 9 (0.0379-0.0381) -- which then run
into the metered-region cap of two and force a recompute at step 10. Cases 03
and 04 lose two dense FlashAttention forwards each, 5.8 s, **13%**. Case 02 is
unchanged: at 0.044 its steps 7, 9 and 11 still exceed the threshold and it
still computes three.

The gate, not the exchange rate, is what this has to clear. Break-even needs
dLPIPS < 0.095 on case 03; the gate allows 0.034 before case 03's worst case
reaches 0.30, and the mean has room for three times that. So the question is
narrow and specific: **does reusing steps 8 and 9 cost more than 0.034?**

There is evidence both ways and it is worth being honest that it does not
settle. Against: round 12 measured a single added reuse at step 4 at +0.154, and
round 5 put case 03 at 0.385 with a threshold of 0.055. For: round 5 had neither
the warmup guard nor the late-region structure and was reusing steps 1-5 as
well, which the current configuration forbids outright; steps 8 and 9 sit three
to five steps further down a curve that reaches +0.0001 per step by step 36; and
these are the two lowest-drift steps in the region, which is exactly the
selection the metric is for. Twenty-four rounds have established that damage is
decided by *which* steps are approximated, and this round asks the question at
the last place on the schedule where the answer is still unknown.

If it fails, the campaign ends on round 24's 3.1223 and the submitted state is
reverted to that commit; a failed twenty-fifth round costs a measurement that
has already been banked. If it holds, cases 03 and 04 gain 13% each for about
2% of quality and the reward lands near **3.31**.

Result: **gate failed, reward 0. Job 19031331 COMPLETED in 7:17.** LPIPS
0.2374 / 0.3328 / 0.1520, mean 0.2407 (limit 0.25), worst 0.3328 (limit 0.30).
Raw reward before the gate **3.2169** -- which would have been the campaign's
best by 3%.

The speed arrived exactly as designed, including the parts that were meant not
to move. Case 02 came out at 41.810 s against round 24's 41.815 and LPIPS
0.237415 against 0.237414: the threshold change genuinely did not touch it, and
that control makes the rest of the round readable. Cases 03 and 04 gained 13%
each, 49.06 -> 43.96 and 48.81 -> 44.29, 3.73x -> 4.16x and 3.74x -> 4.12x.

The two reused steps cost:

    case 03   0.2661 -> 0.3328   +0.0667     gate failed by 0.033
    case 04   0.1304 -> 0.1520   +0.0215     comfortable

Twice the budget on case 03 and three times cheaper than that on case 04 -- for
the same two steps, selected by the same drift values (8: 0.0425 vs 0.0415, 9:
0.0381 vs 0.0379) on traces that are identical to within 2%. That is the fifth
independent confirmation in this campaign that **no in-flight signal orders the
cases by how much an approximation will cost them.** The relative-L1 trace, its
shape, the tail-residual magnitude, the per-case `hold` prediction error and now
a matched pair of drift values have all failed at it. The operating point has to
be global and it has to be set by the case that suffers most, which is why the
threshold stays at 0.036: at 0.044 it is buying case 04 a bargain and case 03 a
gate failure, and there is no way from inside the model to tell them apart.

Reverted to round 24's configuration. **The campaign ends at reward 3.1223.**

---

# Final

Twenty-five rounds started, twenty-five finished, **7.83 H100-hours actually
consumed** (28,188 s by `sacct`) against an 11.67-hour reservation inside the
12-hour budget. Best: **round 24, reward 3.1223**, 4.431x / 3.731x / 3.737x
at LPIPS 0.2374 / 0.2661 / 0.1304, mean 0.2113 and worst 0.2661 against limits
of 0.25 and 0.30. See `FINAL.md`.
