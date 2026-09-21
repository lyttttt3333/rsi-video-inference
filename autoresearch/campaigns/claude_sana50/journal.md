# Claude Opus SANA 50-round journal

Independent baseline: GPU-resident SANA-Video 2.0. All eight prompts are visible
test cases. Gate: mean LPIPS <= 0.30 and worst LPIPS <= 0.50.

Reward per case = `baseline_seconds / candidate_seconds * (1 - LPIPS)`; the round
reward is the mean over the eight cases, zeroed if the split gate fails.

## Fixed facts about the task (measured, round 1)

- Shapes are identical for all eight cases: 544x960, 121 frames, 50 steps.
  Latent `(1, 128, 16, 17, 30)` -> 8160 tokens; CFG makes every denoiser call a
  batch of 2. NFE = 50 (confirmed by counting denoiser calls).
- Hardware/stack: H100 80GB HBM3, torch 2.9.1+cu128. Build (weight load) costs
  215 s and is excluded from latency but not from the 14-minute reservation.
  Round 1 used 536 s of 840 s wall, so ~300 s is available for compilation.
- The pipeline is bit-deterministic: round 1 re-ran the unmodified baseline and
  every case scored LPIPS exactly 0.0. Any non-zero LPIPS later is caused only
  by my changes, which makes the metric a clean divergence signal.

## Round 1 - instrumented baseline replication

Hypothesis: re-running the baseline with phase timers and warmup-only kernel
profiling reproduces reward ~1.0 and localizes the cost.

Result: reward **1.00687**, mean LPIPS 0.0, worst 0.0, gate passed.
Per case speedup 1.0057-1.0099, candidate median 26.218-26.242 s.
Artifacts: `runs/round-01/`. Source commit `49af5f5`.

Phase split of a timed 26.23 s generation (CUDA events):

| phase | time | share |
| --- | --- | --- |
| text encoder (Gemma, 2x300 tokens) | 0.050 s | 0.2% |
| denoise loop (50 NFE, batch 2) | 24.97 s | 95.2% |
| LTX VAE decode | 1.212 s | 4.6% |

Denoiser kernel profile (3 NFEs captured during the excluded warmup,
503 ms/NFE):

| class | ms/NFE | share | note |
| --- | --- | --- | --- |
| GEMM (`nvjet_*`) | 214 | 42.5% | 1.31e14 FLOP/NFE -> 612 TFLOP/s, ~62% of peak |
| attention (flash + cutlass mem-eff) | 54 | 10.7% | 8 dense anchor layers + 32 cross-attn |
| elementwise / reduce / gemv / memcpy | 255 | 50.7% | pure glue |

Conclusions that set the campaign plan:

1. The denoiser is 95% of the cost; the VAE and text encoder are rounding
   errors until the denoiser gets much faster.
2. The GEMMs already run at ~62% of H100 bf16 peak, so there is little to win
   from better GEMM tiling. Half the denoiser is memory-bound glue instead:
   - fp64 complex RoPE (`view_as_complex` on a float64 upcast of q and k in all
     32 layers) - ~1.2 GB of traffic per rope call;
   - `RMSNorm` materializing an fp32 copy of every activation;
   - `BlockAttentionResidual.attend_buffer`, called 64x per NFE, which runs a
     `gemv` over the whole key buffer (33 ms/NFE) and a `bmm`-shaped weighted
     sum over the whole value buffer (46 ms/NFE at only ~400 GB/s).
3. Removing all glue would give 503 -> 268 ms/NFE, i.e. a 1.79x lossless
   ceiling. Beyond that the levers are FP8 GEMMs and NFE reduction.
4. The VAE decode is dominated by `aten::slow_conv_dilated3d` (80% of its CUDA
   time, with `vol2col` alone at 53%): the 3D convolutions are falling back to
   an im2col path instead of cuDNN. Worth ~0.8 s once the denoiser is small.

## Round 2 - lossless glue removal in the denoiser

Hypothesis: the denoiser's memory-bound half can be cut without changing the
mathematics: apply RoPE as a real float32 rotation against cached tables
instead of a float64 -> complex128 multiply, run gated linear attention
head-major so the three layout copies per layer become views, dispatch
last-axis `RMSNorm` to the fused `F.rms_norm` kernel, cache the depth
aggregation's completed-block logits, and build the caption embedding and every
layer's cross-attention K/V once per request instead of once per NFE.

Result: reward **1.13409**, speedup 1.181x (26.23 -> 22.34 s; denoise
24.97 -> 21.07 s = 422.6 ms/NFE), mean LPIPS 0.0396, worst 0.1038 (case 07),
gate passed. Artifacts: `runs/round-02/`, job 19008111, source commit `d3f35de`.

| case | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LPIPS | 0.0118 | 0.0445 | 0.0238 | 0.0198 | 0.0289 | 0.0562 | 0.1038 | 0.0282 |
| speedup | 1.181 | 1.183 | 1.185 | 1.178 | 1.181 | 1.181 | 1.177 | 1.181 |

The important measurement is not the speedup but the LPIPS. Every change in this
round is exact in real arithmetic and was verified against the round-1 source on
CPU in float32 to a relative deviation of 1.3e-6; the only difference on the
H100 is bf16 rounding and reduction order. That alone costs mean LPIPS 0.040 and
worst 0.104. So on this benchmark LPIPS measures trajectory divergence over 50
chained steps, not quality: there is a ~0.04 "chaos floor" that any reordering
pays, case 07 is the most sensitive case, and the real budget for deliberately
lossy methods is the distance from that floor to the 0.30 mean / 0.50 worst gate
rather than the whole gate.

## Round 3 - TorchInductor region compilation

Hypothesis: with the glue now expressed as ordinary elementwise work, compiling
the hot regions fuses it. Wrapped the attention sublayer, the MLP sublayer, the
depth-aggregation logits, and the partial-block weighted sum in
`torch.compile(dynamic=False)`, with `suppress_errors` so a region inductor
cannot lower degrades to eager instead of failing the round.

Result: reward **1.33391** (best so far), mean LPIPS 0.0589, worst 0.1638
(case 07), gate passed. Artifacts: `runs/round-03/`, job 19008746, source commit
`69f106d`. Compilation cost 15 s inside the excluded warmup generation; the job
used 244 s of its 840 s reservation.

| case | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LPIPS | 0.0156 | 0.0377 | 0.0263 | 0.0313 | 0.0213 | 0.0941 | 0.1638 | 0.0813 |
| speedup | 1.554 | 1.387 | 1.363 | 1.385 | 1.380 | 1.353 | 1.547 | 1.380 |
| seconds | 16.97 | 19.07 | 19.43 | 19.06 | 19.13 | 19.49 | 17.06 | 19.11 |

The per-case spread is the finding. Rounds 1 and 2 were uniform to +-0.1 s
across all eight cases; here two cases are 2.2 s faster than the other six. The
two fast cases are exactly case 01, whose shapes were compiled during the
excluded warmup, and case 07. Tokenizing the eight prompts with the Gemma
tokenizer gives valid caption lengths of 108, 115, 105, 110, 113, 116, 113, 117
- and 07's 113 is a repeat of 05's. So the fast cases are precisely the two
whose cross-attention K/V shape had already been compiled, and each new caption
length pays a ~2.2 s recompile *inside the timed region*. Round 2's per-row
prefix slicing, which is what made the K/V length prompt-dependent, is what
exposed the compiled regions to a prompt-dependent guard.


## Round 4 - prompt-independent compiled shapes

Hypothesis: round 3's spread is a recompile inside the timed region. Split the
attention sublayer into a compiled part that produces the self-attention delta
and the cross-attention query, an eager pair of scaled-dot-product calls over
the cached per-row K/V, and a compiled part that projects the heads and adds the
residual. Attention is an opaque kernel to inductor either way, so nothing that
could have been fused is lost, and no compiled kernel keeps a prompt-dependent
guard. Expect all eight cases at case 01's 17.0 s.

Result: reward **1.44286** (best so far), mean LPIPS 0.0640, worst 0.1943
(case 07), gate passed. Artifacts: `runs/round-04/`, job 19009667, source commit
`c63c1a6`.

| case | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LPIPS | 0.0176 | 0.0440 | 0.0270 | 0.0335 | 0.0387 | 0.0935 | 0.1943 | 0.0636 |
| speedup | 1.543 | 1.545 | 1.541 | 1.542 | 1.543 | 1.542 | 1.540 | 1.538 |
| seconds | 17.10 | 17.13 | 17.19 | 17.12 | 17.11 | 17.10 | 17.11 | 17.15 |

The spread is gone: 17.10-17.19 s across all eight cases, and the six cases that
were paying a recompile picked up the 2.2 s. That is the confirmation of the
round-3 diagnosis. Phase split now: text 0.055 s, denoise 15.83 s
(316.6 ms/NFE, down from the baseline's 503 ms/NFE), VAE decode 1.216 s. The VAE
is now 7% of the generation, and compilation costs 15.6 s inside the excluded
warmup call.

Note on verification: the CPU environment on the login node cannot lower
inductor graphs at all (its triton is broken, so every region degrades to eager
under `suppress_errors`). The CPU tests therefore verify Dynamo tracing, guard
structure, and algebraic equivalence against the round-1 source, but the
compiled kernels themselves are only ever exercised on the H100. An
`error_on_recompile` assertion added to the compile test cannot reproduce the
round-3 recompile offline for that reason; the eight uniform timings above are
what actually confirms it.

Where the remaining time is, per NFE: GEMMs were 214 ms/NFE at the baseline and
are untouched by every round so far, so they are now roughly two thirds of the
316 ms. The levers left are NFE reduction (lossy), FP8 GEMMs, CUDA graphs, and
the VAE's im2col 3D convolutions.

## Round 5 - cross-step velocity reuse, one call in three

Hypothesis: all 50 requested solver steps are still taken, but on 13 of them the
sampler integrates the previous denoiser output instead of evaluating the
network (one call in three over steps 6-45, with the first six and last four
always evaluated). The velocity is what is cached, not the data prediction, so
the solver still converts it against the current timestep. NFE 50 -> 37.

Result: reward **1.76267** (best so far), speedup 2.027x (17.10 -> 13.02 s;
denoise 15.83 -> 11.74 s), mean LPIPS 0.1303, worst 0.2391 (case 07), gate
passed. Artifacts: `runs/round-05/`, job 19010159, source commit `b953646`.

| case | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LPIPS | 0.0738 | 0.1715 | 0.1042 | 0.1380 | 0.0905 | 0.0911 | 0.2391 | 0.1340 |
| speedup | 2.027 | 2.031 | 2.034 | 2.024 | 2.023 | 2.023 | 2.024 | 2.028 |

The measurement I wanted: 13 dropped evaluations cost mean LPIPS
0.064 -> 0.130, about 0.005 per dropped NFE at this operating point, and worst
case 0.194 -> 0.239. Time per evaluated NFE is unchanged at 317 ms, so the
trade is linear in NFE on the speed side and (so far) close to linear on the
quality side, while reward = speedup x (1 - LPIPS) is convex in favour of
dropping more. With mean LPIPS 0.13 against a 0.30 gate, the operating point is
far from optimal - the next rounds should push the reuse rate up until the
quality curve turns.

Second consequence worth recording: the per-generation fixed cost is now
1.27 s (VAE decode 1.217 s plus text 0.055 s) against 11.74 s of denoising. At
30 NFE the VAE would be 12% of the generation and at 15 NFE it would be 20%, so
the VAE's im2col 3D convolutions get more valuable the more aggressive the
reuse becomes.

## Round 7 - one evaluation in three, plus chunked VAE convolutions

Two changes, kept separable by the per-phase CUDA-event timers. Denoise:
evaluate every third step over steps 6-45, 26 dropped evaluations, NFE 24. VAE:
run the causal 3D convolutions in temporal windows small enough that ATen keeps
them on cuDNN, since the reference decode spends 80% of its time in
`slow_conv_dilated3d` and ATen refuses cuDNN for convolutions that need 64-bit
indexing and cannot be split along a batch of one. The chunking is exact - each
output frame still sees the same input frames - and was verified bit-identical
on CPU across kernel sizes, temporal strides, and causal and centred padding.

Result: reward **2.38481** (best so far), speedup 2.957x (10.83 -> 8.92 s),
mean LPIPS 0.1936, worst 0.3757 (case 07), gate passed. Artifacts:
`runs/round-07/`, job 19010925, source commit `8e78b7a`.

| case | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LPIPS | 0.1082 | 0.2504 | 0.2024 | 0.2421 | 0.1105 | 0.1312 | 0.3757 | 0.1285 |
| speedup | 2.957 | 2.964 | 2.965 | 2.958 | 2.958 | 2.952 | 2.956 | 2.951 |

The denoise half worked exactly as predicted: 7.65 s at 24 NFE, still 319 ms per
evaluation. **The VAE half did nothing at all: decode stayed at 1217 ms against
round 4's 1216 ms.** So either the chunking never triggered, or the chunks still
do not reach cuDNN, or the decode's cost is not where the round-1 profile
suggested. Round 8 turns the VAE profile back on during the excluded warmup and
prints the size and chunk count of every causal convolution, which distinguishes
the three.

Updated reuse curve:

| dropped | NFE | seconds | speedup | mean LPIPS | worst LPIPS | reward |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 50 | 17.10 | 1.543 | 0.064 | 0.194 | 1.443 |
| 13 | 37 | 13.02 | 2.027 | 0.130 | 0.239 | 1.763 |
| 20 | 30 | 10.83 | 2.438 | 0.155 | 0.300 | 2.060 |
| 26 | 24 | 8.92 | 2.957 | 0.194 | 0.376 | 2.385 |

The worst case is accelerating - 0.0035, then 0.0087, then 0.0126 per dropped
evaluation - and case 07 now sits at 0.376 against the 0.50 gate. At that rate
there are only about six more drops available, so the next lever has to be a
cheaper reuse rule rather than a higher reuse rate.

## Round 8 - extrapolate the reused velocity

Hypothesis: holding the velocity constant on a reused step hands the multistep
solver two identical history entries, which collapses its second-order
correction to first order exactly where the trajectory is least resolved.
Extrapolating along the line through the two most recently *evaluated*
velocities, using the real timestep spacing as the ratio, restores a slope for
the solver at the cost of one elementwise pass over a 2 MB tensor. Only
evaluated velocities enter the history, so error cannot compound through a run
of reused steps. Verified on CPU: a velocity field that is linear in the
timestep is reproduced exactly (error 0.0) by the reused steps, and with
extrapolation disabled the same schedule holds the previous value constant.
The schedule is unchanged from round 7, so this isolates the reuse rule.

Result: reward **2.23144**, speedup 2.939x (unchanged, as designed), mean LPIPS
0.2408, worst 0.4030 (case 07), gate passed. Artifacts: `runs/round-08/`, job
19011454, source commit `1c9eec5`. **This is a regression and is reverted.**

| case | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LPIPS constant (round 7) | 0.1082 | 0.2504 | 0.2024 | 0.2421 | 0.1105 | 0.1312 | 0.3757 | 0.1285 |
| LPIPS extrapolated | 0.0810 | 0.3069 | 0.2718 | 0.2947 | 0.1512 | 0.1694 | 0.4030 | 0.2485 |

Extrapolating the velocity is worse on seven of eight cases, mean 0.194 ->
0.241, at identical speed. The hypothesis was that restoring a slope for the
multistep solver would help; what the measurement says is that over a gap of
two or three steps the velocity field is not linear enough for the
extrapolation to pay, and the synthetic slope amplifies error rather than
cancelling it. Holding the last evaluated velocity constant - which biases the
solver toward a shorter effective step - is the more stable choice. Reverted to
constant reuse; the trial stays in Git as commit `1c9eec5` and behind
`SANA_EXTRAPOLATE`.

The round also carried the VAE diagnosis, and it refutes the round-7 premise
outright. The decode's largest causal convolution input is
`(1, 128, 123, 136, 184)`, 0.39e9 elements - nowhere near the 2^31 limit that
makes ATen refuse cuDNN. Every one of the 84 convolutions ran unchunked, and
every one of them still went to `aten::slow_conv_dilated3d` (80% of decode CUDA
time, `vol2col` alone 53%). So the im2col fallback has nothing to do with tensor
size, and the round-7 chunking was a fix for a problem that does not exist. It
costs nothing measurable (decode 1216 -> 1217 ms) and stays in as a guard, but
the real cause is still open.

The same profile shows a second, unrelated cost: 49,490 `aten::copy_` calls and
49,742 elementwise kernels, averaging 2.5-3 us each, for 272 ms of CUDA time and
562 ms of CPU time inside a 1216 ms decode. The decode is close to CPU-bound on
tiny launches. Round 9 groups the profile by input shape to find where those
50,000 launches come from.

## Round 9 - revert, and measure what to do next

Hypothesis: with extrapolation reverted the reward should return to round 7's
2.385, and the excluded warmup is free to answer three questions it costs
nothing to ask: what the cuDNN backend situation actually is, which spelling of
the decoder's dominant 3D convolution is fastest on this container, and how
fast the velocity field moves at each of the 50 steps. The warmup generation
runs the full unskipped trajectory and records the per-step velocity magnitude
and step-to-step change, which is the curve a reuse schedule should be shaped
around instead of the uniform one in use since round 5.

Result: reward **2.38625**, speedup 2.957x (8.92 s), mean LPIPS 0.1936, worst
0.3757 - per-case LPIPS identical to round 7 to four decimals, which also
confirms the pipeline is reproducible run to run. Artifacts: `runs/round-09/`,
job 19011792, source commit `31139a6`.

Three measurements came out of the excluded warmup.

**1. cuDNN is present and the decode is not using it.** `cudnn available=True
version=91002 enabled=True`. Benchmarking the decoder's dominant convolution,
`(1, 128, 123, 136, 184)` with a 128-channel 3x3x3 kernel in bf16:

| spelling | ms |
| --- | --- |
| `F.conv3d`, contiguous | 29.18 |
| `F.conv3d`, contiguous, `cudnn.benchmark=True` | 29.23 |
| `F.conv3d`, **channels_last_3d** | **4.64** |
| three `conv2d` calls over the temporal taps | 13.53 |
| single-frame `conv2d` x 121 frames (extrapolated) | 9.01 |

So the im2col fallback is a *memory-format* condition, not a size condition:
ATen hands a bf16 3D convolution to cuDNN only when input and weight are both
`channels_last_3d`, and otherwise runs `slow_conv_dilated3d`. That is 6.3x on
the convolutions, which are 80% of the 1216 ms decode.

**2. The 50,000 small launches are real and cost ~240 ms.** Grouped by input
shape the decode shows 49,490 `aten::copy_`, 49,248 `aten::fill_`, 98,856
`aten::select` and 50,500 `cudaLaunchKernel`, averaging 2-3 us of CUDA each, for
123 ms plus 118 ms of CUDA time and 557 ms of *CPU* time. The select:fill:copy
ratio of 2:1:1 is a Python loop writing tensor slices one at a time. Round 10
resolves the Python stacks behind those two operator names to name it.

**3. The velocity field's step-to-step change is U-shaped.** Relative change
between consecutively evaluated velocities over the unskipped 50-step
trajectory: 0.411 at step 1, 0.13-0.18 through step 4, 0.077 and 0.064 at steps
5-6, decaying to a floor of 0.028-0.031 around steps 29-40, then rising again -
0.0369 at 45, 0.0423 at 46, 0.0522 at 47, 0.0784 at 48, 0.1720 at 49. The
uniform schedule in use since round 5 protects the first six and last four
steps, which is roughly right but not shaped to this curve: steps 6-9 still move
at 0.06-0.09 while steps 22-45 sit near the floor. A schedule that spaces
evaluations at equal *accumulated* change, rather than equal index, should buy
quality at the same NFE.

## Round 10 - channels-last decode

Hypothesis: converting the VAE's weights and the decode input to
`channels_last_3d`, and forcing that layout at each causal convolution, moves
all 84 convolutions onto cuDNN. At the benchmark's 6.3x the convolution time
should fall from roughly 970 ms to 155 ms, taking the decode from 1216 ms to
somewhere near 400 ms and the generation from 8.92 s to about 8.1 s, which is
worth 9%. The denoiser is untouched, so the denoise timer and all eight LPIPS
values should be unchanged except for the decoder's own kernel-level rounding.
The warmup additionally resolves the Python stacks behind the decode's
`fill_`/`copy_` storm.

Result: reward **2.54103**, speedup 3.150x (8.35 s), mean LPIPS 0.1936, worst
0.3757 - every per-case LPIPS identical to rounds 7 and 9 to four decimals.
Artifacts: `runs/round-10/`, job 19012261, source commit `7fb8abb`.

The hypothesis held. The decode fell from 1216 ms to 696 ms, and the
convolutions inside it from ~970 ms to 204 ms, which is the 6.3x the warmup
benchmark predicted. That the LPIPS values did not move by a single digit in the
fourth decimal is the useful part: cuDNN's bf16 3D convolution and the im2col
fallback agree bit for bit on this decoder, so 520 ms was free. The layout
counter reported 84 convolutions each converting their input, confirming the
decoder does not preserve `channels_last_3d` between layers.

Where the 686 ms of decode CUDA time now sits:

| category | ms | calls |
| --- | --- | --- |
| elementwise (silu, add, norm, layout conversions) | 356 | 577 |
| `aten::cat` | 253 | 86 |
| `cudnn_convolution` | 204 | 84 |

The `cat` is the causal padding: every convolution prepends a repeat of its
first frame. In the default layout that concatenation is strided and its output
is contiguous, so it both costs a full copy and destroys the layout the previous
convolution produced - which is exactly why 84 conversions were needed. In
`channels_last_3d` the time axis is the outermost non-batch axis, so the same
concatenation is a contiguous block copy and the layout survives it.

## Round 11 - change-proportional reuse schedule, and a decode that keeps its layout

Two independent changes, one in each timer.

Hypothesis A (denoiser). The schedule in use since round 5 evaluates the first
six calls, then every third, then the last four. Round 9's trace measured what
that schedule is trading against: the velocity field's step-to-step change is
0.411 at step 1, sits at a floor of 0.028 between steps 29 and 40, and climbs
back to 0.172 at step 49. Uniform spacing therefore spends evaluations where the
trajectory is already still and skips 0.06-0.09-sized moves at steps 6-9. Spacing
the same 24 evaluations at equal *accumulated* change instead gives
`[0,1,2,3,4,6,8,10,12,14,16,18,20,22,25,28,31,34,37,40,43,46,48,49]`, which cuts
the worst change any reused call stands in for from 0.1451 to 0.0771 (-47%) and
the total from 1.540 to 1.316 (-15%). At identical NFE and identical speed, mean
and worst LPIPS should both fall. This isolates the shape of the schedule from
its length; if it works, the next round spends the gain by shortening it.

Hypothesis B (VAE). Converting each causal pad to `channels_last_3d` before the
concatenation should turn the 253 ms of `cat` into a fraction of that and drop
the 84 layout conversions out of the 356 ms of elementwise time, taking the
decode from 696 ms toward 400 ms. The decoder has been bit-identical under
kernel changes twice now, so LPIPS should not move; the `[conv] ... needed a
layout conversion` counter is the direct check, and it should print near zero.

Together: about 8.35 s -> 8.05 s at a lower LPIPS, so a reward near 2.65 if
hypothesis A buys nothing and more if it does.

Result: reward **2.69055**, speedup 3.256x (8.11 s), mean LPIPS 0.1735, worst
0.2784. Artifacts: `runs/round-11/`, job 19012713, source commit `2020304`.
Both hypotheses paid, and the two timers separate them cleanly.

| | round 10 | round 11 |
| --- | --- | --- |
| total | 8.35 s | 8.11 s |
| text | 0.053 s | 0.054 s |
| denoise (24 NFE) | 7.60 s | 7.65 s |
| VAE decode | 0.696 s | 0.390 s |
| mean LPIPS | 0.1936 | 0.1735 |
| worst LPIPS | 0.3757 | 0.2784 |

Hypothesis B, the decode: 696 ms -> 390 ms, and `84 convolution calls, 0 needed
a layout conversion` - the layout now survives the whole decoder. Concatenating
the causal padding along time in `channels_last_3d` is a contiguous block copy,
so the 253 ms of `cat` and the 84 conversions inside the 356 ms of elementwise
time both went away. The decode is 44% of what it was at round 9 and 4.8% of the
generation; there is not much left in it.

Hypothesis A, the schedule: at unchanged NFE and unchanged speed, mean LPIPS
fell 10% and worst-case LPIPS fell 26%, on all eight cases (the largest, case
07, went 0.376 -> 0.278; the smallest, case 05, 0.099 -> 0.077). The predicted
47% cut in worst carried change translated into a 26% cut in worst LPIPS, so the
offline indicator is directionally right and optimistic in magnitude, which is
what one would expect of a first-order proxy. The important consequence is
headroom: the worst case is now 0.278 against a 0.50 gate, where at round 10 it
was 0.376.

## Round 12 - spend the headroom, and measure what float8 would buy

Hypothesis A. Drop from 24 evaluations to 20 on the same equal-change rule. The
schedule becomes `[0,1,2,3,4,6,8,10,12,15,18,21,24,28,32,36,40,44,47,49]`, whose
worst carried change is 0.1042 - still below the 0.1451 that the uniform
24-evaluation schedule carried at round 10, where worst LPIPS was 0.376. So
worst LPIPS should land near 0.33 and mean near 0.21, both inside the gates.
Denoise should fall to 7.65 x 20/24 = 6.4 s and the generation to about 6.85 s,
which is 3.85x and a reward near 3.04.

Hypothesis B is a measurement, not a change: the round-10 profile puts 54% of
the denoise time in `aten::mm` (512 ms of CUDA per 3 NFE over 840 calls), and at
the dominant shape - 16320 x 2560 x 7680, since patch_size is (1,1,1) so the
16x17x30 latent is 8160 tokens and CFG doubles the rows - 1.007 ms works out to
640 TFLOPS, about 90% of this H100's dense bf16 ceiling. Nothing but a narrower
input type will move it. The warmup now benchmarks row-scaled
`torch._scaled_mm` in float8_e4m3 against `aten::mm` at the denoiser's four GEMM
shapes, including the cost of quantizing the activation on the fly and the
relative error of one GEMM. If float8 delivers near its nominal 2x, the GEMMs
are worth ~85 ms of every 319 ms NFE and round 13 implements it; if the
quantization pass eats the gain at these shapes, the measurement says so before
a round is spent on it.

Result: reward **3.17735**, speedup 3.864x (6.83 s), mean LPIPS 0.1777, worst
0.2844. New best. Artifacts: `runs/round-12/`, job 19012932, source commit
`693e12e`. Denoise 7.65 s -> 6.39 s, VAE 0.390 s, text 0.056 s.

Four fewer evaluations cost almost nothing: mean LPIPS 0.1735 -> 0.1777 and
worst 0.2784 -> 0.2844, for 19% more speed. That is a far better trade than the
uniform schedule ever offered - rounds 5 to 7 paid 0.025 of mean LPIPS for every
three or four evaluations dropped, and this cost 0.004 for four. The
worst-carried-change indicator predicted it: at 20 evaluations the equal-change
schedule carries 0.1042, still less than the 0.1451 the uniform 24-evaluation
schedule carried, and LPIPS tracked the indicator rather than the count. The
gates remain far off - 0.178 against 0.30 and 0.284 against 0.50.

The float8 benchmark, at the denoiser's four GEMM shapes with 16320 rows:

| shape (m x k x n) | bf16 mm | fp8 mm (row-scaled) | eager quantize |
| --- | --- | --- | --- |
| 16320 x 2560 x 7680 | 0.818 ms (784 TF) | 0.647 ms (991 TF) | 0.581 ms |
| 16320 x 2560 x 10240 | 1.117 ms (766 TF) | 0.794 ms (1078 TF) | 0.582 ms |
| 16320 x 10240 x 2560 | 1.064 ms (804 TF) | 0.707 ms (1210 TF) | 2.210 ms |
| 16320 x 2560 x 2560 | 0.291 ms (736 TF) | 0.217 ms (986 TF) | 0.583 ms |

Two things. First, bf16 `mm` is running at 736-804 TFLOPS, which is 75-81% of
this H100's 989 TFLOP dense ceiling, so the GEMMs are genuinely compute-bound
and only a narrower type can help. Second, float8 gives 1.26-1.50x on the GEMM
itself, not the nominal 2x - the row-wise scaling epilogue costs real time - and
one GEMM's relative error is 0.0376, which is an order of magnitude more
perturbation than the bf16 reordering that cost 0.04 LPIPS in round 2. The
quantization column is not the real cost: that spelling upcasts to float32 and
makes five passes over the activation, which is why the k=10240 row takes
2.2 ms. A fused single pass moves 124 MB and should cost about 45 us.

So float8 is worth roughly 44 ms of the 319 ms NFE before quantization, 30 ms
after - about 10% - at a quality cost that is not yet known and looks
significant. Dropping four more evaluations is worth 19% at a cost of 0.004.
The schedule is the better buy and is taken first.

## Round 13 - sixteen evaluations

Hypothesis. Drop to 16 evaluations on the same rule:
`[0,1,3,5,8,11,14,17,20,24,29,34,39,44,48,49]`, worst carried change 0.1381.
That is still within 5% of what the uniform 24-evaluation schedule carried at
round 10, where worst LPIPS was 0.376, so worst LPIPS should land near 0.33 and
mean near 0.21 - inside both gates. Denoise should fall to 6.39 x 16/20 = 5.11 s
and the generation to about 5.56 s, which is 4.74x and a reward near 3.75.

Two free measurements ride along in the excluded warmup. The denoiser profile is
now also grouped by input shape, to name the `gemv2N` kernel that takes 484 us a
call, 62 calls an NFE, 9% of the denoise time - a matrix-vector product that
large has no obvious home in this architecture. And the float8 benchmark adds
the fused quantization cost and the tensor-wise scaled kernel, so the next
round's decision rests on the cost a real implementation would pay rather than
on an eager spelling's five passes.

Result: **quality gate failed**, so the round scores zero. Mean LPIPS 0.3050
against the 0.30 cap; worst 0.4278 against 0.50. Speed was 4.795x (5.51 s,
denoise 5.05 s) and the raw reward would have been 3.33218. Artifacts:
`runs/round-13/`, job 19013144, source commit `fcf8472`.

This is the cliff. Four fewer evaluations, 20 -> 16, took mean LPIPS from 0.1777
to 0.3050 - a 72% rise - where the previous four, 24 -> 20, cost 2.4%. The
worst-carried-change indicator moved only 0.1042 -> 0.1381 over that step, so it
is a good *ordering* of schedules at a fixed count and a bad *predictor* of
quality across counts: past a carried change of about 0.11 the reconstruction
stops being a small perturbation of the trajectory. Every case degraded, and the
two that were already worst (02 and 07) degraded most, 0.184 -> 0.382 and
0.284 -> 0.428.

Reverted to 20 evaluations. The schedule axis is finished at 20; anything below
it needs the reused steps themselves to get cheaper in quality, not more
frequent, and round 8 already showed that linear extrapolation is not that.

The float8 benchmark's second pass answers what round 12 left open:

| shape | bf16 mm | fp8 row-scaled | fp8 tensor-scaled | fused quantize |
| --- | --- | --- | --- | --- |
| 16320 x 2560 x 7680 | 0.812 | 0.653 | 0.469 | 0.133 |
| 16320 x 2560 x 10240 | 1.106 | 0.801 | 0.622 | 0.133 |
| 16320 x 10240 x 2560 | 1.055 | 0.718 | 0.595 | 0.260 |
| 16320 x 2560 x 2560 | 0.287 | 0.217 | 0.176 | 0.132 |

Tensor-wise scaling is 1.73x on the dominant GEMM where row-wise is only 1.24x -
the row-wise epilogue costs more than the accuracy it buys, and on this data
both spellings have the same 0.0376 relative error. The fused quantization costs
0.133 ms, a quarter of what the eager five-pass spelling does. Summing a
transformer block at these shapes: bf16 4.65 ms, float8 2.66 ms of GEMM plus
about 0.66 ms of quantization, so 1.33 ms saved per block per NFE, 43 ms of the
319 ms NFE, 13% end to end. Worth having, but the per-GEMM error is an order of
magnitude larger than the bf16 reordering that cost 0.04 LPIPS in round 2, and
after round 13 the quality budget is no longer generous. It goes after the
cheaper structural win below.

## Round 14 - halve the batch by caching the guidance direction

Classifier-free guidance runs the denoiser on two rows, unconditional and
conditional, and the sampler combines them as `u + s (c - u)`. Writing
`d = c - u` that is exactly `c + (s - 1) d`, so the unconditional row is needed
only through `d`. And `d` should be the slow part: the negative prompt is fixed
for every request, and the guidance direction is a property of the prompt pair
rather than of the point on the trajectory.

Hypothesis. On 7 of the 20 evaluated steps, run the network on the conditional
row alone and reconstruct the unconditional one as `c - d` from the most recent
full pair. The first four evaluations and the last one always run the pair, and
no cached `d` is ever older than one evaluation. That halves the row count of
every GEMM and every attention call on those steps, and at these shapes the
GEMMs are compute-bound, so a single-row step should cost about 55% of a paired
one: 7 x 0.45 x 319 ms = 1.0 s off a 6.39 s denoise, for about 4.5x and a reward
near 3.6 if quality holds.

The error this introduces is `(s - 1)` times the drift in `d` over one
evaluation, which is why the same warmup trace that measured the velocity curve
now measures `d` and its step-to-step change as well - so that if the round
works, the next one can shape the pairing schedule against that curve instead of
a fixed period, and if it does not, the trace says whether `d` was the wrong
thing to cache or merely cached too rarely.

Both the reconstruction and the plan are covered by a new offline test: the
guided combination must equal `c + (s - 1) d` exactly on every conditional-only
step, the conditional state must be the second row of the cached condition, and
the warmup must run exactly one single-row forward so that the batch-1 graphs
are compiled off the timed path - the round-3 failure was a recompilation inside
the timed region, and this change introduces a second shape for every compiled
region in the denoiser.

Result: **quality gate failed**, so the round scores zero. Mean LPIPS 0.3019
against the 0.30 cap, worst 0.4322. Speed 4.626x (5.70 s, denoise 5.25 s), raw
reward would have been 3.22903. Artifacts: `runs/round-14/`, job 19013341,
source commit `10f1dc3`.

The speed prediction was right - 7 conditional-only steps took the denoise from
6.39 s to 5.25 s, 1.14 s against 1.0 s predicted, so a single-row step does cost
about half a paired one. The premise was wrong, and the trace that rode along
says exactly how wrong. Relative step-to-step change, velocity against guidance
direction:

| step | velocity | guidance direction d | \|d\| |
| --- | --- | --- | --- |
| 5 | 0.077 | 0.474 | 0.237 |
| 15 | 0.048 | 0.319 | 0.191 |
| 28 | 0.033 | 0.300 | 0.108 |
| 40 | 0.029 | 0.475 | 0.054 |
| 49 | 0.172 | 0.959 | 0.040 |

`d = c - u` is the *fastest* moving quantity on the trajectory, not the slowest:
its relative change is 0.26-0.96 per step where the velocity's is 0.03-0.05. In
hindsight that is what a small difference between two large, nearly equal
vectors has to look like - the common part cancels and what remains is the part
that moves. And the sampler multiplies it by `s - 1 = 7`, so one step of
staleness perturbs the velocity by `7 x 0.30 x 0.108 / 1.39 = 16%` at step 28,
against the 3-10% that dropping a step outright costs. Halving the batch is
strictly dominated by skipping the step. `CFG_PERIOD` now ships at 0; the code
and its test stay, because the measurement is the useful part.

Two failed rounds have also mapped the shape of the trade. Reward is
`speedup x (1 - LPIPS)`, and along the evaluation-count axis the two move
together almost exactly:

| evaluations | total carried change | mean LPIPS | speedup | reward |
| --- | --- | --- | --- | --- |
| 24 | 1.540 | 0.1735 | 3.256 | 2.69 |
| 20 | 1.917 | 0.1777 | 3.864 | 3.18 |
| 18 | 2.398 | 0.246 (fitted) | 4.26 | 3.21 (predicted) |
| 16 | 2.820 | 0.3050 | 4.795 | 3.33 but **gated to zero** |

Between 24 and 20 the trade was 19% of speed for 0.004 of LPIPS. Past 20 it is
roughly reward-neutral and gate-risky: fitting mean LPIPS linearly against total
carried change - which reproduces both measured points - 18 evaluations predicts
a reward of 3.21 against the 3.18 already banked, for a 0.05 margin on a gate
that has now been missed twice. The lossy axis is finished. What is left is
making a step cheaper at unchanged output, where every 1% of speed is worth
0.032 of reward outright.

## Round 15 - let inductor benchmark the GEMMs it has been handing to cuBLAS

Revert to the round-12 configuration, and change one thing about how it is
compiled. Half the denoise time, 170 ms of every 319 ms NFE, is `aten::mm`:
inductor leaves matrix multiplies as extern cuBLAS calls, so the AdaLN
modulation, the gating and the residual adds around them stay as separate
passes over the same tensors - 23 ms an NFE of Triton pointwise kernels that
read and write what the GEMM just wrote. Turning on `max_autotune_gemm` with
both backends lets inductor benchmark Triton GEMM templates against cuBLAS at
each shape and keep whichever is faster, and a template that wins can absorb its
epilogue. cuBLAS stays in the candidate set, so the GEMM itself cannot get
slower than the round-12 build; the autotuning runs at compile time inside the
excluded warmup.

Expect somewhere between 0% and 8%, with the honest possibility that cuBLAS
wins every shape and nothing changes but the compile time. LPIPS may drift by a
few thousandths if a template changes the reduction order - round 4's
compilation cost 0.005 - but no more than that. The risk being taken is compile
time: autotuning is minutes, and the harness allows 780 s for a warmup and eight
generations that currently take 100 s.

The by-shape profile, which was sorted by CPU time and therefore never showed
what it was added for, is now sorted by CUDA time. It should finally name the
`gemv2N` kernel that takes 486 us a call, 62 calls an NFE, 30 ms and 9.4% of the
denoise - a matrix-vector product that expensive has no obvious home in this
architecture, and if it is a skinny projection falling into a bad kernel it is
the next lossless win.

Result: reward **3.20126**, speedup 3.895x mean (6.83 s -> 6.71 s on seven of
eight cases), mean LPIPS 0.1773, worst 0.2820. New best, narrowly. Artifacts:
`runs/round-15/`, job 19013891, source commit `9259962`.

Autotuning bought about 2% of the denoise, 6.39 s -> 6.26 s, and cost 0.0004 of
mean LPIPS, which is within the noise of a kernel reordering. cuBLAS kept every
`aten::mm`: 514 ms of self CUDA over three NFE, unchanged. What changed is what
was around them. The `gemv2N` kernel that took 486 us a call and 30 ms an NFE is
gone, and so is `aten::bmm`; both were absorbed into two Triton templates,
`triton_tem_fused__softmax__to_copy_bmm_cat_exp_permute...`, that together cost
35 ms an NFE where the pair cost 57 ms. So the mystery GEMV was the linear
attention's own contraction, sitting in a cuBLAS kernel with nothing to do, and
a template with a fused softmax epilogue replaced it. Per NFE the denoiser is
now mm 171 ms, flash attention 43 ms, the fused attention templates 35 ms,
`triton_poi_fused_addmm_mul_silu` 23 ms, `cat` 7 ms.

One blemish: case 01, the first timed generation, spent 700 ms in its text phase
where every later case spent 56 ms, so it measured 3.593x against their 3.94x.
That 643 ms one-off is new with autotuning - deferred compile work, a lazily
created library workspace, or allocator growth after the diagnostics released
their buffers. It costs about 1% of the reward and belongs in the warmup.

## Round 16 - move the one-off into the warmup, and price float8 on real tensors

Two changes, both off the timed path.

The warmup now walks every phase a second time after its diagnostics: text
encode, one denoiser forward, one decode, then a synchronize. Whatever the
643 ms is, paying it there should put case 01 back at 3.94x, which is worth
about 0.03 of reward and, more usefully, makes the eight cases comparable
again.

And the float8 decision gets the measurement it needs. `benchmark_gemm` reported
0.0376 relative error, but on Gaussian inputs, which is the friendliest case for
a single per-tensor scale - real transformers break it with outlier activation
channels. The warmup now captures the actual activation entering each linear
layer of the middle block during the first denoiser call, and reports per-tensor
and per-row float8 error against the bf16 result on those real tensors, with the
amax-to-mean ratio of each weight and activation next to it. Round 13's
benchmark said float8 is worth 13% of the NFE; this says what it would cost, and
the quality budget after round 12 is 0.12 of mean LPIPS, so the number decides
whether round 17 implements it.

The probe attaches forward pre-hooks inside compiled regions and removes them
again, which can change a dynamo guard; the denoiser forward in the settle step
is there to make any resulting recompilation happen in the warmup rather than on
a timed step.

Result: reward **3.24495**, speedup 3.952x (6.68 s), mean LPIPS 0.1773, worst
0.2820 - both identical to round 15, as a change confined to the warmup should
leave them. New best. Artifacts: `runs/round-16/`, job 19014147, source commit
`5332ce5`.

Walking every phase a second time at the end of the warmup put case 01 back with
the rest: 3.593x -> 3.949x, and its text phase 700 ms -> 55 ms. Whatever the
one-off was, it is now paid where it belongs. That is +0.044 of reward for work
that does nothing to the output.

The float8 probe, however, captured nothing: all nine layers reported `no
activation captured`. Dynamo's `skip_nnmodule_hook_guards` defaults to true,
which means a compiled region does not run `nn.Module` forward hooks at all -
the hooks were installed correctly and simply never fired. That is worth
remembering: a hook inside a compiled region is not a measurement, it is
silence.

## Round 17 - price float8 properly, and check the attention backend

The output is unchanged again this round; all three additions are in the warmup.

The probe now runs its capture with `torch._dynamo.config.disable` set, so the
forward executes eagerly and the hooks fire. It also runs on a mid-trajectory
`(x, t)` pair kept from step 25 of the warmup rather than on pure noise, since
that is where a numerical perturbation actually matters.

`model/fp8.py` implements the thing being priced: `Fp8Linear` quantizes its
weight once at conversion and the activation on each call, then calls
`torch._scaled_mm`, in either per-tensor or per-row scaling. It selects the
large linear layers on the per-step path, excluding `cross_attn.kv_linear`,
which runs once per request on a row count that is not a multiple of 16 and has
not been on the per-step path since round 2, and excluding the 20-column
`beta_proj`. A row count the tensor cores cannot take falls back to `F.linear`
rather than raising inside a timed generation. It ships with `SANA_FP8=off`, so
this round's generations are bit-identical bf16; what the warmup adds is a
measurement of two numbers that decide round 18:

- per layer, the error one float8 GEMM makes on the activation the model itself
  produced, next to the amax-to-mean ratio of that activation and that weight;
- whole model, the relative error in the velocity when every eligible layer is
  converted, for both scalings.

The second number is the one that predicts LPIPS. For scale: an exact bf16
reordering perturbs the velocity by about 1e-6 and cost 0.04 LPIPS over 50
steps, and the quality budget left above round 16 is 0.12 of mean LPIPS against
a 13% speedup. The conversion machinery - which layers are chosen, that the
module tree is restored to the identical objects, that 'off' cannot change
anything, that the weight copy is the column-major operand `_scaled_mm` wants,
and that an unaligned row count falls back exactly - is covered offline; the
GEMM itself needs the H100 and is exercised by the probe.

The third addition benchmarks every scaled-dot-product backend at the anchor
layers' shape. Those eight layers spend 43 ms of each 319 ms NFE in one flash
kernel at head_dim 256, which is the edge of what those kernels are tuned for,
and it is worth knowing whether cuDNN's implementation is faster before treating
43 ms as the floor.

Result: reward **2.50149**, speedup 3.045x (8.67 s), mean LPIPS 0.1787, worst
0.2743. A large regression against round 16's 3.24495, and the round was spent.
Artifacts: `runs/round-17/`, job 19014645, source commit `75ad681`.

Float8 shipped `off`, so the generations should have been bit-identical to round
16's. They were not: every case went from 6.68 s to 8.67 s, with text (52 ms)
and VAE (390 ms) unchanged and the whole 1.96 s sitting in the denoiser, and
mean LPIPS moved 0.1773 -> 0.1787. Both symptoms say the same thing - the timed
generations ran different kernels.

The cause is the probe's own scaffolding. `torch._dynamo.config.disable = True`
does not merely skip compilation for the call it wraps: dynamo records the code
objects it walks past while disabled, and they stay recorded. Restoring the flag
afterwards restores nothing. Every denoiser forward for the rest of the process
ran uncompiled, which is also why the warmup VAE went 2343 ms -> 6000 ms. The
lesson generalises past this round: a flag that turns compilation off is not
scoped to the block you set it in, and the only safe way to run something eagerly
in a process that also runs compiled is to not run it in that process.

The measurements it bought are worth having, and they set up the next two rounds.

    [attn-bench] batch=2 heads=10 tokens=8160 head_dim=256
    [attn-bench]   default:              4.239 ms  deviation=0.00e+00
    [attn-bench]   flash_attention:      4.237 ms  deviation=0.00e+00
    [attn-bench]   efficient_attention:  9.341 ms  deviation=2.21e-03
    [attn-bench]   cudnn_attention:      1.911 ms  deviation=2.21e-03
    [attn-bench]   math:                44.157 ms  deviation=1.57e-03

The dispatcher picks flash and cuDNN is 2.2x faster than it at this shape.
head_dim is 256 here - twice what the flash kernels are usually tuned for - which
is the likely reason. Eight anchor layers x 2.33 ms is about 19 ms of every
313 ms NFE, 6%, for a 2.2e-3 perturbation.

    [fp8] velocity error with 260 layers in float8, tensor scaling: 0.0284
    [fp8] velocity error with 260 layers in float8, row scaling:    0.0269

Per-tensor scaling costs 2.84% of the velocity and per-row 2.69% - close enough
that the faster kernel wins on the trade (round 13: 1.73x per-tensor against
1.24x per-row on the dominant GEMM shape). Half the denoise is `aten::mm`, so
that is worth about 13% for a 2.8% perturbation, against 0.12 of mean-LPIPS
budget. That is round 19.

The per-layer capture reported nothing again - `[fp8] 9 linear layers, 0 captured
activations` - because the forward hooks it relies on still did not fire. The
whole-model number is the one that decides anything, so the capture is being
deleted rather than fixed.

## Round 18 - undo the regression, and take the attention measurement

Two changes, both of them consequences of round 17's own output.

The probe and everything that supported it comes out: `_probe_precision`, the
mid-trajectory `(x, t)` sample the step hook kept for it, and the activation
capture. `model/fp8.py` and its conversion hook stay, still shipping `off`, for
round 19. This alone should put the reward back at 3.245.

The eight anchor layers then get the backend that measured fastest. The awkward
part is that `sdpa_kernel` takes a *set* of enabled backends, not a priority
order - and dynamo silently drops its `set_priority` argument, so a list is not a
preference, it is a permission. Naming cuDNN alone means a shape cuDNN refuses
raises inside a timed generation rather than falling back to flash.

So the choice is made once, before anything is compiled, by `select_backend`: it
builds q, k and v at the anchor shape and in the anchor layouts - q and k
contiguous as `_apply_rope` returns them, v transposed as the qkv view leaves it,
because layout is part of what a backend is selected on - times cuDNN against
whatever the dispatcher picks on its own, and adopts cuDNN only if it runs and
wins by more than 10%. It then traces the call through dynamo, because the
backend is selected a second time against fake tensors at compile time and an
empty selection there is a hard error in the middle of the warmup; declining here
is free. The attention layer takes the fast path only for the exact shape and
dtype that were measured, and every other call keeps the dispatcher's choice.

The shape has to be written down (16 x 17 x 30 latent positions, both CFG rows)
because the probe runs before there is a latent to read it off. The warmup
compares that guess against the real thing and prints `[attn] MISMATCH` if they
differ, so a silent no-op cannot be mistaken for a null result.

Expect the denoiser to lose about 19 ms per NFE across 20 NFE - 0.38 s of 6.26 s
- for a perturbation of 2.2e-3 on the anchor outputs, three orders of magnitude
larger than a bf16 reordering but still far below a skipped step. Reward around
3.4 if the LPIPS cost stays small; a small rise in mean LPIPS would not be a
surprise given that reordering alone has cost 0.04 before.

Result: reward **3.40426**, speedup 4.151x (6.36 s), mean LPIPS 0.1796, worst
0.2849. New best, past round 16's 3.24495. Artifacts: `runs/round-18/`, job
19015862, source commit `2993258`.

Per case: 01 0.1080, 02 0.1889, 03 0.1741, 04 0.2387, 05 0.0958, 06 0.1362,
07 0.2849, 08 0.2103.

Both halves landed. Deleting the probe put the denoise back where round 16 left
it, and cuDNN took it further: 6.26 s -> 5.92 s, 340 ms over 20 NFE, 17 ms each
against the 19 ms predicted. Text and VAE are unchanged at 53 ms and 390 ms.

    [attn] (2, 10, 8160, 256) torch.bfloat16: dispatcher 4.258 ms,
           cuDNN 1.867 ms, deviation 2.21e-03 - adopted
    [attn] cuDNN active on the anchor layers at (2, 10, 8160, 256)

The load-time probe agreed with round 17's bench to within 2%, the adoption
criterion fired, and the shape it was told matched the shape the anchors saw -
which the warmup checked rather than assumed. In the profile the flash kernel at
head_dim 256 is gone and `cudnn_generated_fort_native_sdpa_sm90_flash_fprop`
stands in its place; the 230 remaining flash calls are the cross-attention, whose
shapes are small.

The quality cost was 0.0023 of mean LPIPS (0.1773 -> 0.1796) and 0.0029 of the
worst case, for 5% of the wall clock. That is the going rate for a reordering at
this scale, and the trade is clearly worth it - but it is worth writing down that
a 2.2e-3 perturbation of eight attention outputs moves the metric at all. There
is no free reordering here; there is only a cheap one.

What the round leaves is a much less ambiguous target. The denoiser profile:

    aten::mm                                  525.3 ms   49.7%   768 calls
    cudnn_generated_..._sdpa_sm90_flash_fprop  61.8 ms    5.9%    29 calls
    triton_poi_fused_addmm_mul_silu_view_1     37.6 ms    3.6%   116 calls
    pytorch_flash::flash_fwd_kernel            14.0 ms    1.3%   230 calls

over 3 NFE of 1.056 s. `aten::mm` is now half of all device time - 175 ms of
every 352 ms NFE - and nothing else is within a factor of eight of it. Attention
has been optimised to the point where it no longer matters.

## Round 19 - the feed-forward GEMMs in float8

Round 17 measured the whole-model velocity error at 0.0284 with all 260 eligible
linears in per-tensor float8 and 0.0269 per-row. The two are close enough that
the faster kernel wins the trade: round 13 measured 1.73x on the dominant shape
for per-tensor against 1.24x for per-row. So per-tensor it is.

Converting all 260 at once would stake the whole 0.12 of remaining mean-LPIPS
budget on one round, and a gate failure returns no number at all - round 13 and
round 14 both cost a round and told me only "more than 0.30". So this round
scopes the conversion to the feed-forward: `gate_proj`, `up_proj`, `down_proj`,
three projections of 2560x10240 against attention's 2560x7680 and two 2560x2560.
That is about 70% of the linear FLOPs in about half the layers - most of the
speed for a smaller perturbation - and the two scopes are nested, so whichever
way it goes the result says what round 20 should do.

Expect the MLP's share of `aten::mm` - roughly 122 ms of the 175 ms per NFE - to
come down by 1.73x, so about 50 ms off each of 20 NFE: 5.92 s -> 4.9 s of
denoise, 6.36 s -> 5.35 s overall, speedup near 4.9. Mean LPIPS is the open
question. A 2.2e-3 perturbation of the attention outputs cost 0.0023 this round,
and float8 is an order of magnitude coarser than that, so 0.19-0.23 is the range
I would bet on, for a reward somewhere between 3.8 and 4.0. If it lands above
0.30 the gate takes it to zero and float8 is finished as an axis.

Result: reward **3.57110**, speedup 4.517x (5.85 s), mean LPIPS 0.2094, worst
0.3150. New best, past round 18's 3.40426. Artifacts: `runs/round-19/`, job
19016471, source commit `49345c5`.

Per case: 01 0.1386, 02 0.2024, 03 0.2124, 04 0.2424, 05 0.1738, 06 0.1668,
07 0.3150, 08 0.2237.

96 layers converted - three feed-forward projections in each of 32 blocks.
Denoise 5.92 s -> 5.40 s, 26 ms off each of 20 NFE, against the 50 ms predicted.
Mean LPIPS rose 0.030 and the worst case 0.030. Half the predicted speed for the
full predicted quality cost, and the profile says exactly where the other half
went:

    aten::mm                                   193.1 ms   480 calls
    aten::_scaled_mm                           192.1 ms   288 calls
    triton_poi_fused__to_copy_clamp_div_view_2  29.0 ms   336 calls
    triton_red_fused_abs_amax_0                 24.8 ms   336 calls

over 3 NFE. The GEMM did what round 13 promised: the feed-forward's share of
`aten::mm` went from about 110 ms per NFE to 64 ms, 1.72x. Then 18 ms per NFE
came back as two kernels that did not exist before - an `abs.amax` reduction and
a `div.clamp.to` cast.

They exist because the scale is computed from the tensor being scaled. A
per-tensor amax is a global reduction, and a global reduction is a barrier:
inductor cannot fold the cast into the epilogue of the kernel that produced the
activation, because the cast needs a number that depends on all of it. So the
activation is written out in bf16, read back for the amax, and read a third time
for the cast. 18 ms of every 337 ms NFE is the price of that dependency, and it
is 0.36 s of the round's 5.85 s.

Worth recording alongside it: the quality cost of float8 is real but not
catastrophic. Case 07, already the worst at 0.2849, went to 0.3150 - above the
*mean* gate on its own, though the worst-case gate is 0.50 and it is nowhere
near that. The mean has 0.09 of headroom left against 0.30.

## Round 20 - fix the scale, and the cast stops being a kernel

A scale that does not depend on the activation is not a barrier. The warmup
already runs the full 50-step trajectory, so each converted layer records the
largest scale it sees over all of it, and `freeze` then multiplies by a margin
and switches the layers to dividing by that constant. The amax kernel disappears
and the cast becomes an elementwise operation on a value already in registers.

The part worth being explicit about is why a margin is affordable, because this
would be a bad idea on int8. float8_e4m3 is a floating point type with a 4-bit
exponent: the scale only has to land the tensor somewhere inside a 17-binade
window, not fill a fixed-point range. Doubling it changes the exponent and
leaves all three mantissa bits untouched. Measured offline, the same activation
at 1x, 2x and 8x headroom quantizes to a relative error of 0.02254 every time,
where int8 goes 0.01305 -> 0.10459 over the same range. So the margin is free,
and an activation past it clamps rather than overflowing.

Freezing flips an attribute the compiled code is guarded on, so the denoiser
recompiles. That happens before `_settle`, whose denoiser forward pays for the
recompile inside the excluded warmup - the same trap round 3 fell into and round
16 built `_settle` to avoid.

Expect 18 ms off each of 20 NFE, 5.85 s -> 5.5 s, speedup near 4.8, with LPIPS
essentially unchanged: the arithmetic is the same float8 GEMM, only the scale
differs, and it differs by a factor that costs no mantissa. Reward around 3.79.

Result: reward **2.99862**, speedup 3.725x (7.09 s), mean LPIPS 0.1949, worst
0.3016. A regression against round 19's 3.57110. Artifacts: `runs/round-20/`,
job 19017029, source commit `372f47a`.

Calibration and freezing themselves worked exactly as designed:

    [fp8] froze 96 layers at margin 2; calibrated activation amax
          min 0.8 median 1.8 max 274.0

All 96 layers recorded a scale, all 96 froze, and the spread - a median amax of
1.8 against a maximum of 274 - is the argument for per-tensor scaling on a
floating point type in one line. Mean LPIPS even improved, 0.2094 -> 0.1949.

The denoise went 5.40 s -> 6.65 s. The measurement log says why, in a warning
that `suppress_errors = True` turned from a crash into a silent 23% regression:

    WON'T CONVERT forward /submission/model/fp8.py line 90
    torch._inductor.exc.LoweringException: AssertionError:
      target: aten._scaled_mm.default
      args[0]: ComputedBuffer(name='buf3', ... float8_e4m3fn,
               size=[16320, 2560] ... Pointwise ...)

`args[0]` is the quantized activation reaching `_scaled_mm` as an *unrealized*
Pointwise - the cast folded into the GEMM template's prologue. That is precisely
what removing the reduction barrier enabled, and rendering the template around
it trips an assertion in its own index/stride bookkeeping
(`assert stride is not None and len(index) == len(stride)`). Three regions
failed to lower - `fp8.forward`, `SwiGLU.forward`, `forward_mlp_sublayer` - so
the entire feed-forward ran eager for all eight cases.

So the change did what it was supposed to do, and the thing it enabled is what
broke. Two lessons, and they are not the same one:

Removing a barrier is not a local change. I reasoned about the barrier as a cost
- an extra kernel and an extra read - and it was also the thing keeping the cast
out of a code path that cannot compile it. An optimisation that unlocks *new*
fusion has to be tested against the fusion it unlocks, not just against the
arithmetic it removes.

And `suppress_errors = True` is not a safety net, it is a silencer. It was put
there in round 12 so that an unlowerable region would degrade instead of failing
the round, and it did exactly that - but "degrade" meant handing back 16% of the
reward with no failure anywhere in the ledger. It stays, because a crash costs
the whole round rather than 16% of it, but from here anything that changes what
inductor sees has to be measured on the node, not reasoned about.

Worth writing down for later rounds: the container runs torch 2.9.1+cu128 on
python 3.11, while the offline tests in `/tmp/sana_eq` run 2.7.1 on 3.10. Those
tests check Python-level logic and will never see a lowering difference.

## Round 21 - turn off the fusion that broke, and stop betting on it

Prologue fusion goes off (`TORCHINDUCTOR_PROLOGUE_FUSION=0`, plus the config
attribute if this torch has one). It is what was wanted anyway: the point of
freezing the scale was to let the cast fold into the epilogue of the kernel that
*produces* the activation, not to re-quantize it per tile inside the GEMM.

And the decision stops being a bet. `fp8.tune` times a denoiser forward at the
real shape, freezes, pays for the recompile, times it again, and unfreezes if
the frozen path is not at least 2% faster. Six forwards and one recompile, all
inside the excluded warmup, against a round for being wrong - the same trade
round 18's attention probe made.

Either result is informative. If freezing is adopted, expect 18 ms off each of
20 NFE, about 5.5 s and a reward near 3.8. If it is rejected, the round ships
round 19's configuration with prologue fusion off, and the speedup against round
19's 4.517x measures exactly what that knob costs the bf16 GEMMs - which is the
number I would otherwise have no way to get.

Result: reward **3.56985**, speedup 4.515x (5.85 s), mean LPIPS 0.2094, worst
0.3150 - LPIPS identical to round 19's to sixteen digits. Artifacts:
`runs/round-21/`, job 19017733, source commit `808570c`.

The guard fired:

    [fp8] froze 96 layers at margin 2; calibrated ... min 0.8 median 1.8 max 274.0
    [fp8] frozen scales rejected: 264.2 ms -> 332.4 ms per forward

and the round shipped round 19's configuration. The reward matches round 19's to
0.0013, which is the run-to-run noise on this harness.

Three things came out of it.

Freezing is refuted, and not for the reason round 20 suggested.
`TORCHINDUCTOR_PROLOGUE_FUSION=0` was set and the lowering still failed -
`tuned_scaled_mm` raises the same index/stride assertion while *rendering its
Triton template*, which happens at lowering time, before any fusion decision is
made. So prologue fusion was never the trigger; my reading of round 20's dump
was wrong. The trigger is that a constant scale changes what the scaled-mm
template is asked to render, and this torch cannot render it. Nothing about that
is fixable from here, so `SANA_FP8_STATIC` ships off. The mechanism stays - a
scale that does not depend on the activation genuinely is not a barrier - in
case a torch release makes it viable.

Prologue fusion is neutral for the bf16 GEMMs. This is the number the round
bought that I could not have obtained any other way: round 21 ran round 19's
exact configuration with the knob off and returned 4.515x against 4.517x. Off it
stays, since it costs nothing and removes a class of lowering surprise.

And the guard itself works, for about six forwards inside the excluded warmup.
Round 20 lost 0.57 of reward to a silent eager fallback; round 21 caught the
same fallback, rolled back, and lost 0.0013. Measuring the thing on the node is
cheaper than reasoning about it, every time.

## Round 22 - the other half of the GEMMs

Round 19 converted 96 feed-forward projections: `aten::mm` went from 175 ms per
NFE to 64 ms of `mm` plus 64 ms of `_scaled_mm`, for 8.8% of the wall clock and
+0.030 of mean LPIPS. That 64 ms still in bf16 is attention's qkv, output
projection and gate, and the cross-attention query - the other half of the
linear FLOPs, and now the largest single item in the profile.

`SANA_FP8_SCOPE` goes to 'all'. The scopes are nested, so this is the same trade
at the same price on the other half: expect about 27 ms off each of 20 NFE,
5.85 s -> 5.3 s, speedup near 4.97, mean LPIPS somewhere around 0.23-0.24 with
the worst case near 0.34. The gate is 0.30 and 0.50, so both have room, and
round 17 measured the whole-model velocity error over exactly this scope at
0.0284.

Freezing ships off, so the six probe forwards go away with it.

The warmup also starts reporting dynamo's graph-break counters. Round 19's
profile shows `mul`, `add`, `cat` and `silu` arriving as separate eager kernels
worth about 38 ms of every NFE - 11% of the denoiser - inside sublayers that are
supposed to be compiled end to end. I have been guessing at why; the counters
say it for nothing.

Result: reward **3.47744**, speedup 4.517x (5.84 s), mean LPIPS 0.2303, worst
0.3262 - a regression of 0.094 against round 19. Artifacts: `runs/round-22/`,
job 19018277, source commit `7ba79df`.

    [fp8] 260 linear layers converted, scaling=tensor scope=all

164 more layers than round 19, and the wall clock did not move: 4.5168x against
round 19's 4.5170x, denoise 5.40 s in both. The quality did: +0.021 of mean
LPIPS and +0.011 of worst. Zero speed for a real cost, which is the worst shape
a result can have.

I predicted 4.97x from "the same trade on the other half of the GEMMs", and the
arithmetic says that was never available. Quantizing an activation is one pass
over it - M*K elements read, reduced and written. The GEMM it feeds is 2*M*K*N.
So the overhead-to-gain ratio goes as 1/N, and float8 pays where the output is
wide. `gate_proj` and `up_proj` are 2560 -> 10240 and pay four times over.
`proj`, `output_gate` and the cross-attention query are 2560 -> 2560: the same
pass over the activation buys a quarter of the arithmetic. I had the 1.73x
kernel number from round 13 and multiplied it by FLOPs instead of by shape.

That is the half of the explanation I could have derived. The log holds the
other half, and it is larger.

## The regions that were never compiled

Round 22 printed nine of these:

    WON'T CONVERT _self_attn_delta /submission/model/attention.py line 395
    due to: LoweringException: AssertionError
      target: aten._scaled_mm.default

with `suppress_errors = True` turning each one into a silent eager region.
`aot_autograd.not_ok: 9` against `ok: 30`. So round 22 converted 164 more
layers and, in exchange, ran the self-attention delta, the cross-attention
merge, the linear-attention query and merge, and the float8 layer itself
outside the compiler. The GEMM got faster and everything around it got slower,
which is exactly the 4.5168-against-4.5170 that came back.

Then I looked at the earlier rounds:

    round 16  0 failures      round 19  3 failures
    round 17  0 failures      round 21  3 failures
    round 18  0 failures      round 22  9 failures

Round 19 - the best round of this campaign, the one `FINAL.md` currently points
at - has been running `forward_mlp_sublayer` eager since it shipped. That is
the 38 ms per NFE of `mul`, `add`, `cat` and `silu` I have been attributing to
"glue outside the compiled sublayers" for three rounds, and it is not outside
them at all. `[compile] 0 graph break(s)` was true and also irrelevant: dynamo
traced the whole thing, and inductor threw it away afterwards.

Round 18 is the one that explains it. Round 18 is per-row, and the difference
between per-row and per-tensor here is the rank of the scale. Inductor's
scaled-mm template loads its epilogue inputs at the output's index - two
symbols, `idx_m` and `idx_n` - and `FixedLayout.indexer` asserts
`len(index) == len(stride)`. A per-row scale is rank 2 and passes. A per-tensor
scale is `FixedLayout(torch.float32, size=[1], stride=[1])`, rank 1, and cannot.

Per-tensor float8 has therefore never once been compiled in this campaign.
Rounds 19, 20, 21 and 22 all measured it eager. Round 20's "frozen scales are
unlowerable" was the same assertion on the same rank-1 scale, and my prologue
fusion theory in round 20 and my "a constant scale changes what the template is
asked to render" correction in round 21 were both wrong about a mechanism that
has nothing to do with freezing.

## Round 23 - pay what the arithmetic says, and get it compiled

Three changes, each with its own line in the log so the round can be read even
if it regresses.

First, race the two paths before converting anything. For every distinct
(in_features, out_features) among the eligible layers, time bf16 `F.linear`
against `Fp8Linear.forward` at the row count the denoiser actually runs -
16320, both classifier-free-guidance rows of all 8160 latent positions - and
convert only the shapes where float8 wins by 5%. This is the same thing
`select_backend` does for cuDNN attention and for the same reason: the shapes
are known at load time, the measurement costs a few hundred microseconds inside
the excluded warmup, and I keep getting the constants in these arguments wrong.
It should reproduce round 19's selection from measurement rather than from a
scope string, and it will keep doing so if a later round changes the shapes.

Second, decline inductor's Triton template for `_scaled_mm`. cuBLAS is what
every float8 GEMM in this model has actually executed on, in eager, in every
round that used one; offering only that choice makes the lowering succeed and
the surrounding sublayer compile. The narrowing is scoped to that one lowering -
the backend list is flipped to ATEN for the duration of the call and restored
in a `finally` - so the bf16 GEMMs keep their templates and their epilogue
fusion. Reshaping the scale to (1, 1) would also get past the assertion, and
would then be read as a per-row scale by the same template, which would index a
one-element tensor with row indices; that is a wrong answer rather than a
failure, so it is not on the table.

Third, drop the running activation maximum from the hot path. It is a device
reduction and a buffer mutation in the middle of the region on every converted
layer of every step, and it feeds only the calibration that round 21 refuted.
It is now gated on `SANA_FP8_STATIC`, which ships off.

Expect the WON'T CONVERT count to go to zero and `aot_autograd.not_ok` with it,
the converted-layer count to come back to about 96, and mean LPIPS to return to
round 19's 0.2094. The speed is the open question: if the eager `forward_mlp_
sublayer` really is the 38 ms per NFE, compiling it is worth 0.7 s of the 5.4 s
denoise and the speedup goes past 5. If it is not, the round costs nothing and
tells me the glue lives somewhere else - which is worth knowing, because that
number has been the largest unexplained item in the profile since round 16.

Result: reward **3.41651**, speedup 4.152x (6.34 s), mean LPIPS 0.1796, worst
0.2849. Artifacts: `runs/round-23/`, job 19019088, source commit `bbaed30`.

Half of it worked completely and half of it broke in a way that was worth the
round.

The lowering is fixed. Zero `WON'T CONVERT` in the whole log, against nine in
round 22 and three in round 19, and `aot_autograd` went from `ok: 30,
not_ok: 9` to `ok: 20, not_ok: 0`. Declining the Triton template for
`_scaled_mm` is enough; the twenty regions that survive are the whole denoiser.
That is a durable fix and it applies to every future round that uses float8.

The race then rejected every shape in the model:

    [fp8] 16320x2304x2560:  bf16 0.250 ms, float8 0.677 ms, 0.37x - keep bf16
    [fp8] 16320x2560x2560:  bf16 0.274 ms, float8 0.739 ms, 0.37x - keep bf16
    [fp8] 16320x2560x7680:  bf16 0.803 ms, float8 1.053 ms, 0.76x - keep bf16
    [fp8] 16320x2560x10240: bf16 1.075 ms, float8 1.214 ms, 0.89x - keep bf16
    [fp8] 16320x2560x15360: bf16 1.591 ms, float8 1.534 ms, 1.04x - keep bf16
    [fp8] 16320x10240x2560: bf16 1.036 ms, float8 2.764 ms, 0.37x - keep bf16
    [fp8] 0 linear layers converted

So the round shipped bf16 everywhere and returned round 16's numerics with
round 18's speed. Reward 3.4165.

2560 -> 10240 at 0.89x is the line that gives it away. That is `gate_proj` and
`up_proj`, and round 19 measured them winning 8.8% of the wall clock in the
model. A measurement that contradicts a measurement is a broken harness, and I
shipped it anyway because I only checked it returned `None` on CPU.

What it measured was `Fp8Linear.forward` in eager. Eager, quantizing is four
separate passes over the activation and the first one, `flat.float()`,
materializes a 16320 x 2560 float32 tensor - 167 MB written and read back
before the divide, the clamp and the cast each take their own turn. In the
model those four passes are one fused Triton kernel. The race charged float8
roughly four times its real quantization cost and then reported, correctly,
that nothing could pay for it. The row count was the production row count, the
dtype was the production dtype, and the *regime* was not, which is the same
class of error as testing a collective on an even shard it never runs on.

Two other things the numbers say. The eager float8 penalty scales with K, not
N - 10240 -> 2560 lost 2.7x where 2560 -> 2560 lost 2.0x - which is the
quantization pass being proportional to M*K, exactly as the 1/N argument says.
And bf16 2560 -> 15360 runs at 1.591 ms against 2560 -> 10240 at 1.075 ms, a
1.48x for a 1.5x, so these GEMMs really are compute-bound and there is nothing
to win in bf16 by rearranging them.

## Round 24 - the same race, in the regime the model runs in

Compile the float8 side before timing it. The bf16 side stays eager because an
isolated `F.linear` is one extern cuBLAS call either way, with nothing around
it to fuse into; the float8 side is exactly the side where compiling changes
the answer. Autotuning is off for those six compiles - what is being measured
is the cost of the cast, and six autotuned compiles do not fit in a
fourteen-minute reservation.

The compiled measurement is still conservative. In the model, inductor can fold
the quantizing pass into the epilogue of the kernel that produced the
activation, so it costs less there than in a standalone compiled layer, and a
shape that wins the race wins by more in situ.

And a race that rejects everything now falls back to the feed-forward instead
of to nothing, with a line in the log saying so. Unanimous rejection is
evidence the harness broke.

Prediction: `gate_proj` and `up_proj` at 2560 -> 10240 convert, `qkv` at
2560 -> 7680 and the 2560 -> 15360 projection probably convert, `down_proj` at
10240 -> 2560 is the interesting one - its quantization pass is four times the
others' and its output is narrow, so the 1/N law says no and round 19's
measured 8.8% over all three says the group wins together. Whichever way it
falls, it falls on a number. With the lowering fixed as well, round 19's
configuration should now be strictly better than round 19 was: same layers,
same numerics, but `forward_mlp_sublayer` compiled. Speedup past 4.8, mean
LPIPS at or below 0.2094, reward above 3.8.

Result: reward **3.87605** - the best of the campaign, 0.305 clear of round 19 -
speedup 4.980x (5.30 s), mean LPIPS 0.2206, worst 0.3180. Artifacts:
`runs/round-24/`, job 19019533, source commit `bafa79a`.

Compiling the float8 side reversed every line of the race:

    shape              eager (round 23)      compiled (round 24)
    2304 -> 2560       0.37x  keep bf16      0.96x  keep bf16
    2560 -> 2560       0.37x  keep bf16      0.94x  keep bf16
    2560 -> 7680       0.76x  keep bf16      1.36x  convert
    2560 -> 10240      0.89x  keep bf16      1.43x  convert
    2560 -> 15360      1.04x  keep bf16      1.49x  convert
    10240 -> 2560      0.37x  keep bf16      1.12x  convert

Same GEMMs, same row count, same dtype; the only difference is whether the four
passes of the quantization are four kernels or one. It is worth a factor of
2.6 at 2560 -> 2560 and a factor of 3 at 10240 -> 2560.

The selection it produced is the one the 1/N argument predicts and neither
round 19 nor round 22 found. 129 layers: `qkv`, `gate_proj`, `up_proj`,
`down_proj` and the wide shared projection - everything whose output is at
least three times its input, plus `down_proj` at 1.12x, which the argument said
to drop and the measurement said to keep. `proj`, `output_gate` and the
cross-attention query at 2560 -> 2560 stay bf16 at 0.94x, which is why the mean
LPIPS came in at 0.2206 against round 22's 0.2303 while running 10% faster.

Denoise went 5.39 s (round 19) -> 4.86 s. Two things did that: `down_proj` and
`qkv` are converted where round 19 left `qkv` in bf16, and, more importantly,
`forward_mlp_sublayer` is compiled for the first time since round 18.

One `WON'T CONVERT` is left and it belongs to the race, not the model:
compiling `Fp8Linear` on a layer whose weights still require grad makes
autograd ask for a derivative of `_scaled_mm`, which does not exist. It cost
nothing - the timings are all clearly the fused ones - but the race should run
under `inference_mode` like `select_backend` does.

## Round 25 - the scale that does not need to be measured

The profile has moved. Per NFE, in the warmup's instrumented pass:

    _scaled_mm                                      79 ms
    aten::mm (bf16)                                 37 ms
    linear-attention Triton templates               35 ms
    abs/amax + layer_norm reductions                24 ms
    cuDNN softmax attention                         20 ms
    RMS-norm / depth aggregation reductions         20 ms
    quantizing casts (clamp/div/to_copy)            19 ms
    silu + amax                                     17 ms

Roughly 60 ms of every 243 ms NFE is now spent finding and applying activation
scales. The amax itself is fused into the layer norm that precedes it, so it is
nearly free; what is not free is that a global reduction cannot be fused with
the cast that depends on it, so every converted layer reads its activation
twice.

A calibrated scale removes the dependency and lets the cast fold into the
epilogue of the kernel that produced the activation. That is what round 20
tried and round 21 measured at 264 ms -> 332 ms per forward, and both of those
rounds were measuring the eager fallback: a frozen scale is a rank-1 constant,
and rendering the Triton scaled-mm template around one raises the same
assertion that round 23 fixed by declining the template. The mechanism was
never tested; the lowering bug was.

So `SANA_FP8_STATIC` goes back on, and the guard that saved round 21 stays: the
warmup times six forwards dynamic, freezes, times six more, and rolls back
unless freezing is 2% faster. If the 19 ms of casts fold away, that is 8% of
the denoise and a speedup near 5.4.

The margin stays at 2.0. float8_e4m3 has a 4-bit exponent, so a factor of two
of headroom against a prompt whose activations run hotter than the warmup's
moves the exponent and costs none of the three mantissa bits - but only for
activations at the top of the range. Round 21's calibration found per-layer
amax spanning 0.8 to 274, so the quality risk is real and this round has 0.079
of mean-LPIPS headroom to absorb it.

The race also moves under `inference_mode`.

Result: reward **4.00739** - best of the campaign - speedup 5.137x (5.14 s),
mean LPIPS 0.2199, worst 0.3146. Artifacts: `runs/round-25/`, job 19019784,
source commit `1276805`.

    [fp8] froze 128 layers at margin 2; calibrated activation amax
          min 0.8 median 3.2 max 258.8
    [fp8] frozen scales adopted: 240.1 ms -> 231.0 ms per forward

Adopted, on its own guard, at 3.8% per forward; the denoise went 4.86 s ->
4.70 s. And it cost nothing in quality - mean LPIPS 0.2199 against round 24's
0.2206, worst 0.3146 against 0.3180, both inside the noise and both slightly
better. Zero `WON'T CONVERT`, `aot_autograd ok: 26, not_ok: 0`.

Rounds 20 and 21 called this refuted and spent two rounds on it. It was worth
about 0.13 of reward all along, and what stood in the way was never the idea -
it was inductor failing to render a Triton template around a rank-1 scale, with
`suppress_errors` turning the failure into a silent eager region. Round 21's
probe measured 264 ms -> 332 ms and was measuring that fallback, not the frozen
scale. Two rounds spent refuting a mechanism, on evidence that was actually
about a lowering bug three layers away.

I got less than I predicted - 9 ms per forward where I expected 19 ms per NFE.
The cast does fold into the producer's epilogue, but the amax it replaces was
already fused into the layer norm, so what was removed is one pass over the
activation rather than a reduction and a pass.

## What the trade is worth now

Reward is `speedup * (1 - LPIPS)`, so at 5.14x and 0.22 a one percent speedup
is worth +0.040 and a hundredth of mean LPIPS costs 0.051. Break-even is about
0.008 of LPIPS per percent of speed. Everything since round 16 has been bought
at far better than that rate, and it is worth writing down because it changes
what is worth trying.

The schedule, for instance, looks like an obvious lever and is not. Round 13
measured 20 evaluations at mean LPIPS 0.178 and 16 at 0.305, and the thing that
tracks quality is the carried change per reused step, which the bisection puts
at 0.1042 for 20 evaluations and 0.1381 for 16. Interpolating on that axis,
18 evaluations sits at 0.1230 and costs about +0.07 of mean LPIPS - landing
near 0.29 against a 0.30 gate - for 11% fewer network evaluations, worth +0.44
against a cost of -0.36. A coin flip against a gate, for a tenth of a reward
point. The schedule stays at 20, and that is now a measured decision rather
than an inherited one.

## Round 26 - the accumulate the fp8 tensor cores were built for

`_scaled_mm` is the largest item in the profile at 79 ms of every 243 ms NFE,
ahead of the bf16 `mm` at 37 ms and the linear-attention templates at 35 ms.
Every one of those GEMMs runs with `use_fast_accum=False`, which is
`torch._scaled_mm`'s default and the conservative choice: it accumulates each
tensor-core tile into a float32 accumulator through the full-precision path.
The fast path accumulates within the tile at reduced precision and hands the
result to float32 only at the tile boundary, which is what the H100's fp8 MMA
was designed around.

The quality cost is bounded by the tile depth rather than by K - the error is
per 32 or 64 accumulations, not per 2560 - and these activations are already
carrying a 2e-3 relative quantization error from the cast in front of them, so
a change in the last bits of the accumulator is well under the noise the round
already accepts.

Expect 10-20% of 79 ms: 8 to 16 ms off each NFE, denoise 4.70 s -> 4.5 s,
speedup near 5.4, reward near 4.2.

The race decides the layer set as always, and it will also price fast
accumulation, since it times the layer as configured. Worth watching: the two
narrow shapes sit at 0.96x and 0.98x, and if the faster accumulate carries them
past the 1.05x margin they convert, which by the rate above is a bad trade -
round 22 measured those same 2560 -> 2560 projections at +0.021 of mean LPIPS,
and they are 15% of the NFE so they would have to win 2.6% of the total to pay
for it. If that happens the margin, not the accumulate, is what needs fixing.

Result: reward **4.29257** - best of the campaign - speedup 5.689x (4.63 s),
mean LPIPS 0.2455, worst 0.3816. Artifacts: `runs/round-26/`, job 19020124,
source commit `07a5b87`.

Fast accumulation is worth more than I estimated, and the thing I said to watch
happened:

    shape            round 25            round 26
    2304 -> 2560     0.98x keep bf16     0.97x keep bf16
    2560 -> 2560     0.96x keep bf16     1.05x convert
    2560 -> 7680     1.36x convert       1.50x convert
    2560 -> 10240    1.44x convert       1.58x convert
    2560 -> 15360    1.49x convert       1.66x convert
    10240 -> 2560    1.13x convert       1.23x convert

10% on the fp8 GEMM itself, which is the 10-20% predicted, and the
2560 -> 2560 projections crossed the 1.05x margin by a hundredth and brought
130 more layers in with them. 259 converted, denoise 4.70 s -> 4.20 s,
mean LPIPS 0.2199 -> 0.2455, worst 0.3146 -> 0.3816.

Net it is +0.285 of reward, so the round stands and the best moves. But it is
two changes in one measurement and the second one is the one I argued against
in advance: at 15% of the NFE and a 1.05x win, those layers bought about 0.9%
of the total, worth +0.036, and round 22 priced them at +0.021 of mean LPIPS,
worth -0.12. On that arithmetic they should be losing about 0.08 of reward
right now, hidden inside a 0.285 gain that came from somewhere else.

The frozen scales were adopted again, 227.8 ms -> 207.8 ms per forward, 8.8%
against round 25's 3.8% - a bigger share now that there are twice as many
converted layers whose casts fold away. The calibrated amax range widened from
0.2-258.8 to 0.2-3465.4, which is the attention projections joining: their
activations run three orders of magnitude apart across layers, and a single
per-tensor scale per layer still covers each of them.

## Round 27 - separate the two changes

`SANA_FP8_RACE_MARGIN` goes from 1.05 to 1.20. That is not a round number
chosen to exclude something; it is what the exchange rate asks for. A converted
shape has to win enough of the *total* NFE to pay for the quality it costs, and
at 0.008 of mean LPIPS per percent of speed, a group of layers priced at +0.021
of LPIPS needs 2.6% of the total. The 2560 -> 2560 group is about 15% of the
NFE, so it needs 1.2x within itself before it pays. The wide shapes are at
1.50x to 1.66x and are not close to the line; `down_proj` at 1.23x is, and it
stays in on the same arithmetic that puts the narrow ones out.

This restores round 25's 129-layer selection with fast accumulation on top,
which is the clean version of the experiment round 26 ran. Expect the narrow
layers' 0.9% of speed to go back - speedup near 5.64 - and the quality to
return to somewhere near 0.22, for a reward around 4.37. If instead the mean
stays near 0.245, then it was fast accumulation rather than the extra layers
that cost the quality, and that is worth knowing too, because fast accumulation
is paying for itself several times over and the layer selection is not.

Result: reward **4.11462**, speedup 5.330x (4.95 s), mean LPIPS 0.2280, worst
0.3653. Artifacts: `runs/round-27/`, job 19020560, source commit `031ba5b`.

A clean separation, and it says the opposite of what I argued.

Excluding the 130 narrow projections put mean LPIPS back almost exactly where
round 25 had it - 0.2280 against 0.2199, so those layers are worth +0.0176 of
mean LPIPS, within a hair of the +0.021 round 22 measured. That part of the
argument was right.

The speed was not. I predicted they were worth 0.9% of the total and they are
worth 6.3%: denoise 4.20 s -> 4.52 s, speedup 5.689 -> 5.330. So they pay
+0.178 of reward and round 26's margin was correct by accident.

The error is the same one round 23 made, in the opposite direction. The race
times a layer standing alone, and I read 1.05x as "this layer is 5% faster".
In the model it is not standing alone: its scale is frozen, so the quantizing
cast folds into the epilogue of the kernel that produced the activation and
costs nothing, where in the race it is a kernel of its own. **The race
understates the in-model win, and it understates it most for exactly the shapes
that sit near the line.** A shape that ties in the race wins in situ. The race
is a filter against shapes that lose badly, not a scale on which to do reward
arithmetic.

`down_proj` is the other evidence for that reading: it raced at 1.13x in round
25 and 1.22-1.23x here, and it is the shape whose quantization pass is four
times its neighbours'. The bigger the cast the race charges for, the more the
race is wrong by.

## Round 28 - restore the selection, and stop paying for headroom

`SANA_FP8_RACE_MARGIN` goes back to 1.05. Round 26's selection stands as
measured, and it stands on a measurement rather than on my arithmetic, which
has now been wrong about this in both directions.

The new variable is the calibration margin. The frozen scale is the warmup's
observed amax times `SANA_FP8_MARGIN`, which has been 2.0 since round 20 on
the argument that a factor of two moves float8_e4m3's 4-bit exponent and costs
none of its three mantissa bits. That argument is wrong in the direction that
matters here: the exponent is per value, not per tensor. Scaling the whole
tensor by two halves every value's position in the representable range, so
every activation that is not near the top of the range loses a mantissa step.
Two is a full extra bit of headroom on a three-bit mantissa, paid by every
value of every converted layer, against the possibility that some prompt other
than case 01 runs hotter than case 01 did.

It goes to 1.25. Anything above the calibrated amax still clamps rather than
overflowing, so the failure mode is bounded and gradual, and 1.25 is still
headroom - it is the difference between covering a 100% excursion and a 25%
one.

The quality budget is the reason this is worth a round. Mean LPIPS is 0.2455
against a 0.30 gate and the worst case is 0.3816 against 0.50; there is no
room left to buy speed with quality, so the remaining moves are the ones that
buy quality at no cost in speed. At 5.69x, a hundredth of mean LPIPS is worth
0.057 of reward. Expect the speed to be unchanged at 5.69x - not one kernel
changes - and the mean somewhere between 0.235 and 0.245, for a reward between
4.32 and 4.38.

Result: reward **4.06100**, speedup 5.085x (5.18 s), mean LPIPS 0.2014, worst
0.2926 - the best quality since round 16 and not the best reward. Artifacts:
`runs/round-28/`, job 19021412, source commit `0de1883`.

Two things came back, one of them uninvited.

The precision schedule works and does not pay. Against round 27, which is the
identical 129-layer build with no bf16 head, three bf16 evaluations cost 4.6%
of the wall clock and bought 0.0266 of mean LPIPS and 0.073 of worst-case. So
the head of the trajectory does carry a disproportionate share of the
quantization cost - three evaluations out of twenty hold a quarter of it - but
at 0.0058 of LPIPS per percent of speed against a break-even of 0.0075, it is a
losing trade by about a fifth. `BF16_WARMUP` ships at 0, and the finding is
kept: if the reward ever gets speed-rich and quality-poor, this is the lever
that buys quality back, and it is the only one found so far that does.

The uninvited part: the round was supposed to restore round 26's 259-layer
selection and it converted 129.

    round 26   2560 -> 2560   bf16 0.278 ms, float8 0.265 ms, 1.05x - convert
    round 28   2560 -> 2560   bf16 0.271 ms, float8 0.261 ms, 1.04x - keep bf16

Same six shapes, same node, same margin of 1.05, a hundredth apart, and that
hundredth moves 130 layers and 6.3% of the wall clock. The threshold sits
inside the measurement noise, which means the build has not been deterministic
since round 26 and the two rounds either side of it were comparing different
models for reasons that had nothing to do with what they were testing. Round
26's 259 layers and round 28's 129 were both coin flips.

That is worse than a wrong threshold. A wrong threshold is a decision I can
argue with; a threshold inside the noise is no decision at all, and it silently
re-randomizes the configuration on every round.

## Round 29 - take the coin flip out, and profile what actually ships

`SANA_FP8_RACE_MARGIN` goes to 0.95. Round 27 measured what the group at the
line is actually worth in the model - +6.3% of speed for +0.0176 of mean LPIPS,
a net +0.178 of reward - so a shape that merely ties in the race belongs in,
and the margin only has to keep out shapes that lose. At 0.95 every one of the
six shapes converts, including the single 2304 -> 2560 projection, and none of
them is within noise of the line. The selection becomes a constant rather than
a sample.

The second change is instrumentation. `fp8.tune` freezes the scales *after* the
warmup generation, and the denoiser profile is taken *during* it. So every
profile I have read since round 25 - including the one that chose round 26's
target and the per-NFE breakdown in round 27's entry - is of a build with the
amax reductions still in it, which is not the build being timed. The warmup now
profiles three forwards after `tune` returns, and prints them as `denoiser
after tune`. It costs three forwards inside the excluded warmup and it is the
only profile in this log that describes the shipped model.

Expect round 26's numbers back, plus one layer: speedup near 5.69, mean LPIPS
near 0.245, reward near 4.30. The value of the round is not the reward - it is
that the next twenty-one rounds get a stable configuration and a profile of it.

Result: reward **4.21575**, speedup 5.588x (4.71 s), mean LPIPS 0.2455, worst
0.3816. Artifacts: `runs/round-29/`, job 19021984, source commit `c8ceece`.

All six shapes converted at the 0.95 margin, 260 layers, and the mean LPIPS
came back as 0.24547132261651605 - round 26's value to every digit. So the
extra layer over round 26's 259 is the single 2304 -> 2560 projection, and it
changes nothing: it runs once per request on a row count that is not a multiple
of 16, so `Fp8Linear` falls back to the bf16 linear and the model is
numerically identical to round 26's. The selection is now a constant. The two
shapes that were flipping sit at 1.02x and 1.03x, ten times the noise away from
a 0.95 line.

The speed came back 1.8% short of round 26 - 5.588x against 5.689x on an
identical model - and the post-freeze probe says the same thing before my new
profiler block ever runs: 210.9 ms per forward here against 207.8 ms in round
26. That is node variance, and it is a useful calibration: run-to-run on this
harness is worth about 1.5%, so any future result under 2% is not a result.

## The profile of what actually ships

The new `denoiser after tune` table is the first one in this log taken on the
frozen-scale build. Per NFE, from three forwards at 211 ms each:

    _scaled_mm                                      95 ms   45%
    cuDNN softmax attention                         17 ms    8%
    linear-attention Triton templates               15 ms    7%
    layer norm + cast, fused                        12 ms    6%
    depth RMS-norm / aggregation logits              9 ms    4%
    standalone quantizing casts                      9 ms    4%
    cat                                              6 ms    3%
    add                                              5 ms    2%
    everything else                                 43 ms   20%

and the GEMM number reconciles exactly with the race: 0.988 ms for the fused
gate/up at 2560 -> 15360, 0.862 for `down_proj`, 0.550 for `qkv`, 0.273 each
for `proj` and the cross-attention query, is 2.95 ms a block and 94 ms over 32
blocks. The denoiser is now a GEMM at cuBLAS fp8 speed with 55% of an NFE of
other work around it, and there is no lever left on the GEMM itself: the only
faster kernel would be a Triton template, which is the one inductor cannot
render around a per-tensor scale.

## Round 30 - the 8.2% nobody has looked at

The denoise is 4.27 s of a 4.71 s request and the VAE decode is 0.39 s. Round
26's decode profile is entirely `aten::`: `cudnn_convolution`, then `add_`,
`div`, `copy_`, `cat`, `mean`, `pow`, `silu` and six flavours of
`elementwise_kernel`, each a separate pass over a tensor that is
(1, C, 121, 544, 960). Not one Triton kernel in it. The decoder was never
compiled.

`LTX2VideoResnetBlock3d.forward` gets `@maybe_compile`. It is the block that
holds all of that: two causal convolutions with a normalization, a modulation,
a SiLU and a residual add around them. The convolutions stay cuDNN - that is
what inductor will pick too - and everything between them collapses into one
pass. Compiling at block granularity rather than at the decoder's keeps the
graph small and the shape count to the handful of resolutions the decoder
actually runs.

The elementwise half of the decode is around 180 ms. Halving it is 90 ms of
4710, so 1.9% and about +0.08 of reward - which, against the 1.5% run-to-run
this round just measured, is barely more than noise. It is worth the round
anyway: it is the last untouched component, and if block-level compilation is
worth anything here it will show in the decode timer directly rather than in
the reward, since `[timing]` reports the decode separately.

Result: reward **4.34600** - best of the campaign - speedup 5.759x (4.57 s),
mean LPIPS 0.2455, worst 0.3816. Artifacts: `runs/round-30/`, job 19022616,
source commit `a51ed05`.

The decode went 387 ms -> 287 ms, a 26% cut, for a mean LPIPS of
0.24545660056173801 against round 29's 0.24547132261651605 - a change in the
fourteenth decimal place, which is the fused arithmetic reassociating and
nothing else. 100 ms of a 4.71 s request, +0.13 of reward.

Two things in the compile log are worth keeping. Inductor autotuned the
convolutions as well as the elementwise work, and cuDNN won every one of them:

    AUTOTUNE convolution(1x512x33x34x46, 512x512x3x3x3)
      convolution             0.5409 ms  100.0%
      triton_convolution3d_182 1.2623 ms   42.8%
      ...
      triton_convolution3d_184 8.2924 ms    6.5%

So the 209 ms of convolution in the old profile was already at its floor, and
all 100 ms came from the normalizations, modulations, SiLUs and residual adds
collapsing into the two kernels around each convolution. That also means the
decode has no second act: what is left is 210 ms of cuDNN convolution and about
75 ms of everything else.

And block granularity was the right unit. The decoder runs at four resolutions
behind two width tiles, so compiling at the decoder's granularity would have
been one enormous graph per shape; at the block's it is a handful of small ones
that inductor can autotune in the time available.

## Round 31 - the tiling the heuristics never measured

Round 30's profile of the shipped denoiser is flat below the GEMM. `_scaled_mm`
is 45% of an NFE and nothing can be done about it - it reconciles exactly with
the race, so it is running at cuBLAS fp8 speed. Everything else is generated
Triton: 13% in quantizing casts, 6% in the fused layer norm, 6% in the depth
normalization and aggregation logits, 7% in the linear-attention templates, 3%
in `cat`. No single item is worth chasing, and the run-to-run noise round 29
measured is 1.5%, so an item worth 2% is not distinguishable from nothing.

What is worth chasing is all of them at once. Every one of those kernels got
its block size and warp count from inductor's heuristics, which pick without
measuring. `coordinate_descent_tuning` searches those parameters per kernel
against a real benchmark, and it applies to roughly 40% of an NFE. It changes
no arithmetic, so the LPIPS should come back bit-identical - which is also the
check: if it does not, something has been reassociated and I want to know.

The cost is compile time, and compile time is the constraint that is actually
binding: round 30 used 519 s of an 840 s reservation. `check_all_directions`
stays off for that reason. If the round times out it costs one round and round
30 stands.

Result: **timed out** - no reward. `TimeoutExpired ... after 780 seconds`, job
19023075, source commit `9e4d09b`. The round is consumed and round 30 stands as
the best at 4.34600. Artifacts: `runs/round-31/`.

The log got 465 lines in before the kill, and it contains the answer to both
questions - whether the tiling search is worth anything, and what it costs.

It is worth a great deal. The `denoiser (3 of 50 NFE)` profile is the same
build, the same shapes and the same dynamic-scale path as round 30's:

    round 30   Self CUDA time total: 796.107 ms   265 ms / NFE
    round 31   Self CUDA time total: 740.416 ms   247 ms / NFE

7.0% off the denoiser, against a run-to-run noise of 1.5% - and this is the
*unfrozen* build, so it is a like-for-like comparison of the kernels alone.
Every one of those kernels is the same arithmetic with a different block size
and warp count. The heuristic was leaving 7% on the table in the 55% of an NFE
that is not the cuBLAS GEMM.

And the cost is not where I assumed. The last table in the log is the VAE
decode:

    Self CPU time total: 160.787 s
    Self CUDA time total:  29.610 s
    triton_red_fused_add_convolution_div_mean_pow_silu_s...   15545 calls
    triton_poi_fused_add_clone_div_mean_pow_silu_sqrt_2       11213 calls
    triton_red_fused_mean_pow_0                               12036 calls

A decode that runs in 287 ms took 29.6 s of GPU and 161 s of CPU, and the call
counts say why: those are the autotuner's benchmark iterations, not the
decode's. Round 30 gave `LTX2VideoResnetBlock3d` a `@maybe_compile`, the decoder
runs at four resolutions behind two width tiles, and coordinate descent searched
every kernel of every one of them. The decoder is 6% of the request and it ate
most of the overrun.

So the mistake was not the technique, it was the scope. I set a global config
flag at import and it applied to three graph sets - the dynamic-scale denoiser,
the frozen-scale denoiser, and the decoder - of which exactly one is ever timed.

## Round 32 - the same search, aimed only at what ships

`coordinate_descent_tuning` moves out of the import-time block and into
`runtime.set_coordinate_descent`, and `candidate.py` turns it on at one line:
immediately before `fp8.tune`, which freezes the activation scales, flips an
attribute every compiled region is guarded on, and forces the denoiser to
retrace. That retrace produces the kernels that run on all eight timed cases.
The warmup's dynamic-scale denoiser and all four of the decoder's resolutions
compile with it off, exactly as in round 30.

Two guards, because a timeout costs a whole round:

  - the search is skipped unless more than 300 s of the 780 s reservation are
    left when the recompile is reached - eight timed cases and their encodes
    need about 150 s, the search about 150 more - so the failure mode is round
    30's kernels and round 30's reward, not a zero;
  - the flag goes off again as soon as `tune` returns, so no timed step can
    trigger a search even if something recompiles late.

A `[clock]` line now prints elapsed seconds at five points - model resident,
warmup sampling, warmup decode, either side of the frozen recompile, warmup
complete. Round 31 had no idea how much of the reservation it had left and
spent the lot; this is the number that decides what is affordable for the rest
of the campaign.

If the 7% carries from the unfrozen build to the frozen one, the NFE goes 211 ms
-> 196 ms, the request 4.57 s -> 4.27 s, speedup 5.76x -> 6.16x, and the reward
4.35 -> 4.65 at unchanged LPIPS. It changes no arithmetic, so the mean LPIPS
should come back at 0.2455 to the digit; if it does not, a reassociation has
happened and that is worth knowing too.

Result: reward **4.47007** - best of the campaign - speedup 5.863x (4.49 s),
mean LPIPS 0.23762, worst 0.37676. Artifacts: `runs/round-32/`, job 19023782,
source commit `5bdf192`.

The scoping worked and the round came home, but not for the reason predicted,
and the `[clock]` line is why I know.

    [clock]   230.0 s  model resident
    [clock]   281.3 s  warmup sampling complete
    [clock]   328.1 s  warmup decode complete
    [clock]   431.9 s  before the frozen recompile
    [compile] coordinate-descent tuning on
    [fp8] frozen scales adopted: 227.4 ms -> 203.9 ms per forward
    [compile] coordinate-descent tuning off
    [clock]   464.4 s  after the frozen recompile
    [clock]   466.9 s  warmup complete

The search cost **32.5 seconds**. Round 31 timed out on 780 and I had budgeted
150 for this; the thing I was frightened of is a rounding error on the
reservation. Nearly half the wall clock - 230 s - is loading the checkpoint and
racing the six GEMM shapes, and the diagnostics between the decode and the
recompile are another 104 s.

And the gain was a third of what round 31 measured:

    denoiser (3 of 50 NFE), unfrozen   round 30  796.1 ms   round 32  794.8 ms
    denoiser after tune (3 forwards)   round 30  626.1 ms   round 32  616.7 ms

The unfrozen profile is unchanged to a tenth of a percent, which confirms the
scoping did what it said. The shipped build moved 1.5%, and the denoise wall
moved 1.9% - 4231 ms to 4149 ms. Against round 31's 7.0%.

The reason is in what `frozen` actually guards. Freezing flips an attribute on
`Fp8Linear`, so the regions that retrace are the regions that *contain* one.
Every region that does not - the depth normalization and aggregation, the
logit reduction, the cat, the residual adds, everything between the GEMMs that
is not adjacent to a float8 layer - was already compiled during the warmup
sampling with coordinate descent off, and it stays compiled. I tuned the half
of the denoiser that happens to sit next to a quantized matmul.

The other half of the reward came from somewhere I did not ask for it. Mean
LPIPS went 0.24546 to 0.23762 and worst 0.38165 to 0.37676. I wrote in the
hypothesis that the arithmetic does not change and the LPIPS should return to
the digit; it changed. A different block size is a different reduction order, so
the sums are reassociated, and this time the reassociation landed on the good
side of the reference by 0.0078 - worth +0.045 of reward on its own, about a
third of the round's gain. That is luck, not a technique, and the next retiling
could just as easily cost it.

## Round 33 - tune the other half

Coordinate descent moves to cover the whole denoiser: on after the text encode,
off immediately before the VAE decode, and on again for the frozen recompile
exactly as it is now. The decoder's four resolutions - the 161 s of CPU that
killed round 31 - stay excluded.

The cost is now a measured quantity rather than a fear. The frozen recompile
searched its regions in 32.5 s; the warmup sampling compiles the same denoiser
plus the regions that have no float8 layer in them, and round 32 ran that whole
phase, compile included, in 51.3 s. Budget 100 s and the warmup still finishes
near 570 s of 780. The deadline guard stays, with the wider margin the larger
search needs.

Round 31 measured the answer on exactly this configuration - 796 ms to 740 ms
per 3 NFE on the unfrozen build, 7.0% - so the prediction is not extrapolated:
the shipped build goes 616.7 ms to about 575 ms per 3 forwards, the denoise
4149 ms to about 3870 ms, speedup 5.86x to about 6.2x, reward to about 4.75 if
the LPIPS holds. It will not hold exactly, because retiling reassociates, and
which way it moves is the part I cannot predict.

Result: reward **4.73152** - best of the campaign - speedup 6.105x (4.32 s),
mean LPIPS 0.22506, worst 0.36095. Artifacts: `runs/round-33/`, job 19024308,
source commit `dfb9e40`.

The 7% was real and it reproduced exactly:

    denoiser (3 of 50 NFE), unfrozen   round 32  794.8 ms   round 33  736.9 ms
    denoiser after tune (3 forwards)   round 32  616.7 ms   round 33  594.6 ms
    denoise wall per case              round 32  4149 ms    round 33  3983 ms

Round 31 measured 740.4 ms on the unfrozen build with the search applied
globally; round 33 gets 736.9 ms with it applied to the denoiser alone. Same
number, and the decoder - the thing that cost round 31 the whole reservation -
contributed nothing to it.

Quality moved again, and again in the good direction: mean LPIPS 0.23762 to
0.22506, worst 0.37676 to 0.36095. That is 0.0126 of mean on top of round 32's
0.0078, so retiling has now given back 0.0204 of LPIPS across two rounds while
also making the model faster. I do not have an account of why it is
systematically favourable rather than a coin flip, and I am not going to invent
one: two rounds is two samples. What I can say is that the reference is the
unquantized bf16 model, so any change that happens to cancel part of the float8
error moves toward it, and there is no reason the heuristic tiling should have
been the arrangement that cancels least.

## The guard fired, and it was set from a guess

    [clock]   232.8 s  before the warmup sampling
    [clock]   356.2 s  warmup sampling complete      <- search: 72 s
    [clock]   507.0 s  before the frozen recompile
    [compile] only 273 s left - skipping the tiling search
    [clock]   517.4 s  after the frozen recompile
    [clock]   519.8 s  warmup complete

So round 33 tuned the regions round 32 did not, and skipped the regions round
32 did. The two rounds are complements. Round 32's float8 regions were worth
1.5% (626.1 -> 616.7 ms) and this round's everything-else was worth 5.0%
(626.1 -> 594.6 ms), and no round so far has had both.

The guard refused on 273 s because I wrote 300 into it before round 32 had
measured what the search costs. It costs 32.5 s. The threshold was three times
the number it was guarding.

## Round 34 - both halves

The frozen-recompile guard drops from 300 s to 150 s, which is the measured
32.5 s search plus the tail. Nothing else changes: the sampling search stays,
the decoder stays excluded, the sampling guard stays at 400 s.

The tail is now measured too - a `[clock]` line prints after each timed case -
because both guards are thresholds on how much reservation is left, and I have
been setting them from a guess about work I have never timed.

Expect the two effects to stack to about 585 ms per 3 forwards, the denoise
near 3920 ms, speedup near 6.2x. On quality I predict nothing.

Result: reward **4.87599** - best of the campaign - speedup 6.261x (4.21 s),
mean LPIPS 0.22131, worst 0.38095. Artifacts: `runs/round-34/`, job 19024980,
source commit `7ab499a`.

The two halves stack, and almost exactly additively:

    denoiser after tune (3 forwards)   round 30  626.1 ms
                                       round 32  616.7 ms   float8 regions only
                                       round 33  594.6 ms   everything else only
                                       round 34  563.8 ms   both
    frozen forward, from `fp8.tune`    round 33  198.0 ms   round 34  189.6 ms

10.0% off the shipped denoiser since round 30, for no change to weights,
schedule, precision or numerics - only the block sizes and warp counts the
kernels are compiled with. Round 30's entry said the profile below the GEMM was
flat and that no single item in it was worth chasing. That was true and it was
the wrong conclusion: the right unit of attack was not an item, it was the
compiler's choice of tiling across all of them at once.

Quality: mean LPIPS 0.22506 -> 0.22131, worst 0.36095 -> 0.38095. The mean
improved and the worst case got worse, which is the first time the two have
moved apart since retiling started. Neither gate is close - 0.30 and 0.50 - but
the worst case is the one with the shorter fuse, and it is now moving on a knob
that is supposed to be numerically neutral. Worth watching, not yet worth a
round.

The guard behaved:

    [clock]   512.7 s  before the frozen recompile
    [clock]   545.7 s  after the frozen recompile     <- the search, 33.0 s
    [clock]   548.0 s  warmup complete
    [clock]   551.9 s  timed case 1 returned
    ...

33.0 s against round 32's 32.5 s, so the cost of that search is now a known
constant, and the timed cases run 4.6 s apart, which puts the whole post-warmup
tail at about 40 s plus the LPIPS and the encodes.

## Round 35 - the search the heuristic starts from, and paying for it

`coordinate_descent_tuning` hill-climbs from whatever config the heuristic hands
it, and `max_autotune_pointwise` is what decides that starting config: without
it, inductor takes the first tiling its heuristic proposes for a pointwise or
reduction kernel; with it, it benchmarks the several it proposes and keeps the
fastest. Rounds 32 to 34 ran the hill-climb from an unmeasured starting point.
Both now turn on and off together, on the same denoiser-only scope.

It costs more compile time, and the reservation has to come from somewhere, so
the diagnostics block goes. Round 34's clock puts 107 s between the warmup
decode and the frozen recompile, and all of it is probing: a second VAE decode
under a launch tracer, the four GEMM shapes benchmarked again, the attention
backends benchmarked again. Those questions were answered in rounds 12 to 26.
`_report_compilation` stays, because it is a counter dump and it is free.

So the round is one change to the build and one to the instrumentation around
it, and the second is there to pay for the first. If the search finds nothing,
the round costs the 107 s of probes I no longer read and returns round 34's
number; if the deadline guards fire, they fire on a budget 107 s larger than
the one they were calibrated against.

Result: reward **4.80162** - below round 34's 4.87599. Speedup 6.266x (4.21 s),
mean LPIPS 0.23380, worst 0.39608. Artifacts: `runs/round-35/`, job 19025526,
source commit `5958296`.

`max_autotune_pointwise` is worth nothing on speed and it cost quality:

                       round 34            round 35
    unfrozen denoiser  731.1 ms            730.8 ms
    after tune         563.8 ms            570.7 ms
    frozen forward     189.6 ms            189.0 ms
    speedup            6.2608x             6.2662x
    mean LPIPS         0.22131             0.23380
    worst LPIPS        0.38095             0.39608

The speed is the same to a twentieth of a percent, in four independent places.
Benchmarking the heuristic's candidate tilings before hill-climbing from them
does not find a better summit than hill-climbing from the first one; the search
converges to the same place. It did not even cost much compile time - 123.5 s
against 130.7 s for the sampling search.

## The quality movement is a lottery, and this round is the control

Rounds 32, 33 and 34 each moved mean LPIPS favourably - 0.24546, 0.23762,
0.22506, 0.22131 - and round 33's entry said that was two samples with no
account of the mechanism. Here is the third sample, and it goes the other way:
0.23380, on a build that is exactly as fast.

That settles it. Retiling reassociates the reductions, the reassociation shifts
the float8 error by a few thousandths of LPIPS, and the direction is not
something I am choosing. Across five retilings the mean sits at about 0.228
with a spread of roughly +/-0.006, which is +/-0.04 of reward - comparable to
the entire gain of a good systems round. Rounds 32 to 34 drew favourably three
times running and I let the entries imply the searches were buying quality.
They were not. Round 34's 0.22131 is a good draw on a build whose real quality
is about 0.228, and the 4.87599 in the ledger is that draw.

Two consequences. Retiling knobs must be judged on speed alone, because the
LPIPS they return is a sample of a distribution I do not control. And for the
rest of the campaign a reward difference under about 0.08 between two builds is
not a difference.

The diagnostics cut was smaller than advertised: 107 s became 76 s, not zero,
because what is left between the decode and the recompile is the profiler
tables, not the probes. `[diagnostics] skipped` is in the log and 30 s came
back.

## Round 36 - the exchange rate, measured where the model now is

`max_autotune_pointwise` goes back off - round 21's precedent: a neutral knob
does not ship - and the build returns to round 34's exactly.

The change is `EVALUATIONS` 20 -> 19. The sampler still takes all 50 requested
steps; one more of them reuses the preceding denoiser output instead of
evaluating the network, and `_equal_change_plan` re-bisects its threshold so the
19 evaluations are the ones that leave the smallest worst-case gap in
accumulated velocity change.

The arithmetic of the trade has moved since it was last looked at. Reward is
speedup x (1 - LPIPS), so one percent of speed is worth 0.0078 of LPIPS at the
current operating point. Dropping an evaluation is 4.6% of the request - the
denoise is 94% of it - which buys 0.036 of LPIPS of room. Round 13 measured the
cliff on the pre-float8 build: 20 evaluations at 0.178 and 16 at 0.305, an
average of 0.032 per evaluation over a stretch that is convex, so the first one
off the top should cost less than that and the trade should clear.

It is the first lossy lever tried since round 28, and it is the only one left
that is independent of the compiler work and multiplies with it. Mean LPIPS
would land near 0.25 against a 0.30 gate and worst near 0.42 against 0.50, so
there is room, but not much more of it: if this clears, 18 evaluations is not a
round I can take without the gate getting close.

Result: reward **4.89559** - best of the campaign, by 0.02 over round 34, which
is inside the band I declared last round and therefore not a win on its own
terms. Speedup 6.421x (4.11 s), mean LPIPS 0.23753, worst 0.37982. Artifacts:
`runs/round-36/`, job 19026064, source commit `a267dbd`.

The trade clears, and by much more than the reward says.

    evaluations   build            mean LPIPS   speedup
    20            round 34         0.22131      6.2608x
    20            round 35         0.23380      6.2662x
    19            round 36         0.23753      6.4207x

Against round 34 the nineteenth evaluation cost 0.0162 of mean LPIPS; against
round 35, which is the same build to within a refuted knob, it cost 0.0037. The
two 20-evaluation rounds differ by 0.0125 from each other, so the honest
statement is that the model sits near 0.228 at twenty evaluations and near
0.2375 at nineteen, and one evaluation costs about 0.010 - a third of round
13's 0.032 and a quarter of the 0.036 the speed buys.

Round 13 measured that 0.032 on the pre-float8 build, where mean LPIPS at twenty
evaluations was 0.178. It is 0.228 now, and the extra 0.05 is quantization
error, which does not accumulate along the trajectory the way schedule error
does. The two errors combine far short of additively, so every further
evaluation removed is cheaper against this build than round 13's curve implies.

## The lottery is on speed as well

Round 36's build is round 34's, one evaluation apart, and its shipped profile is
not:

    denoiser after tune (3 forwards)   round 34  563.8 ms
                                       round 35  570.7 ms
                                       round 36  584.7 ms

Same source, same search settings, same node type, three different answers
spanning 3.7%. The tiling search picks by benchmarking, the benchmarks are
noisy, and the summit it climbs to differs from run to run. That is why the
denoise only fell 2.4% (3871 -> 3779 ms) when one evaluation in twenty is 5%:
the round drew a per-forward cost 3.7% worse than round 34's and spent most of
the schedule's gain paying for it.

So the campaign's numbers carry two independent lotteries, +/-0.006 on mean
LPIPS and about +/-2% on speed, and a single round is one draw from their
product. Rounds 34 and 36 are 4.876 and 4.896 on builds whose difference is
worth more than 0.02. Nothing under about 0.10 separates two builds.

## Round 37 - one more evaluation off

`EVALUATIONS` 18. If an evaluation costs 0.010 of mean LPIPS the model lands
near 0.248 against a 0.30 gate, and the worst case - which has tracked the mean
at about 1.6x all campaign - near 0.40 against 0.50. Both have room.

The speed is 5.3% of the denoise, which is 94% of the request, so 5.0% of the
total before the lottery takes its cut. Expect speedup near 6.74x and reward
near 5.04, and expect the draw to move it by 0.10 either way.

The ladder has two more rungs in it at most. At seventeen the mean would be near
0.27 and the worst near 0.43; at sixteen round 13 measured 0.305 on a build that
started 0.05 lower, and the mean gate is 0.30. Seventeen is a round I can take.
Sixteen is not, and I would rather spend the tail of the campaign drawing again
from a configuration that passes than find the cliff with a zero.

Result: reward **5.12609** - best of the campaign, and the first move since
round 33 that is larger than the lottery. Speedup 6.858x (3.85 s), mean LPIPS
0.25252, worst 0.38235. Artifacts: `runs/round-37/`, job 19026452, source
commit `7f7a24f`.

    evaluations   mean LPIPS   worst LPIPS   speedup    reward
    20            0.22131      0.38095       6.2608x    4.87599
    20            0.23380      0.39608       6.2662x    4.80162
    19            0.23753      0.37982       6.4207x    4.89559
    18            0.25252      0.38235       6.8584x    5.12609

The eighteenth evaluation cost 0.0150 of mean LPIPS where the nineteenth cost
0.010, so the curve is steepening as round 13 said it would, and both are far
under the 0.036 that 4.6% of speed buys. The speed came in at 6.8% rather than
5% because the tiling draw was a good one - 576.5 ms per 3 forwards against
round 36's 584.7 - which is the lottery working in the round's favour for once.

The number worth noticing is the worst case: 0.38095, 0.39608, 0.37982,
0.38235. Four rounds, two evaluations removed, and it has not moved outside its
own noise. The schedule error is being spent evenly across the eight prompts,
so the binding gate is the mean at 0.30 and not the worst at 0.50 - which is
the opposite of what I assumed when I started down this ladder, and it is worth
saying because it decides where the ladder stops.

## Round 38 - seventeen, and that is the last rung

`EVALUATIONS` 17. At the observed 0.010 then 0.015, the seventeenth should cost
about 0.020 and land the mean near 0.272, against a gate of 0.30 and a lottery
of +/-0.006. Speed is 5.5% of the request, so expect near 7.2x and a reward
near 5.27.

Sixteen would land near 0.295. That is not a rung, it is a coin flip on a zero,
and the campaign has twelve rounds left and a 5.13 in the ledger. The ladder
stops at seventeen whatever this round returns.

What is left after it is small and I should say so plainly. Per NFE the shipped
denoiser is 45% cuBLAS float8 GEMM at its floor, 15% attention across the eight
softmax anchors and twenty-four linear layers, and the rest generated Triton
that three rounds of tiling search have already taken 10% out of. The decode is
283 ms of which round 30 measured 209 in cuDNN convolution, also at its floor.
The remaining ideas are `coordinate_descent_check_all_directions`, worth
perhaps one to three percent, and the decoder's own tiling search, worth under
one. Beyond those the honest use of a round is another draw from the two
lotteries this build carries.

Result: reward **5.15936** - nominally the best of the campaign, by 0.033 over
round 37, which is inside the lottery and therefore a tie. Speedup 7.217x
(3.65 s), mean LPIPS 0.28508, worst 0.40261, gate passed. Artifacts:
`runs/round-38/`, job 19026760, source commit `2560736`.

    evaluations   mean LPIPS   cost of the rung   worst LPIPS   speedup    reward
    20            0.228 (2)    -                  0.388 (2)     6.26x      4.876 / 4.802
    19            0.23753      0.010              0.37982       6.4207x    4.89559
    18            0.25252      0.015              0.38235       6.8584x    5.12609
    17            0.28508      0.033              0.40261       7.2165x    5.15936

The seventeenth evaluation cost 0.033, not the 0.020 I predicted from
extrapolating 0.010 and 0.015. The cliff round 13 described is where round 13
said it was; I had been reading the shallow part of a convex curve and treating
the slope as if it continued.

So the ladder is over, and it is over one rung before the gate rather than at
it. Mean LPIPS 0.28508 against a limit of 0.30 is 0.015 of margin on a quantity
whose run-to-run spread this campaign has measured at +/-0.006. That is not a
margin, it is a one-in-several chance of scoring zero, for a reward that ties
what eighteen evaluations already returns. Seventeen bought 5.2% of speed for
0.033 of LPIPS - 0.0063 per percent against a break-even of 0.0078 - so it is
still a winning trade on paper, and it is the first rung where the trade got
close enough to the line that the variance matters more than the mean.

Eighteen evaluations is where the build ships: 5.12609, mean 0.25252, and
0.047 of gate margin.

## Round 39 - the one lever left that is bigger than the noise

Per NFE the shipped denoiser is 45% cuBLAS float8 GEMM. Round 29 called that a
floor and stopped looking. It is not quite a floor: the MLP's fused gate/up
projection is 2 x 16320 x 2560 x 15360 = 1.283 TFLOP in 0.962 ms, which is 1334
TFLOP/s against this card's 1979 dense float8 - 67% of peak. A Triton template
that fuses its epilogue would plausibly reach the mid-seventies, and 10% of 45%
is 4.5% of an NFE, which is the first thing in twenty rounds that is bigger
than the +/-2% tiling lottery.

Inductor cannot render that template around this build's scale. The reason is
in `_use_cublas_for_scaled_mm`: the template loads its epilogue inputs at the
output's two index symbols and `FixedLayout.indexer` asserts
`len(index) == len(stride)`, which a rank-1 per-tensor scale cannot satisfy.
Rounds 19 to 22 lost whole regions to eager over it and round 23 installed the
cuBLAS-only redirect to stop the bleeding. That docstring also rules out the
obvious dodge - reshaping the per-tensor scale to (1, 1) would pass the assert
and then be read as per-row by the same template, indexing one element with row
indices, which is a wrong answer rather than a failure.

The scale that does satisfy the assert honestly is a real per-row scale, which
is `SANA_FP8=row`, the mode round 18 ran and the only mode that has never
printed a lowering failure. So the round is: per-row activation scales, and the
cuBLAS redirect lifted so the Triton template competes with cuBLAS at every
shape under the autotuner that has been in the build since round 12.

The two changes have to go together. The redirect exists only because
per-tensor cannot render; per-row without lifting it would be the accuracy cost
with none of the upside.

What it costs is the frozen scale. A per-row scale is a property of the rows in
front of the layer, not of the layer, so it cannot be calibrated and frozen -
`freeze` skips row layers by construction - and rounds 25 to 30 measured the
frozen per-tensor scale at 4 to 9% per forward. So this round spends a known 5%
for an uncertain 4.5%, and wins only if the template also absorbs the
quantizing cast that the frozen path currently folds into its producer.

It could easily lose. It is worth a round because it is the last hypothesis on
the list whose upside clears the noise floor, because per-row is strictly more
accurate than per-tensor and some of the cost may come back as LPIPS, and
because twelve rounds remain with a 5.16 and a 5.13 already in the ledger.

A guard goes in with it: a one-shot probe runs `_scaled_mm` with row scaling
and `use_fast_accum` before any layer is converted - that combination has never
run in this campaign - and falls back to per-tensor with the redirect
reinstalled if it raises. `EVALUATIONS` returns to 18.

Result: reward **4.94809** - below round 37's 5.12609 on the identical
schedule. Speedup 6.435x (4.11 s), mean LPIPS 0.23109, worst 0.29590.
Artifacts: `runs/round-39/`, job 19027554, source commit `3ae1a49`.

The mechanism worked exactly as designed and the hypothesis is refuted by what
it produced. The template rendered, entered the autotuner at every shape, and
lost every one:

    AUTOTUNE scaled_mm(16320x2560, 2560x15360, 16320x1, 1x15360, 15360)
      _scaled_mm    0.9459 ms 100.0%
      triton_mm_402 1.9848 ms  47.7%  SCALING_ROWWISE=True, USE_FAST_ACCUM=True
      triton_mm_401 2.0217 ms  46.8%
      triton_mm_400 2.1431 ms  44.1%

    AUTOTUNE scaled_mm(16320x10240, 10240x2560, 16320x1, 1x2560, 2560)
      _scaled_mm    0.6552 ms 100.0%
      triton_mm_494 0.8350 ms  78.5%

Rank-2 scales are indeed what the template needs - the `16320x1, 1x15360` in
the signature is the assert being satisfied - and `USE_FAST_ACCUM=True` in the
winning config line shows fast accumulation was never the gate either. cuBLAS
is 2.1x ahead of the best Triton template on the MLP shape and 1.27x ahead on
`down_proj`. The 67%-of-peak figure was real and it is not headroom that a
generated template can take; it is what an fp8 GEMM of this shape costs.

That closes the question rounds 19 to 23 left open. The redirect was never
costing anything at the GEMM, and now there is a measurement saying so rather
than an argument.

## What per-row actually costs, and what it buys

The GEMM itself is not the cost. Per-row `_scaled_mm` at the big shape is
0.9459 ms against the per-tensor race's 0.962 - the same kernel. The 6.2% comes
entirely from the quantizer:

    triton_red_fused__scaled_mm__to_copy_abs_amax_clamp_...   6.56%
    triton_red_fused__scaled_mm__to_copy_abs_add_amax_cl...   3.88%
    triton_red_fused__scaled_mm__to_copy_abs_add_amax_cl...   3.17%
    triton_red_fused__scaled_mm__to_copy__unsafe_view_ab...   1.83%
    ... about 17.7% of an NFE in fused quantize-and-amax

against about 13% for the frozen per-tensor cast. A frozen scale removes a
reduction; a per-row scale is a reduction that cannot be removed, because it
belongs to the rows in front of the layer.

What it buys is quality, and more than I expected:

                  mean LPIPS   worst LPIPS   speedup
    per-tensor    0.25252      0.38235       6.8584x
    per-row       0.23109      0.29590       6.4350x

0.021 of mean and 0.086 of worst case. At 6.2% of speed that is 0.0034 of LPIPS
per percent against a break-even of 0.0078, so it is a losing trade by more
than half, and the reward says so: 4.948 against 5.126. But the worst case
falling from 0.382 to 0.296 is the largest single quality movement in the
campaign, and it says the per-tensor scale is costing one or two prompts far
more than the average - which is the only thing left that might make the
schedule ladder's lower rungs reachable.

## Round 40 - back to the best build, and what the rest of the campaign is for

Per-tensor returns, the cuBLAS redirect goes back on, and `EVALUATIONS` goes to
17 rather than 18.

That last one is a change of strategy, not of technique. Round 38's 5.15936 is
banked: a round that fails the gate scores zero for that round and does not
touch it. So for the remaining rounds the downside of the riskier schedule is
already paid and the upside is not, and the right configuration to run is the
one with the highest mean rather than the one with the most margin. Seventeen
evaluations at mean 0.285 clears the gate most of the time and returns about
5.16; eighteen returns about 5.13 with certainty. The banked best makes the
first strictly better.

The round also carries `coordinate_descent_check_all_directions`, scoped to the
frozen-scale recompile alone - the one compile whose cost this campaign has
measured, twice, at 33 s. It is worth perhaps a percent, which is inside the
lottery, so it will not be distinguishable on its own; it is there because a
round that is a draw anyway may as well carry a knob.

I should say plainly what the last eleven rounds are. Per NFE the denoiser is
45% cuBLAS float8 GEMM now measured against every alternative inductor can
generate, 15% attention on backends that were raced in round 12, and the rest
generated Triton that the tiling search has already taken 10% out of. The
decode is 209 ms of cuDNN convolution that autotuning confirmed is at its floor
plus 75 ms of everything else. The schedule is one rung from its gate. There is
no remaining item whose plausible gain exceeds the +/-2% speed and +/-0.006
LPIPS that this build returns run to run. What is left is a small number of
untested knobs, each worth less than the noise, and draws.

Result: reward **5.16828** - nominally the best, by 0.009 over round 38, which
is a tie. Speedup 7.300x (3.61 s), mean LPIPS 0.29213, worst 0.40843, gate
passed. Artifacts: `runs/round-40/`, job 19028112, source commit `3df84d6`.

The shipped profile went 574.6 ms to 563.7 per 3 forwards, 1.9%, and
`check_all_directions` cost about 60 s on top of the 33 s the plain search
takes - the frozen recompile ran 479.0 s to about 580 s against round 38's 40.
1.9% is inside the +/-2% lottery, so the knob is not separable from the draw,
which is what the hypothesis said it would be. It stays because it is not
expensive and it did not lose.

The number that matters in this round is not the reward. Mean LPIPS drew
0.29213 against a gate of 0.30. Round 38 drew 0.28508 on the same schedule. Two
draws at seventeen evaluations, 0.0071 apart, and the second one landed 0.008
from a zero.

## Round 41 - where in the model the quality actually is

Round 39 measured per-row scaling at 0.021 of mean LPIPS and 0.086 of worst
case for 6.2% of the wall clock: 0.0034 per percent against a 0.0078
break-even, a losing trade taken as a whole. But it need not be taken as a
whole, because the cost is not spread the way the layers are.

A per-row scale is an amax reduction over the layer's input, so a layer costs
what its input costs to read. At 16320 rows, `down_proj` reads 10240 columns
and every other converted layer reads 2560 or 2304. One projection is about
57% of the per-row traffic in a block. If the quality is not concentrated in
that same projection, then buying per-row everywhere else costs about 2.7%
instead of 6.2%, and at 0.0034 per percent it only has to return half the
quality to turn the trade over.

So `ROW_PROJECTIONS` names `qkv`, `q_linear`, `gate_proj` and `up_proj`, and
`down_proj` keeps the per-tensor scale it can freeze. The schedule stays at 17
evaluations.

There are two ways to win and they are worth different amounts. If the mean
comes back near 0.28 at a cost of 2.7%, the reward is about 5.25 and the gate
margin doubles. If it comes back near 0.2925 unchanged, then the quality was
all in `down_proj`, the round costs 2.7% and returns about 5.02, and the next
round runs per-row on `down_proj` alone instead - which is the same experiment
with the complement.

Either answer localises something this campaign has only measured in
aggregate: which layer's activations the per-tensor scale is actually hurting.

Result: reward **4.99640**, gate passed. Speedup 7.0365x (3.755 s), mean LPIPS
0.28997, worst 0.42161. Artifacts: `runs/round-41/`, job 19029047, source
commit `fb738d1`. Per case: 0.16357, 0.32658, 0.24638, 0.33325, 0.22675,
0.38051, 0.42161, 0.22108.

The hypothesis named two outcomes in advance and the second one happened,
almost to the digit. It said "if it comes back near 0.2925 unchanged, then the
quality was all in `down_proj`, the round costs 2.7% and returns about 5.02".
The mean came back 0.28997 and the round returned 4.996.

Against round 40, which differs by this change and nothing else: the four
narrow-input projections per-row cost **3.60%** of the wall clock (7.2996 ->
7.0365) and returned **0.0022** of mean LPIPS, which is a third of the +/-0.006
draw, while making the worst case 0.013 *worse*. `qkv`, `q_linear`,
`gate_proj` and `up_proj` are not where the per-tensor scale hurts. They are
77% of the converted layers and they were free to fix, and fixing them bought
nothing.

By subtraction `down_proj` alone is the rest of round 39's per-row result:
about 2.6% of the wall clock for about 0.019 of mean LPIPS, which is 0.0073 of
LPIPS per percent against a 0.0078 break-even. The hypothesis said the
complement would be the next round. It will not be. The arithmetic already
answers it - it is a losing trade, by a little - and the answer does not unlock
anything downstream, because the rung it would buy margin for is 0.05 of LPIPS
away and not 0.019. With the best banked at 5.16828 a round that is neutral by
construction is dominated by any round that is not, since a failed round costs
only itself. `SANA_FP8_ROW` stays in the tree with its measurement written next
to it, defaulting to empty.

What the localisation is worth keeping: whatever the per-tensor scale costs in
quality, it costs it in the one projection whose input is the 10240-wide SwiGLU
intermediate. That is the activation with the widest dynamic range in the
block, and the one a single scalar fits worst. That is a fact about this model,
not about float8.

## What is left to buy, and where it is

The shipped build's denoise phase is 3278.7 ms over seventeen evaluations, or
192.9 ms each; `fp8.tune`'s own timer, which never sees the profiler, says
193 ms. The sampler arithmetic between the evaluations is therefore worth
essentially nothing, and the fifty DPM-Solver updates - thirty-three of them on
a reused velocity - do not appear in the wall clock at all. The forward is the
request: denoise is 91% of the case, the VAE decode 7.5%, the text encoder
1.4%.

Inside the forward, from the profile of the build that actually ships:

    aten::_scaled_mm                       49.5%
    cuDNN attention, 8 anchor layers        8.7%
    the float8 cast, fused into its producer 8.8%
    RMSNorm and the AdaLN modulation         4.5%
    depth aggregation (softmax, addcmul)     3.3%
    aten::cat                                3.2%
    residual adds                            2.7%
    flash attention, 64 cross-attention      2.0%
    everything else                         17.3%

Half of it is a GEMM that rounds 15, 23, 24 and 26 established is already at
the cuBLAS optimum, and that round 35 established no rendered Triton template
can reach. A tenth is attention at about 640 TFLOP/s, which is 65% of the
card's bf16 peak on a shape with no sparsity to exploit that the gate would
survive. The remaining 40% is memory-bound glue that inductor has already fused
into the kernels that produce it. Every lever left in that list is worth 1-3%,
and the speed draw alone is +/-2%.

Meanwhile reward is speedup x (1 - LPIPS). At 7.3x, one hundredth of mean LPIPS
is worth 0.073 of reward - more than a whole percent of speed, which is worth
0.052. Quality bought at zero speed cost is the best trade on the board, and
there is exactly one place this campaign has never looked for it.

## Round 42 - the placement nobody has measured

Every round since 11 has shipped one reuse schedule: bisect a threshold on
accumulated `VELOCITY_CHANGE` until seventeen calls survive. The ladder rounds
- 13, 37, 38 - moved the *number* of evaluations. No round has ever moved their
*placement*, and no round has ever measured a second rule against the first.

Equal accumulated change is a statement about the local error a reused call
commits. The score is a statement about the distance between the finished
latent and the reference one, and the two differ by how much of a local error
survives to the end. The steps after an error re-predict the sample from
wherever the perturbed trajectory has arrived, so an early error is partly
integrated away and a late one is not. The rule as written prices a hundredth
of change at step 3 and a hundredth at step 47 the same.

So: a linear ramp on the change curve before the same bisection runs. Bias 0 is
the rule that has been shipping; bias 2 counts the last step three times the
first; bias -0.5 leans the other way, in case the prior is backwards. At
seventeen evaluations the family is genuinely different - bias 0 evaluates
0,1,2,3 and then leaves gaps of five at 28-43; bias 4 evaluates 0,1,3,6,10 and
never leaves a gap over four after step 37.

And the round measures the family instead of guessing at it, because the
measurement is free. The warmup generation is excluded from latency, runs all
fifty evaluations, and has been ending with about 160 s of the reservation
unused - round 40's warmup finished at 580.6 s of 780 and its last timed case
returned at 613. That fifty-evaluation latent is precisely the trajectory the
timed cases are trying to reproduce. After `fp8.tune` has frozen the scales and
the kernels are final, one seventeen-evaluation trajectory costs 3.4 s. Seven
of them price the whole family and both neighbouring rungs of the ladder for
about 24 s, against a guard that refuses to start a candidate with less than
150 s of reservation left. The best seventeen-evaluation plan is pinned and
every timed case runs it.

Latent distance is a proxy for LPIPS and a known-imperfect one: a latent metric
can be blind to exactly the high-frequency detail LPIPS reads. That is what the
two off-rung candidates are for. Eighteen and sixteen evaluations bracket
seventeen on a ladder whose LPIPS is already measured - 0.25252 at eighteen,
0.28508 at seventeen - so the table this round prints can be converted into
LPIPS from the next round onwards instead of trusted in the abstract. That
conversion is the round's real product: it makes every later schedule question
answerable inside a round rather than costing one.

The timed build is otherwise round 40's exactly, `SANA_FP8_ROW` back to empty.
Two outcomes, again named before the run. If placement is worth nothing, every
seventeen-evaluation candidate scores within a few per cent of the control,
bias 0 ships, and the round is a free re-roll of the banked best at about 5.17.
If the ramp is worth 0.01 of mean LPIPS it returns about 5.24 at identical
speed, and the two rungs say how much more there is in the same direction.

Result: reward **5.11967**, gate passed. Speedup 7.2163x (3.657 s), mean LPIPS
0.29055, worst 0.39313. Artifacts: `runs/round-42/`, job 19030243, source
commit `948c794`. The search chose the control, so the timed build was round
40's exactly - this is a third independent draw of one build:

    round 38   7.2165x   0.28508   5.15936
    round 40   7.2996x   0.29213   5.16828
    round 42   7.2163x   0.29055   5.11967

0.049 of reward between the best and the worst of three runs of identical
source. That is the lottery, measured three times now, and it is the resolution
limit on everything below.

The search itself worked: seven trajectories, 25 s, every one priced, and the
round finished its last timed case at 421.5 s of 780 because the weights came
off page cache this time. The table:

    nfe=17 bias=0      0.42303      <- the rule rounds 11-41 shipped
    nfe=17 bias=-0.5   0.44374
    nfe=17 bias=1      0.63695
    nfe=17 bias=2      0.63945
    nfe=17 bias=4      0.56838
    nfe=18 bias=0      0.43690
    nfe=16 bias=0      0.63870

**The tail ramp is refuted.** Leaning the schedule late costs 50% more relative
latent distance, and leaning it early costs too. Equal accumulated change is a
sharp local optimum in that direction, and the prior round 42's hypothesis was
built on - that a late error survives where an early one is integrated away -
is wrong for this trajectory. The reason is visible in the curve it was
reweighting: `VELOCITY_CHANGE` already rises into the last four steps, 0.0369,
0.0423, 0.0522, 0.0784, 0.1720, so equal change already spends four of its
seventeen evaluations after step 43. The ramp does not add evaluations to the
tail, it takes them off the head, where the curve is 0.41 and 0.18 and there is
nothing to spare.

**And the metric failed its own control.** Eighteen evaluations scored 0.43690,
*worse* than seventeen's 0.42303, where the gate has measured 0.25252 at
eighteen (round 37) and 0.28508 at seventeen (round 38). A relative latent
distance ranked a 0.033 difference in LPIPS backwards.

That is exactly why the two off-rung candidates were in the table, and it is
the round's real result. The ramp refutation survives it - bias 1 and 2 sit
0.21 from the control, fifteen times the 0.014 the metric got backwards - but
nothing at the scale the campaign actually needs can be read off a latent
number. Which leaves the question of what can.

## Round 43 - stop approximating the score and compute it

`reward.json` says what the harness measures: lpips 0.1.4, AlexNet, frames
reduced by their mean, the video clamped to [-1, 1] and fed in unresized. All
of that is available inside the round. The warmup generation decodes its
fifty-evaluation latent into a video and throws it away; that video is the
reference every timed case is trying to reproduce, up to the float8 precision
every candidate shares. So the search stops scoring in latent space and scores
in the gate's own metric against the warmup's own decode, for a decode that
costs 0.28 s and a metric pass that costs a fraction of that.

The round then has to establish whether that number can be trusted, and it has
a way to: the eighteenth rung can never ship, and it is priced anyway, because
the gate has already measured that eighteen beats seventeen. If the metric
reproduces that ordering it is allowed to choose. If it does not, it prints the
contradiction and ships the control. Round 42 is why that check exists and
`test_ship_schedule` is why it will still be there in round 50.

Given a metric that ranks, the round asks three questions at once:

*Placement, on the axis the ramp was not.* `CHANGE_EXPONENT` bisects on
`VELOCITY_CHANGE ** p` instead of on the curve itself. Exponent 1 is equal
change; exponent 0 is uniform spacing, gaps of four everywhere; exponent 2
spends everything where the field moves fastest and leaves gaps of eight in the
middle. The ramp asked *when* an error hurts. This asks *how superlinear* the
cost of a gap is in a region that is already moving - a different question, and
one the equal-change rule answers by assuming the answer is exactly linear.

*The reuse rule.* Round 8 measured velocity extrapolation against holding the
last velocity constant and it cost 0.194 -> 0.241 of mean LPIPS. That was a
uniform schedule at a different count on a bf16 build. It is priced again here
on the control's own plan, so the reuse rule is isolated from the placement,
and it costs 4.5 s to find out whether the refutation still holds where the
model now is.

*The rung below.* Sixteen evaluations is 5.6% faster, which at this reward is
worth 0.044 of LPIPS. Round 38 established seventeen is the last rung the
0.30 gate survives - but at one placement. Four exponents at sixteen ask
whether any placement gets it under the ceiling.

The decision rule is written down rather than eyeballed. The metric is trusted
for differences and not for absolutes, so the control's distance from the
0.28925 the gate has reported for it - 0.28508, 0.29213, 0.29055 over rounds
38, 40 and 42 - is subtracted from every candidate. Predicted reward is
26.40 / (0.329 + 0.193 x evaluations) x (1 - predicted mean), the time model
fitted to round 42's own 3.610 s. Nothing whose predicted mean is above 0.295
may ship, because the draw on the mean is +/-0.006 and the gate zeroes the
round at 0.30.

Twelve candidates, about 55 s, behind a guard that stops starting them with
less than 140 s of reservation left and prices the control and the ladder
anchor first so a late round still gets to decide. Three outcomes: if nothing
beats the control this is a fourth draw of the same build at about 5.15 and the
campaign leaves with an in-round oracle for every lossy question it has left;
if an exponent is worth 0.01 of mean LPIPS, about 5.24 at identical speed; if
any sixteen-evaluation placement lands under the ceiling, about 5.45.

Result: reward **5.19124** - the best of the campaign. Speedup 7.2729x
(3.627 s), mean LPIPS 0.28624, worst 0.40216. Artifacts: `runs/round-43/`, job
19031772, source commit `a38d9bb`. The search validated its metric, found the
metric could not rank the ladder, and shipped the control - so the timed path
was rounds 38, 40 and 42's exactly, and this is a fourth draw of one build:

    round 38   7.2165x   0.28508   5.15936
    round 40   7.2996x   0.29213   5.16828
    round 42   7.2163x   0.29055   5.11967
    round 43   7.2729x   0.28624   5.19124

0.072 of reward between the best and worst of four runs of identical source.
The lottery is larger than three draws suggested and it is larger than every
lever this campaign has left. That is a result about what the remaining rounds
are worth, and it is worth writing down before the table below.

Twelve trajectories, 47 s, each decoded and scored in the gate's own metric
against the warmup's own fifty-evaluation decode:

    nfe=17 p=1                0.12082   latent 0.37441   <- control
    nfe=18 p=1                0.12418   latent 0.39868
    nfe=17 p=2                0.12737   latent 0.39343
    nfe=17 p=1.5              0.13214   latent 0.40792
    nfe=17 p=1 extrapolated   0.17999   latent 0.55967
    nfe=17 p=0.75             0.22595   latent 0.63254
    nfe=16 p=0.75             0.22643   latent 0.63465
    nfe=16 p=1                0.22681   latent 0.63275
    nfe=17 p=0.5              0.22713   latent 0.63383
    nfe=16 p=0.25             0.32681   latent 0.80031
    nfe=16 p=0.5              0.32796   latent 0.79595
    nfe=17 p=0.25             0.32877   latent 0.79494

Four things fall out of it.

**Equal change wins the second axis too.** Flattening the curve toward uniform
spacing is catastrophic - 0.226 at exponent 0.75, 0.329 at 0.25 - and
sharpening it is mildly bad, 0.127 at exponent 2. Two orthogonal families have
now been priced around the rule rounds 11-41 shipped and it is the optimum of
both. A wrong placement at seventeen evaluations costs exactly what losing an
evaluation costs: 17 at p=0.75 scores 0.22595 and 16 at p=1 scores 0.22681.

**Extrapolation is refuted again**, on the control's own plan so the reuse rule
is isolated from the placement, in the regime the model is actually in: 0.180
against 0.121. Round 8's answer holds four builds and thirty-five rounds later.

**Sixteen evaluations is a cliff, not a rung.** 0.2268 against 0.1208, nearly
double, for 5.6% of speed. Round 38's conclusion stands.

**And the latent metric was never the problem.** Round 42's table said eighteen
evaluations were worse than seventeen and I called that a failure of the proxy.
The gate's own metric says the same thing - 0.12418 against 0.12082 - so the
search refused to choose, which is what it was built to do. But the refusal
was written around an ordering I had assumed and never checked.

## Case 06, and why the warmup can never see it

Rounds 37 and 38 differ only in the evaluation count, and their per-case LPIPS
is on disk. Eighteen minus seventeen, case by case:

    +0.00440  -0.05045  -0.02040  +0.00465  +0.03377  -0.21337  -0.02026  +0.00113

Case 01 at eighteen evaluations scored 0.17124 against 0.16685 at seventeen.
The gate itself says eighteen is worse for case 01, by 0.0044, and round 43's
in-round metric predicted 0.0034. **The metric is accurate.** What was wrong is
the ordering I asserted it had to reproduce, which was the eight-case mean.

Because the eight-case mean is one case. Case 06 goes 0.1476 at nineteen
evaluations, 0.1599 at eighteen, and then 0.3733, 0.3816, 0.3741 at seventeen
across three independent rounds. That is a deterministic cliff, not a draw, and
it is the whole 0.033 the ladder appeared to measure. Four of the eight cases
are *better* at seventeen.

The mechanism is that the bisection is not nested. Seventeen evaluates
`0,1,2,3,5,8,...`; eighteen evaluates `0,1,2,3,4,6,8,...` and the tails are
identical. The extra evaluation is spent at the head, and step 5 is given up to
buy steps 4 and 6. One case's trajectory turns on that, and there is no
monotonicity in the count to appeal to.

So the in-round oracle is accurate, it is accurate about case 01, and case 01
is the one case whose answer does not matter. The mean is set by a cliff on a
prompt the warmup never sees and the round cannot sample. Further schedule work
would be tuning on the wrong prompt, so the remaining rounds stop asking the
quality question.

## Round 44 - the last three percent that is not a draw

What is left is speed, and the profile of the shipped build says where. Three
changes, bundled, because each is below the +/-2% speed draw on its own and
none of them is separable from it in a single round - but each has its own
kernel in the profile, so the log attributes them individually even though the
reward cannot.

*The cross-attention row join, 3.2% of the denoiser.* The two
classifier-free-guidance rows attend over captions of different lengths, so
cross-attention runs as two calls and the results are rejoined with
`torch.cat`. That lands in `CatArrayBatchedCopy` at 189.6 us a call for 167 MB
of traffic - 884 GB/s, a quarter of what copying two contiguous 42 MB tensors
should run at. It is a kernel written for many small inputs handed two large
ones. A preallocated buffer and one `copy_` per row moves the same bytes
through the elementwise copy kernel, and `test_glue` asserts the result is
bit-identical.

*The depth aggregation, 1.07%.* `einsum("nbt,nbtd->btd")` over a slice of the
value buffer lowers to a batched gemv: 503 us a call for at most 418 MB, which
is 660 GB/s. Four calls a forward, one at the head of each block group and one
for the final aggregation. Spelled as a reduction, inductor emits one pass and
folds the softmax into it.

*The decoder's own tiling search.* Round 31 spent 161 s of CPU searching the
decoder's four resolutions and lost the round to the harness timeout, and every
round since has turned the search off there without asking. The decode is
277 ms of round 43's 3.61 s - 7.7% of the request - and rounds 42 and 43
reached that line at 144 s of 780 because the weights came off page cache. 161 s
is affordable out of 636 and is not out of 438, so the code asks the clock.

If all three land it is about 3% and a reward near 5.35. If only the join lands
it is 2% and about 5.29. If none of them does, it is a fifth draw of a build
whose four draws span 0.072, which given the size of that number is not the
worst way to spend a round.

Result: reward **5.20987** - the best of the campaign. Speedup 7.3323x
(3.602 s), mean LPIPS 0.28952, worst 0.39339, gate passed. Artifacts:
`runs/round-44/`, job 19032616, source commit `e7a0802`.

Two of the three changes landed, the third never ran, and the profile says
which is which without having to infer it from the reward.

    kernel row (3 forwards)                  round 43        round 44
    CatArrayBatchedCopy                   17.902 ms  96     -
    elementwise_kernel<128, 4>                 -         12.430 ms 194
    gemv2N_kernel<bfloat16>                5.949 ms  12     -
    Self CUDA time total                    556.573 ms      551.890 ms

**The row join is real but it is worth a third of what the cat cost.** The
187 us `CatArrayBatchedCopy` is gone and 192 `copy_` calls at 64 us have taken
its place: 17.90 ms becomes 12.43 ms. The bytes still have to move. What the
rewrite bought was the difference between a kernel written for many small
inputs at 884 GB/s and one copy per row at 2.6 TB/s, and the only way to buy
the rest is to not make the copy at all - which means giving the two
cross-attention calls a preallocated output, and
`scaled_dot_product_attention` has no `out=`.

**The depth aggregation is gone outright.** No `gemv2N_kernel` at any depth in
the round-44 table, and the `aten::bmm` row that dispatched to it is gone with
it. Four 500 us gemvs a forward folded into kernels that were already running.

**The decoder's tiling search never fired**, and the guard is the reason. The
round came up cold - the model was resident at 229.6 s where rounds 42 and 43
had it at 144 s, because the weights came off disk and not page cache - so the
warmup reached the decode with 416 s of reservation left against a 560 s
reserve, printed `the decoder keeps the heuristic kernels`, and went on. That
is exactly the round 31 failure mode the reserve was written for, avoided, and
the lever is still untested. It costs nothing to leave armed.

A caution about that table, because it is the first round where I read one
against another: the four GEMM rows drifted +1.7, +2.1, +3.3 and +1.0 ms
between two builds whose GEMMs are byte-identical. Profile rows are worth
about +/-1.5% against each other. The two rows above are worth reading because
one went to zero and the other is a rename of the same bytes; the totals are
worth reading because they are the sum. Nothing smaller than that is.

What the wall says, which is the number that gets scored:

    per evaluation   r40 193.34 ms   r42 195.38 ms   r43 194.14 ms   r44 192.31 ms
    mean speedup     r40 7.2996x     r42 7.2163x     r43 7.2729x     r44 7.3323x
                                                    (r38 7.2165x)

Every one of round 44's eight cases is faster than every one of round 43's
eight, and the round mean is above all four draws of the previous build. But
it is above them by 0.45% where those four spread 1.15%, so the reward
difference on its own would prove nothing. The profile's -0.84% is the reason
to believe it, and 0.84% is what it is worth - not the 4.3% the two profile
lines advertised.

Mean LPIPS moved 0.28624 -> 0.28952 on a build whose arithmetic changed by one
float32 accumulation order in the depth aggregation. Case 05 moved +0.0406 and
case 04 moved -0.0110 the other way, which is the draw, not the rewrite.

## Rounds 45-50 - what is left

The schedule search ran again and refused again, on the same contradiction it
found in round 43: 0.14292 at eighteen evaluations against 0.13620 at
seventeen, where the gate has 0.25252 against 0.28508. Round 43 established
why that refusal is correct and permanent - the mean is set by case 06's
deterministic cliff and the warmup only ever sees case 01 - so quality stays
closed and the remaining rounds are speed, each of which is also a draw.

The profile after round 44 has one piece of fat left that is not a GEMM:

    vectorized_elementwise_kernel<8>             15.391 ms   2.79%   180 calls
    triton_red_fused__to_copy_mean_mul_pow_...    9.458 ms   1.71%   195 calls

The first is the Attention-Residual accumulation, `partial_block + delta`, 60
times a forward. The second is `source_logits`, the three-query depth
projection, 64 times a forward. Both run well on their own - the add moves
250 MB in 85 us, which is 2.9 TB/s, near the card's roof - so neither is a bad
kernel. They are bad *placement*: the add reads back a tensor that the
compiled region which produced it had in registers, and the logits reduction
then reads the add's output a third time.

Round 45 moves both inside the compiled sublayers. `_cross_attn_merge` and
`forward_mlp_sublayer` take the carry and the query matrix and return the
accumulated block together with its logits, so inductor writes the sum once
and reduces it in the same pass. The association is preserved exactly -
`carry + (attn_delta + merge(heads))`, and `carry + mlp_delta` - so this is the
same arithmetic in the same order, and `test_glue` asserts it against the
spelling it replaces. Neither compiled region gains a second graph variant:
only blocks 0, 8, 16 and 24 ever see a `None` carry, and they never see
anything else.

Round 44's lesson is priced in. Those two rows total 24.8 ms of 551.9, 4.5%,
and round 44 says to expect a fraction of what a row advertises. A complete
fusion saves the add's two reads and the reduction's one, not the whole 24.8:
call it 2%, and a reward near 5.31. If inductor only fuses the reduction into
the add it is about 1% and 5.25. If it fuses nothing it is a sixth draw of a
build whose draws span 1.15% of speed.

Result: reward **5.34328** - the best of the campaign by a margin that is not a
draw. Speedup 7.5051x (3.519 s), mean LPIPS 0.28808, worst 0.38909, gate
passed. Artifacts: `runs/round-45/`, job 19035321, source commit `d47f43e`.

Both target rows are gone and nothing replaced them at the old size:

    kernel row (3 forwards)                  round 44        round 45
    vectorized_elementwise_kernel<8>      15.391 ms 180      -
    triton_red_..._mean_mul_pow_select_sum 9.458 ms 195      -
    triton_red_..._select_sum (fused)          -         10.827 ms  96
    triton_red_..._select_sp. (fused)          -          8.856 ms  96
    Self CUDA time total                    551.890 ms      539.353 ms

375 kernel launches became 192, 24.85 ms became 19.68 ms in those rows, and the
total fell by 12.54 ms - more than the rows themselves gave up, because the
sublayer that used to write its delta out for an eager add to read now writes
the accumulated block directly. The wall agrees and then some:

    per evaluation   r43 194.14 ms   r44 192.31 ms   r45 187.31 ms
    mean speedup     r43 7.2729x     r44 7.3323x     r45 7.5051x

2.36% on the round mean, against a per-case spread of 0.8% (7.484 to 7.544) and
a four-draw spread of 1.15% on the build before last. Every one of the eight
cases is faster than every case of every previous round. This is the first
change in ten rounds that is unambiguously larger than the lottery, and the
reason it is larger is that it did not try to make a kernel faster - it removed
the reason for the kernel to exist.

Mean LPIPS moved 0.28952 -> 0.28808 and worst 0.39339 -> 0.38909, both slightly
better, on a build whose arithmetic is the same arithmetic in the same order.
That is the draw.

The lesson of rounds 44 and 45 together, which is worth more than either: a
profile row that runs at the card's bandwidth roof is not evidence that the
work is necessary. Round 44 attacked two rows that were running *badly* and got
0.84%. Round 45 attacked two rows that were running *well* - the add moved
250 MB in 85 us, 2.9 TB/s - and got 2.36%, because the question was never how
fast those bytes moved but why they were moving at all. Both rows were reading
back a tensor the kernel before them had in registers.

**The decoder's tiling search did not fire again**, and the guard is right to
refuse. Round 45 reached the decode with 395 s left against a 560 s reserve.
The arithmetic is not close: after that line the round still needs 48 s of
decode, 129 s of frozen recompile, 31 s of timed cases and the scoring, and it
finished at about 700 s of a 780 s reservation. 161 s of decoder search does
not fit in 80 s of slack. It only ever fit on a round whose weights came off
page cache, and the last three rounds have loaded the model in 230, 242 and
242 s against rounds 42 and 43's 144. The lever stays armed because an armed
guard that refuses costs nothing, but it should be read as closed.

## Round 46 - the second copy in the cross-attention merge

Round 44 took the row join from `torch.cat` to one `copy_` per row and left
12.5 ms of `elementwise_kernel` behind, 2.3% of the denoiser, and called the
remainder unavoidable because the bytes have to move. They do. What is
avoidable is moving them twice.

`merge` takes attention output as `(batch, heads, tokens, dim)` and does
`out.transpose(1, 2).reshape(batch, tokens, d_model)` before the projection,
because attention produces `tokens` contiguous and the projection needs
`heads * dim` contiguous. That reshape of a transposed tensor is a second full
pass over the same 84 MB, 32 times a forward, and it happens immediately after
the join has just finished writing that tensor out.

So the join writes into the merge's layout directly: allocate
`(rows, tokens, heads * dim)`, take a transposed view of it, and `copy_` each
row of attention output through the view. One strided copy replaces a
contiguous copy plus a strided one. `_join_rows_reference` now spells out the
whole two-pass composition it replaces and `test_glue` asserts the result is
bit-identical to it, which it must be, because both are copies; a third check
runs a real `MultiHeadCrossAttention` and asserts `merge_flat` on the new
layout equals `merge` on the old one.

Round 44's discipline applies: the row advertises 2.3% and what this removes is
the *other* pass, which is not in a row of its own because it is fused inside
the compiled merge region. Call it 1%, a reward near 5.40. If the transpose was
already being folded into the projection's input read it is nothing, and a
seventh draw.

Result: reward **5.33723**, gate passed. Speedup 7.4912x (3.522 s), mean LPIPS
0.28760, worst 0.39697. Artifacts: `runs/round-46/`, job 19038617, source
commit `c5e5cc0`. Round 45 stands as the best at 5.34328.

**A null result, and the hypothesis named it in advance.** The join got
cheaper and the wall did not move:

    aten::copy_ (3 forwards)     r45  13.247 ms 234   r46  7.408 ms 234
    the copy's own kernel row    r45  12.487 ms       r46  below the cutoff
    denoise, mean of 7 cases     r45  3184.7 ms       r46  3185.2 ms
    mean speedup                 r45  7.5051x         r46  7.4912x

Two passes became one and the request takes the same time. The reason is
visible in a kernel rename: `triton_poi_fused__scaled_mm__to_copy__unsafe_view_cl`
at 4.952 ms became `triton_poi_fused__scaled_mm__to_copy_clamp_div_view` at
4.503 ms. That kernel is the projection's fp8 quantizing cast, and in round 45
it was reading its input through the transposed view - `unsafe_view` is the
transpose that never happened. Inductor had already folded `merge`'s reshape
into the cast's input read. There was no second pass to remove. What round 46
actually did was move a strided access from the cast to the join, and the two
sides of that trade are worth the same.

The profile total went the wrong way, 539.353 -> 545.596 ms, which the wall
says is not real. Round 44 put the row-to-row drift at about 1.5% and this is
1.2% of the total, so the two measurements agree that the round is neutral. The
wall is the one that gets scored.

Per case the speedups are 7.440 to 7.550 against round 45's 7.484 to 7.544 -
the same band, wider. Mean LPIPS 0.28808 -> 0.28760 and worst 0.38909 ->
0.39697, both inside the draw; case 06 moved 0.36899 -> 0.32011 and case 02
moved 0.28018 -> 0.31585 on a build whose only change is a copy's memory
layout. The per-case LPIPS is a draw on *any* arithmetic change, including one
that only reorders bytes, which is worth writing down because rounds 38 to 43
treated case 06 as deterministic. It was deterministic across three rounds of
*identical* source. It is not deterministic across a relayout.

That is the second thing this round measured and it was not in the hypothesis:
every remaining round is a fresh draw on all eight cases, not only on speed.

## Rounds 47-50

Three glue rounds have now bracketed what this kind of change is worth. Round
44 removed two badly-running kernels for 0.84%. Round 45 removed two
well-running kernels for 2.36%, by deleting the reason they ran. Round 46
removed a pass that inductor had already removed, for nothing. The pattern is
that what pays is a tensor that stops being written, and what does not is a
tensor that gets written differently.

There is one of the first kind left. `forward_mlp_sublayer` is handed `hidden`
from the depth aggregation and uses it exactly once, in `norm2`; the
aggregation writes 84 MB and the norm reads it straight back, 32 times a
forward. The attention sublayer cannot do this - it uses `hidden` twice, for
`norm1` and for the cross query - but the MLP sublayer can, so round 47 hands
it the value buffer and the logits and lets it aggregate inside the compiled
region. It also reverts round 46, because round 45's build is the measured
best and a null change is not worth carrying into the last rounds.

The honest odds are worse than round 45's. `norm2` is a reduction over 2560
channels, which is above inductor's persistent-reduction threshold, so the
fused loop would have to read `hidden` twice and inductor will very likely
realize the buffer anyway - in which case this is exactly round 46 again, a
null and a draw. If it does fuse it is about 1%. It is the last structural
change the denoiser offers, so it is worth the round either way, and rounds 48
to 50 are then draws of the best build - which round 46 has just established
are draws on all eight cases and not only on speed.

## Round 47 - aggregate inside the MLP sublayer

Result: **0, walltime timeout.** Job 19039912 exited FAILED at 13:32 of a 14
minute reservation with no report written; `supervisor.py finish` recorded
`{"invalid": 1, "reward": 0.0, "error": "missing report"}`. Artifacts:
`runs/round-47/`, source commit `ae8d6cd`, reverted in `d5f16dd`.

The change was correct and never ran. Folding `aggregate_partial` into
`forward_mlp_sublayer` roughly doubled what inductor had to do with that region
and the round ran out of clock inside the compiler:

    warmup sampling ends      r45  373.9 s     r47  440.2 s
    frozen recompile starts   r45  ~500 s      r47  562.1 s
    frozen recompile ends     r45  done        r47  never, deadline ~780 s

The log is full of `Online softmax is disabled on the fly since Inductor
decides to split the reduction`, which is the answer to the question the
hypothesis asked out loud: `norm2` reduces 2560 channels, that is past the
persistent-reduction threshold, inductor split the reduction, and a split
reduction reads its input twice - so the aggregation it was handed had to be
realized anyway. The buffer this was meant to delete was going to be written
regardless. It is round 46's null with a compile bill attached.

Two things go in the ledger. The first is that the odds were written down in
advance and they were right, which is the only reason this costs a round and
not a direction. The second is new and matters for what is left: **compile time
is a live constraint, not slack.** Rounds 44-46 each added a little to it and
the reservation absorbed it; round 47 added enough that it did not. Any
remaining change that enlarges a compiled region is now a timeout risk before
it is a speed question, and there are three rounds left to spend.

## Round 48 - the nested seventeen

Hypothesis: **case 06's cliff between eighteen and seventeen evaluations is a
property of the head of the schedule, not of the number of calls, and it can be
bought back at seventeen calls.**

The evidence has been sitting in the ladder since round 38 and I have been
reading it as a count. Case 06 scores 0.1476 at nineteen evaluations, 0.1599 at
eighteen, and 0.3733, 0.3816, 0.3741 at seventeen across three separate rounds
- a 0.21 step on one case, while four of the other seven cases are *better* at
seventeen than at eighteen. Mean LPIPS across the rungs moves 0.033 and case 06
alone accounts for 0.026 of it. One case falling off a cliff while four improve
is not what a count of function evaluations looks like.

What it does look like is visible once the two plans are written out. The
bisection evaluates

    17: 0 1 2 3 5 8 11 14 17 20 24 28 33 38 43 47 49
    18: 0 1 2 3 4 6 8 11 14 17 20 24 28 33 38 43 47 49

The tails are identical from step 8 on. Every difference between the two is in
the first eight steps, and the bisection is *not* nested: seventeen is not
eighteen minus a call. It gives up steps 4 and 6 and buys step 5 instead. So
the ladder was never measuring "one more evaluation" - it was measuring two
different heads, and attributing the difference to the count.

This round ships the seventeen-call plan that *is* eighteen minus a call. Drop
the single call whose removal leaves the smallest worst-case carried change:
that is step 2, and the plan becomes

    0 1 3 4 6 8 11 14 17 20 24 28 33 38 43 47 49

It carries 0.1381 of unobserved change in its worst gap against the bisection's
0.1336 - 3.4% worse by the equal-change rule, which is exactly why the
bisection never produces it, and exactly what is under test: whether *where* a
call sits in the head matters beyond how much change it carries. Every other
single-call trim of the eighteen is worse still, and the offline suite asserts
that.

This is the cheapest bet left on the board. Seventeen calls either way, so the
speed is round 45's 7.5x and nothing about the compiled graph moves - after
round 47 that last clause is worth as much as the hypothesis. Only the LPIPS
can change. If case 06 recovers toward 0.16 the mean falls about 0.026 and the
reward rises about 0.19, which is larger than every systems lever left
combined. If the mean instead rises past 0.30 the round scores 0, but round 45
stands as the best build and rounds 49 and 50 remain.

One deliberate change to the oracle: `_ship_schedule` refuses to ship anything
at all this round. It reads every candidate through `LADDER_MEAN`, which is the
gate's mean for the *bisected* seventeen and says nothing about the nested one,
and its scoring prompt is case 01 - one of the four cases that prefers
seventeen and therefore the one prompt in the set that cannot see the effect
being tested. It prices the table for the record and ships what was compiled
in. Round 43 already concluded the in-round oracle is accurate about case 01
and that case 01 is the least informative case; this is that conclusion applied
rather than restated.

Source commit `c1bf267`.

Result: reward **5.24611**, gate passed at mean LPIPS 0.29895 against a limit of
0.30. Speedup 7.4833x (3.528 s), worst 0.37250. Artifacts: `runs/round-48/`,
job 19042419, source commit `c1bf267`. Round 45 stands as the best at 5.34328.

**The hypothesis is confirmed and the round still lost.** Case 06 fell off the
cliff the other way:

    case   round 45   round 48    delta
    01      0.17828    0.25717   +0.079
    02      0.28018    0.37151   +0.091
    03      0.25685    0.29961   +0.043
    04      0.33722    0.36777   +0.031
    05      0.25618    0.22506   -0.031
    06      0.36899    0.18823   -0.181
    07      0.38909    0.37250   -0.017
    08      0.23784    0.30978   +0.072
    mean    0.28808    0.29895   +0.011

Case 06 moved **-0.181** on a change that alters seventeen integers and nothing
else. That is the largest single-case move this campaign has produced by any
means, and it settles the question the ladder could not: case 06's 0.21 step
between seventeen and eighteen evaluations was never about the count. It was
about the head. With steps 4 and 6 evaluated, case 06 scores 0.188 at seventeen
calls - better than the 0.1599 the eighteen-call plan gets it, within the draw.
The eighteenth call was buying case 06 steps 4 and 6, and this round bought
them for free.

It cost more than it bought. Case 02 moved +0.091, case 01 +0.079, case 08
+0.072, and the mean went the wrong way by 0.011. Speed was unchanged, 7.4833
against 7.5051, inside the lottery band, exactly as the hypothesis said it
would be.

So the real result is a **decomposition of the head, and it is per-case**:

    step 2 evaluated     cases 01, 02, 08 want it  (+0.07 to +0.09 without it)
    steps 4, 6 evaluated case 06 wants them        (-0.18 without them)
    step 5 evaluated     nobody has asked for it

The bisected seventeen buys step 5 and gives up 2, 4 and 6 - and step 5 is the
one head step no case in the set has shown any preference for. Both seventeens
are trading against each other because they each hold half of what the eight
cases want, and the eighteen-call plan holds all of it, which is why the ladder
looked like a clean function of the count.

This also explains a result I have been treating as noise since round 38. The
mean LPIPS across the rungs moves 0.033 and case 06 alone accounts for 0.026 of
it, while four cases are *better* at seventeen than at eighteen. I read that as
a strange ladder. It is not a ladder at all - it is two different head
placements being compared, with the count as a confound.

## Round 49 - protect the whole head, pay in the tail

Hypothesis: **the head is worth more than the tail, so the seventeen-call plan
should keep every call the eighteen-call plan spends in the head and give up
one in the tail instead.**

Round 48 measured what a single head call is worth: steps 4 and 6 are worth
0.18 of LPIPS to case 06, step 2 is worth 0.07 to 0.09 to three others. Nothing
in this campaign has measured a tail call worth anything like that. So this
round applies the same nesting rule with the head fenced off: drop the call
whose removal leaves the smallest worst-case carried change **among the calls
after the head**. That is step 17, and the plan becomes

    0 1 2 3 4 6 8 11 14 20 24 28 33 38 43 47 49

which is the eighteen-call plan's head entire - every step round 48 showed a
case wanting - with one call removed from the middle of the tail. The gap it
opens is 14 to 20, reusing five calls and carrying 0.2342 of unobserved change
against the bisected seventeen's worst of 0.1336. By the equal-change rule this
is the most aggressive schedule the campaign has shipped; by what round 48
actually measured it is the cheapest place to take the loss.

If the head is the whole story, the per-case LPIPS should land near the
eighteen-call plan's - case 06 around 0.19, cases 01, 02 and 08 back at round
45's values - at seventeen calls and round 45's speed. That is a mean near 0.26
and a reward near 5.5. If the tail is not free, the mean rises and the round is
a null or a zero; the mean is already 0.299 at the nested seventeen, so there
is no room to be wrong in the direction of the gate.

That risk is deliberate and it is the right one to take with two rounds left.
Round 45's 5.34328 is banked and immutable - it is a commit, and round 50 can
be a draw of it if this fails. A round spent re-drawing the best build buys at
most the 0.07 of reward the lottery spans; this one has a measured 0.18 on one
case behind it.

Source commit `TBD`.
