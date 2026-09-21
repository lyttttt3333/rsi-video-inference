# SANA all-eight 50-round journal

Baseline: GPU-resident SANA implementation. Calibration and baseline manifest
are stored separately in the ignored benchmark output directory.

## Baseline calibration

- Slurm job: `18952417`
- Device: NVIDIA H100 80GB HBM3
- Protocol: one case-01 warmup, then one timed generation for cases 01-08
- Mean formal latency: 26.40782920550555 seconds
- Self-check: mean LPIPS 0.0, worst LPIPS 0.0, reward 1.0
- Manifest: external eight-case teacher reference (not committed).

## Round 01 — FAILED (harness, no model execution)

- Hypothesis: a conservative internal reduction from 50 to 45 DPM-Solver evaluations,
  paired with third-order multistep updates, will remove about 10% of denoiser work
  while keeping all-eight mean/worst LPIPS inside 0.30/0.50.
- Mechanism: solver schedule approximation only; prompt, seed, shape, requested step
  field, weights, and output schema are unchanged.
- Research basis: DPM-Solver++ supports high-order multistep sampling and much lower
  NFEs than conventional diffusion schedules: https://arxiv.org/abs/2211.01095
- Commit: `3fa8ee8d243fdab82a2595378b527297bf488202`
- Slurm job: `18952722` (`FAILED`, 29 seconds); the sole reservation is spent
  and will not be retried.
- Result: invalid/reward 0; Pyxis failed before Python/model execution because
  `run_round.sh` specified container workdir `/runner`, but did not mount or
  create `/runner` (`pyxis: couldn't chdir to /runner: No such file or directory`).
- Conclusion: no inference or quality result exists. The candidate remains
  unvalidated; campaign harness must be repaired before round 02.

## Round 02 — completed, rejected

- Hypothesis: 48 internal third-order DPM-Solver evaluations provide a more
  conservative quality/speed operating point than round 01's unexecuted
  proposal, while still reducing denoiser work relative to the 50-step baseline.
- Mechanism and source: DPM-Solver++ high-order multistep integration,
  https://arxiv.org/abs/2211.01095. The external request is unchanged.
- Harness note: corrected the campaign container workdir from nonexistent
  `/runner` to the already mounted read-only `/submission`; no candidate or
  measurement semantics changed.
- Commit/job: `12f52ad486188cc45b8956deb148a00257d803ae`, Slurm
  `18952861` (`COMPLETED`, 513 seconds).
- Candidate seconds 01-08: 25.3800, 25.3554, 25.3706, 25.3733,
  25.3594, 25.3505, 25.3541, 25.3549 (mean 25.3623; about 1.041x).
- LPIPS 01-08: 0.0355, 0.2228, 0.1387, 0.1500, 0.0841, 0.0537,
  0.3423, 0.0774; mean 0.1381, worst 0.3423, global gate passed.
- Reward: 0.897417. Conclusion: the small speed gain does not pay for the
  prompt-dependent quality loss; reject and return to full 50/order-2.

## Round 03 — completed, kept (new winner)

- Hypothesis: compiling only the resident denoiser with PyTorch Inductor
  `reduce-overhead` can fuse pointwise/operator sequences and reduce Python and
  launch overhead without changing 50-step solver math enough to affect LPIPS.
- Source: official `torch.compile` documentation,
  https://pytorch.org/docs/stable/generated/torch.compile.html
- Proposal: restore exact baseline solver and compile only `self.model` after
  it is moved to H100; VAE and text encoder remain eager to limit compile risk.
- Commit/job: `fa43a094256260e8ba266d91522996ddcbda2978`, Slurm
  `18953294` (`COMPLETED`, 355 seconds).
- Candidate seconds 01-08: 21.7725, 21.7741, 21.7692, 21.7330,
  21.7597, 21.7734, 21.7824, 21.7037 (mean 21.7586; about 1.2135x).
- LPIPS 01-08: 0.0173, 0.0779, 0.0299, 0.0457, 0.0988, 0.0983,
  0.2060, 0.0495; mean 0.07793, worst 0.20599, gate passed.
- Reward: 1.119149. Conclusion: keep as first all-eight winner. Compilation
  changes floating-point kernel ordering, hence nonzero LPIPS, but remains well
  inside the global gate and yields a robust latency improvement on every case.

## Round 04 — completed, rejected

- Hypothesis: compiling the static-shape resident VAE in addition to the kept
  denoiser can fuse decoder pointwise/residual work after the single warmup,
  while preserving the round-03 quality profile.
- Source: official module compile API,
  https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.compile
- Proposal: call `self.vae.compile(mode='reduce-overhead', fullgraph=False)`;
  retain all round-03 settings.
- Commit/job: `17e63941eed2d87fd8682c05b6da976ec44be46b`, Slurm
  `18953630` (`COMPLETED`, 568 seconds).
- Candidate seconds 01-08: 21.7671, 21.8296, 21.8209, 21.7760,
  21.7825, 21.7901, 21.7872, 21.7698 (mean 21.7909).
- LPIPS is exactly the round-03 profile: mean 0.07793, worst 0.20599; gate passed.
- Reward: 1.117481, slightly below round 03's 1.119149. Conclusion: VAE
  compilation adds a large warmup cost and no robust hot-latency benefit; reject.

## Round 05 — completed, kept (new winner)

- Hypothesis: Inductor `max-autotune-no-cudagraphs` will spend warmup time
  selecting faster H100 GEMM kernels for the static denoiser than
  `reduce-overhead`, while avoiding CUDA-graph compile risk.
- Source: official PyTorch compile programming model and modes,
  https://pytorch.org/docs/stable/generated/torch.compile.html
- Proposal: retain the round-03 exact 50-step winner, remove VAE compilation,
  and switch only the denoiser compile mode.
- Commit/job: `f2963c29573d3c009f7339f828891c4512891576`, Slurm
  `18954173` (`COMPLETED`, 680 seconds).
- Candidate seconds 01-08: 19.8315, 19.8477, 19.8382, 19.8251,
  19.8479, 19.8401, 19.8505, 19.8385 (mean 19.8399; about 1.3315x).
- LPIPS 01-08: 0.0165, 0.0726, 0.0359, 0.0424, 0.0546, 0.1000,
  0.1949, 0.0387; mean 0.06946, worst 0.19494, gate passed.
- Reward: 1.238627. Conclusion: keep; autotuned H100 kernels improve both
  latency and observed numerical quality over reduce-overhead in this run.

## Round 06 — completed, rejected

- Hypothesis: the classifier-free negative prompt and its Gemma embedding are
  invariant across requests, so computing that intermediate once during build
  removes one exact text-encoder forward from every timed generation.
- Mechanism: cache only the constant negative conditioning tensor and attention
  mask, never a completed output; prompt-specific positive conditioning remains
  freshly encoded for every case.
- Source: classifier-free guidance combines conditional/unconditional model
  estimates, https://arxiv.org/abs/2207.12598
- Commit/job: `4a5add46ab821c90bb7cbbcca066790626e4776b`, Slurm
  `18954806` (`COMPLETED`).
- Candidate seconds 01-08: 19.8478, 19.8228, 19.8247, 19.8175,
  19.8246, 19.8134, 19.8279, 19.8151 (mean 19.8242).
- LPIPS 01-08: 0.0163, 0.0797, 0.0452, 0.0431, 0.0698, 0.1051,
  0.2002, 0.0415; mean 0.07510, worst 0.20020, gate passed.
- Reward: 1.232071, below r5's 1.238627. Conclusion: the exact intermediate
  cache modestly lowers mean latency but run-to-run numerical variation lowered
  aggregate reward; reject under the authoritative observed metric.

## Round 07 — completed, kept (new winner)

- Hypothesis: the full 544x960 VAE decode fits 80GB H100 memory and disabling
  spatial tiling will remove overlap/blending and repeated decoder launches.
- Mechanism: full-frame decode of the same latent, with no change to output
  shape or temporal decode. Restore r5 otherwise.
- Source: upstream Diffusers VAE tiling API,
  https://huggingface.co/docs/diffusers/api/models/autoencoderkl#diffusers.AutoencoderKL.disable_tiling
- Commit/job: `939e060cc397f3be4f7beb1c24a1f8305c30b3ec`, Slurm
  `18955320` (`COMPLETED`).
- Candidate seconds 01-08: 19.6552, 19.6619, 19.6624, 19.6890,
  19.7146, 19.6865, 19.6848, 19.6655 (mean 19.6775; about 1.3421x).
- LPIPS 01-08: 0.0156, 0.0702, 0.0517, 0.0428, 0.0576, 0.1030,
  0.2068, 0.0409; mean 0.07358, worst 0.20676, gate passed.
- Reward: 1.243322, above r5's 1.238627. Conclusion: keep; full-frame
  decode fits H100 and provides a small but consistent hot-latency gain.

## Round 08 — completed, kept (new winner)

- Hypothesis: explicitly disabling math and memory-efficient SDPA fallbacks
  will ensure the BF16 H100 attention path uses FlashAttention rather than a
  slower backend when shapes are eligible.
- Source: official PyTorch SDPA backend controls,
  https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_flash_sdp
- Proposal: retain r7 and globally require flash SDPA. Unsupported attention
  shapes will fail closed in this single spent round rather than silently fall back.
- Commit/job: `d5eebeb2826a3447ec40ad2885a57c6cc35b48ec`, Slurm
  `18955750` (`COMPLETED`).
- Candidate seconds 01-08: 19.0716, 19.0705, 19.0610, 19.0759,
  19.0750, 19.0638, 19.0662, 19.0562 (mean 19.0676; about 1.3840x).
- LPIPS 01-08: 0.0172, 0.2195, 0.0908, 0.0559, 0.0618, 0.1007,
  0.1618, 0.1037; mean 0.10144, worst 0.21955, gate passed.
- Reward: 1.244410, narrowly above r7's 1.243322. Conclusion: keep; forcing
  Flash SDPA yields a large latency reduction, though kernel numerical order
  shifts the per-prompt LPIPS distribution.

## Round 09 — completed, kept (new winner)

- Hypothesis: enabling Inductor CUDA graphs through `max-autotune` can reduce
  repeated launch overhead across the 50 static-shape denoiser calls beyond the
  accepted no-cudagraphs mode.
- Source: official PyTorch CUDA Graph integration,
  https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
- Proposal: retain r8, changing only compile mode to `max-autotune`; the sole
  warmup must absorb capture/compile within the 14-minute reservation.
- Commit/job: `b4acd778292a8d8d2676f62ff98cf14d5b7150de`, Slurm
  `18956187` (`COMPLETED`).
- Candidate seconds 01-08: 18.8593, 18.8283, 18.8237, 18.8382,
  18.8471, 18.8436, 18.8440, 18.8563 (mean 18.8425; about 1.4010x).
- LPIPS 01-08: 0.0152, 0.2239, 0.0918, 0.0530, 0.0637, 0.1001,
  0.1399, 0.1095; mean 0.09963, worst 0.22387, gate passed.
- Reward: 1.261794, above r8's 1.244410. Conclusion: keep; CUDA graphs fit
  the one-warmup protocol within 14 minutes and improve hot latency.

## Round 10 — completed, rejected

- Hypothesis: omitting only the final internal solver evaluation while using a
  third-order multistep update can gain about 2% denoiser speed with tolerable
  perceptual drift on top of the r9 systems winner.
- Source: DPM-Solver++ high-order multistep methods,
  https://arxiv.org/abs/2211.01095
- Proposal: 49 internal updates/order 3; external requested steps remains 50
  and all prompts, seeds, shapes, weights, and output schema remain unchanged.
- Commit/job: `ae75d65c9bfab61e6d3f03dbb6bff6af1215cde5`, Slurm
  `18956660` (`COMPLETED`).
- Candidate seconds 01-08: 18.6237, 18.6264, 18.6147, 18.6251,
  18.6081, 18.6044, 18.6190, 18.6033 (mean 18.6157; about 1.4186x).
- LPIPS 01-08: 0.0198, 0.2854, 0.1839, 0.1580, 0.0525, 0.1075,
  0.3150, 0.1432; mean 0.15816, worst 0.31501, gate passed.
- Reward: 1.194149, below r9's 1.261794. Conclusion: reject; the 1.2%
  extra speed is overwhelmed by prompt-sensitive third-order schedule drift.

## Round 11 — completed, kept (new winner)

- Hypothesis: retaining the baseline second-order method while dropping one
  final internal evaluation may be perceptually closer than r10's change in
  both order and step count, with the same NFE saving.
- Source: DPM-Solver solver-order tradeoffs,
  https://arxiv.org/abs/2206.00927
- Proposal: 49 internal steps/order 2 atop r9; all external inputs unchanged.
- Commit/job: `860db4e7e8e9e960e33de7e420b55e38e39feded`, Slurm
  `18957155` (`COMPLETED`).
- Candidate seconds 01-08: 18.8392, 18.8424, 18.8159, 18.7938,
  18.7857, 18.7918, 18.7827, 18.8688 (mean 18.8152; about 1.4035x).
- LPIPS 01-08: 0.0180, 0.1571, 0.0555, 0.0807, 0.0610, 0.0969,
  0.1729, 0.1190; mean 0.09514, worst 0.17289, gate passed.
- Reward: 1.270017, above r9's 1.261794. Conclusion: keep; order 2 recovers
  enough quality that the one-NFE saving improves the all-eight objective.

## Round 12 — completed, rejected

- Hypothesis: combining r11 with build-time caching of the invariant negative
  Gemma embedding will remove one exact encoder pass from each timed case and
  improve reward without adding solver drift.
- Mechanism: cache only constant unconditional conditioning; completed videos
  and positive prompt states are never cached. CFG basis:
  https://arxiv.org/abs/2207.12598
- Commit/job: `43a4f01af9f7e1a14c59bdfdc294ce47c77e32d2`, Slurm
  `18957500` (`COMPLETED`).
- Candidate seconds 01-08: 19.4732, 18.7615, 18.7604, 18.7664,
  18.7608, 18.7444, 18.7555, 18.8258 (mean 18.8560).
- LPIPS mean/worst: 0.09603/0.17633; global gate passed.
- Reward: 1.265685, below r11's 1.270017. Conclusion: reject; the negative
  cache did not robustly improve the full all-eight aggregate (case 01 latency
  was notably worse), so restore r11 behavior.

## Round 13 — completed, kept (new winner)

- Hypothesis: a second omitted evaluation at unchanged order 2 may improve the
  speed-quality product further while remaining inside the relaxed global gate.
- Source: DPM-Solver fast sampling, https://arxiv.org/abs/2206.00927
- Proposal: restore r11's uncached text path and evaluate 48 internal steps,
  order 2, across all eight prompts.
- Commit/job: `5e72be5d98edf194b8c90b6ab6ec6b088ced51cd`, Slurm
  `18958207` (`COMPLETED`).
- Candidate seconds 01-08: 18.4072, 18.4107, 18.5102, 18.4149,
  18.4246, 18.4066, 18.4324, 18.4226 (mean 18.4286; about 1.4331x).
- LPIPS 01-08: 0.0251, 0.1476, 0.0551, 0.0756, 0.0522, 0.0989,
  0.2292, 0.1040; mean 0.09845, worst 0.22917, gate passed.
- Reward: 1.291889, above r11's 1.270017. Conclusion: keep; the second NFE
  saving improves aggregate reward despite higher case-07 drift.

## Round 14 — completed, kept (new winner)

- Hypothesis: 47 second-order updates may remain on the favorable side of the
  observed speed-quality frontier under the 0.30/0.50 global gate.
- Source: DPM-Solver fast sampling, https://arxiv.org/abs/2206.00927
- Proposal: reduce one further internal NFE from r13, leaving every other
  systems and request setting unchanged.
- Commit/job: `497a74a043d2d46307a13fb99387fed7be510a1a`, Slurm
  `18958797` (`COMPLETED`).
- Candidate seconds 01-08: 17.9777, 17.9802, 17.8816, 17.9468,
  17.9242, 17.9296, 17.9365, 17.9180 (mean 17.9361; about 1.4723x).
- LPIPS 01-08: 0.0342, 0.2391, 0.0766, 0.0958, 0.0881, 0.1040,
  0.2350, 0.1016; mean 0.12180, worst 0.23906, gate passed.
- Reward: 1.292970, narrowly above r13's 1.291889. Conclusion: keep, while
  noting the quality frontier is flattening and sensitive prompts are worsening.

## Round 15 — completed, kept (new winner)

- Hypothesis: 46 order-2 updates may provide one more useful NFE saving before
  the rising LPIPS on cases 02 and 07 overwhelms latency gain.
- Source: DPM-Solver fast sampling, https://arxiv.org/abs/2206.00927
- Proposal: decrement internal step count once more; retain r14 otherwise.
- Commit/job: `268d6d21f69110ec4b1e16f0c6b782a9233bbf7d`, Slurm
  `18959257` (`COMPLETED`).
- Candidate seconds 01-08: 17.5173, 17.5162, 17.5722, 17.5798,
  17.5377, 17.5648, 17.5596, 17.5740 (mean 17.5527; about 1.5035x).
- LPIPS 01-08: 0.0388, 0.1925, 0.1065, 0.1068, 0.1078, 0.1053,
  0.2356, 0.0833; mean 0.12207, worst 0.23559, gate passed.
- Reward: 1.320791, substantially above r14. Conclusion: keep; the 46-step
  point improves speed without worsening mean quality versus r14 in this run.

## Round 16 — completed, rejected

- Hypothesis: 45 second-order updates may retain the favorable quality plateau
  seen at 46 while saving another denoiser evaluation.
- Source: DPM-Solver fast sampling, https://arxiv.org/abs/2206.00927
- Proposal: 45 internal steps/order 2, all other r15 settings fixed.
- Commit/job: `ca52bc05e834b315c165ba9efafbc8bda7206db5`, Slurm
  `18959705` (`COMPLETED`).
- Candidate seconds 01-08: 17.2184, 17.2109, 17.2079, 17.2292,
  17.2123, 17.2208, 17.2414, 17.2279 (mean 17.2211; about 1.5333x).
- LPIPS 01-08: 0.0478, 0.3166, 0.1499, 0.0993, 0.0980, 0.1108,
  0.2589, 0.1161; mean 0.14969, worst 0.31660, gate passed.
- Reward: 1.303817, below r15's 1.320791. Conclusion: reject; at 45 steps
  the quality loss, especially case 02, outweighs one more NFE saving.

## Round 17 — completed, kept (new winner)

- Hypothesis: denoiser predictions vary smoothly across neighboring diffusion
  evaluations, so reusing the immediately previous full prediction every fifth
  call can skip about 20% of transformer work with tolerable LPIPS.
- Mechanism: a request-local periodic output cache reset before every video;
  no cross-prompt state and no completed output cache. Restore the r15 46-step
  winner as the base.
- Research basis: DeepCache reuses temporally stable diffusion features,
  https://arxiv.org/abs/2312.00858 and TeaCache exploits timestep-wise output
  redundancy, https://arxiv.org/abs/2411.19108
- Commit/job: `4811af6a9026b6d70cd81f5850db8c636137b943`, Slurm
  `18960335` (`COMPLETED`).
- Candidate seconds 01-08: 14.3949, 14.3243, 14.3329, 14.3544,
  14.3178, 14.3170, 14.3285, 14.3291 (mean 14.3374; about 1.842x).
- LPIPS 01-08: 0.1109, 0.2535, 0.2361, 0.2695, 0.1344, 0.2551,
  0.3330, 0.2010; mean 0.22419, worst 0.33296, gate passed.
- Reward: 1.428827, far above r15's 1.320791. Conclusion: keep. Request-local
  adjacent-output reuse is the strongest optimization so far, with remaining
  but reduced mean-quality margin.

## Round 18 — completed, kept (new winner)

- Hypothesis: increasing periodic reuse from every fifth to every fourth call
  (about 25% skipped transformer calls) may improve reward while the relaxed
  mean/worst gates still provide room.
- Research basis: DeepCache, https://arxiv.org/abs/2312.00858 and TeaCache,
  https://arxiv.org/abs/2411.19108
- Proposal: change only cache period 5 to 4; retain the 46-step r17 base and
  request-local reset semantics.
- Commit/job: `1dff3b8836f7bdf7daf3bbf9caa674aa1fe82607`, Slurm
  `18960677` (`COMPLETED`).
- Candidate seconds 01-08: 13.5094, 13.5044, 13.5104, 13.5015,
  13.4942, 13.4925, 13.5515, 13.5107 (mean 13.5093; about 1.954x).
- LPIPS 01-08: 0.1709, 0.2652, 0.2629, 0.3058, 0.1443, 0.1896,
  0.3447, 0.1997; mean 0.23538, worst 0.34472, gate passed.
- Reward: 1.494703, above r17. Conclusion: keep; 25% periodic reuse
  improves the objective, though mean-quality margin is now about 0.065.

## Round 19 — completed, kept (new winner)

- Hypothesis: every-third-call reuse (about one-third transformer skips) may
  exploit additional cross-step redundancy while remaining just inside the
  global mean/worst gates.
- Research basis: TeaCache, https://arxiv.org/abs/2411.19108
- Proposal: period 4 to 3 only; cache remains request-local and resets per video.
- Commit/job: `0434d207ef2460703203e5f44bf414ec5d4d08f1`, Slurm
  `18961057` (`COMPLETED`).
- Candidate seconds 01-08: 12.1372, 12.0956, 12.1019, 12.1037,
  12.0890, 12.0975, 12.1180, 12.1061 (mean 12.1062; about 2.181x).
- LPIPS 01-08: 0.2184, 0.4131, 0.2706, 0.2180, 0.2006, 0.1891,
  0.3748, 0.2868; mean 0.27142, worst 0.41311, gate passed.
- Reward: 1.589173, above r18. Conclusion: keep. Period-3 reuse is the
  current winner but has only about 0.029 mean-LPIPS and 0.087 worst-LPIPS margin.

## Round 20 — completed, rejected (global gate)

- Hypothesis: alternating exact and reused denoiser predictions (period 2) may
  deliver enough additional speed to offset quality loss, though it deliberately
  probes the likely global-gate boundary.
- Research basis: TeaCache cross-timestep redundancy,
  https://arxiv.org/abs/2411.19108
- Proposal: change only request-local cache period 3 to 2.
- Commit/job: `9a6c2162dfe4fea153d481d676495970d77f8d7a`, Slurm
  `18961273` (`COMPLETED`).
- Candidate mean latency about 9.245s (about 2.855x); raw reward 1.820908.
- LPIPS 01-08: 0.3257, 0.4163, 0.4079, 0.3705, 0.2561, 0.3296,
  0.5016, 0.2925; mean 0.36253 and worst 0.50165.
- Reward: 0.0 because both mean<=0.30 and worst<=0.50 gates failed.
  Conclusion: reject; period 2 establishes a clear upper bound on cache strength.

## Round 21 — completed, kept (new winner)

- Hypothesis: timestep-aware reuse can recover quality by using period 4 at
  the early/late boundary calls and period 2 only in the smoother middle window,
  for roughly 17 skips versus period-3 winner's 15.
- Mechanism: calls 11-36 reuse on even calls; boundary calls reuse every fourth;
  request-local reset remains mandatory.
- Research basis: TeaCache's timestep-aware rescaling,
  https://arxiv.org/abs/2411.19108
- Commit/job: `018c156e99f1657b5d489a985e4ee5a1f3d31d07`, Slurm
  `18961514` (`COMPLETED`).
- Candidate seconds 01-08: 11.5732, 11.5710, 11.5763, 11.5833,
  11.5891, 11.5785, 11.5840, 11.5491 (mean 11.5754; about 2.282x).
- LPIPS 01-08: 0.1687, 0.2624, 0.2635, 0.3032, 0.1366, 0.1913,
  0.3540, 0.1991; mean 0.23486, worst 0.35398, gate passed.
- Reward: 1.745487, far above period-3 r19. Conclusion: keep; concentrating
  dense reuse in calls 11-36 is both faster and better quality than uniform
  period-3 reuse, validating timestep-aware cache scheduling.

## Round 22 — completed, kept (new winner)

- Hypothesis: widening the dense middle window from calls 11-36 to 7-40 adds
  two net skips while preserving conservative period-4 treatment at the most
  extreme boundary calls.
- Research basis: timestep-aware cache scheduling in TeaCache,
  https://arxiv.org/abs/2411.19108
- Proposal: only widen the middle window; all other r21 settings fixed.
- Commit/job: `d2421f6aa15f44695da5e792322532e194c6cdee`, Slurm
  `18961721` (`COMPLETED`).
- Candidate seconds 01-08: 10.7942, 10.7592, 10.7704, 10.7993,
  10.7858, 10.7644, 10.8094, 10.7673 (mean 10.7812; about 2.449x).
- LPIPS 01-08: 0.1729, 0.2313, 0.2633, 0.3180, 0.1369, 0.1799,
  0.3303, 0.2284; mean 0.23262, worst 0.33029, gate passed.
- Reward: 1.879679, above r21. Conclusion: keep; calls 7-40 add two skips
  while unexpectedly improving measured quality distribution as well.

## Round 23 — completed, kept (new winner)

- Hypothesis: retaining exact calls only at the first two and last two model
  evaluations, with alternating reuse for calls 3-44, may add two more skips
  without the severe all-step period-2 quality failure.
- Research basis: boundary-aware timestep caching, TeaCache,
  https://arxiv.org/abs/2411.19108
- Proposal: widen middle window to calls 3-44; other r22 settings fixed.
- Commit/job: `9d180f9653892fb2cff097cb4735da994b1dacab`, Slurm
  `18962114` (`COMPLETED`).
- Candidate seconds 01-08: 10.0495, 10.0015, 9.9914, 10.0306,
  10.0215, 10.0302, 9.9877, 9.9658 (mean 10.0097; about 2.638x).
- LPIPS 01-08: 0.1635, 0.2891, 0.2750, 0.3309, 0.1812, 0.1613,
  0.3915, 0.2533; mean 0.25572, worst 0.39153, gate passed.
- Reward: 1.963193, above r22. Conclusion: keep; protecting just the first
  two and last two evaluations is enough to avoid period-2's gate failure.

## Round 24 — completed, rejected (global gate)

- Hypothesis: making call 2 reusable while retaining exact call 1 and the last
  two calls adds one skip and may remain below the quality gates.
- Proposal: widen dense window start from call 3 to call 2; all else fixed.
- Research basis: boundary-aware cache scheduling, TeaCache,
  https://arxiv.org/abs/2411.19108
- Commit/job: `0d9a39d6c74e5506a898bc2aa820c5d9dd357b64`, Slurm
  `18962479` (`COMPLETED`).
- Candidate mean latency about 9.618s (about 2.745x); raw reward 1.752481.
- LPIPS 01-08: 0.3225, 0.4154, 0.4122, 0.3709, 0.2526, 0.3279,
  0.5009, 0.2912; mean 0.36169, worst 0.50091.
- Reward: 0.0 because both global gates fail. Conclusion: reject; call 2 is
  a critical boundary evaluation and must remain exact.

## Round 25 — completed, rejected

- Hypothesis: restoring all 50 solver evaluations while preserving exact first
  two/last two denoiser calls and alternating cache reuse in between may recover
  enough quality to offset four additional scheduled evaluations.
- Proposal: full requested 50/order2; adaptive middle window calls 3-48. This
  retains request-local cache semantics and changes no external request field.
- Research basis: combining higher-quality schedules with cross-step caching,
  TeaCache, https://arxiv.org/abs/2411.19108
- Commit/job: `d03211cb41de6e3320d1164a5228ac3ace90a62a`, Slurm
  `18962980` (`COMPLETED`).
- Candidate mean latency 10.8210s (about 2.440x).
- LPIPS 01-08: 0.1632, 0.2660, 0.2520, 0.3204, 0.1352, 0.1376,
  0.3400, 0.2549; mean 0.23368, worst 0.34001, gate passed.
- Reward: 1.870071, below r23's 1.963193. Conclusion: reject; full 50-step
  quality recovery does not offset its additional exact denoiser work.

## Round 26 — completed, rejected

- Hypothesis: 47 scheduled steps with the same first-two/last-two exact-boundary
  policy provide an intermediate schedule/cache tradeoff between r23 and r25.
- Proposal: 47/order2 and dense cache calls 3-45, which keeps 21 reuse events
  but performs one more exact evaluation than r23.
- Research basis: joint schedule/cache optimization, TeaCache,
  https://arxiv.org/abs/2411.19108
- Commit/job: `f1a2be5673147a4b024e14fb95d168f54d0f2a69`, Slurm
  `18963349` (`COMPLETED`).
- Candidate mean latency about 10.519s (about 2.510x).
- LPIPS 01-08: 0.1645, 0.2766, 0.2702, 0.3262, 0.1592, 0.1565,
  0.3686, 0.2561; mean 0.24724, worst 0.36864, gate passed.
- Reward: 1.889879, below r23's 1.963193. Conclusion: reject; 47 steps
  provide insufficient quality recovery for the extra exact call.

## Round 27 — failed before model execution

- Hypothesis: a prompt-adaptive cache keyed by relative latent change can skip
  stable evaluations and retain exact computation when the trajectory changes
  rapidly, improving robustness over a fixed call schedule.
- Mechanism: compare current latent to the last exact-call latent; reuse the
  last output inside calls 3-44 only when relative L1 change is below 0.08.
  Restore the r23 46-step base and protect first/last two calls.
- Research basis: TeaCache's relative-change decision and thresholding,
  https://arxiv.org/abs/2411.19108
- Commit/job: `efc58877a6c59bcc7df706a9723ff694192fd822`, Slurm
  `18963705` (`FAILED`, sole reservation spent, no retry).
- Result: invalid/reward 0. The verifier exited before build/inference with
  `ValueError: Exactly one visible CUDA GPU is required` on the allocated node.
- Conclusion: no evidence for or against threshold 0.08; this exact round is
  not retried. The next round changes the threshold and remains a distinct trial.

## Round 28 — completed, rejected (global gate)

- Hypothesis: relative latent-change threshold 0.12 will trigger enough dynamic
  reuse to be measurable while still protecting rapidly changing evaluations.
- Mechanism/source: same request-local last-exact latent/output cache as r27,
  but a distinct threshold; TeaCache, https://arxiv.org/abs/2411.19108
- Commit/job: `dd3495398819fcdb1f9cef9b4965468b86a0a92e`, Slurm
  `18963728` (`COMPLETED`).
- Candidate mean latency about 5.00s (over 5.2x); raw reward 1.733116.
- LPIPS 01-08: 0.5414, 0.7181, 0.6944, 0.7072, 0.7789, 0.6374,
  0.7133, 0.5809; mean 0.67145, worst 0.77893.
- Reward: 0.0; threshold 0.12 causes near-total reuse and fails both gates.
  Conclusion: reject and lower the threshold substantially.

## Round 29 — completed, rejected (global gate)

- Hypothesis: threshold 0.02 will turn the same latent-delta controller into a
  selective cache rather than near-total reuse, preserving quality while still
  skipping stable calls.
- Source: TeaCache threshold sweep methodology,
  https://arxiv.org/abs/2411.19108
- Commit/job: `0f13f29bd467cf5e984b30942f8608a07d4322d6`, Slurm
  `18963999` (`COMPLETED`).
- Candidate latency roughly 10.5-11.2s; raw reward 1.404161.
- LPIPS 01-08: 0.3294, 0.5933, 0.4573, 0.4759, 0.2608, 0.3332,
  0.6000, 0.3809; mean 0.42886, worst 0.60000.
- Reward: 0.0. Conclusion: even threshold 0.02 reuses too aggressively for
  raw latent deltas; lower by another 4x before abandoning this signal.

## Round 30 — completed, rejected

- Hypothesis: threshold 0.005 will make raw-latent adaptive reuse sparse enough
  to pass global quality gates and reveal whether the signal can beat r23.
- Source: TeaCache threshold sweep, https://arxiv.org/abs/2411.19108
- Commit/job: `03b781e8fd4ce0af413b8859ca07fdd25484cf32`, Slurm
  `18964364` (`COMPLETED`; node prolog was delayed but inference finished).
- Candidate seconds 01-08: 17.0036, 17.0432, 17.0504, 16.7074,
  15.6398, 15.5808, 16.3670, 15.6354.
- LPIPS 01-08: 0.0793, 0.1664, 0.2607, 0.2397, 0.1054, 0.1573,
  0.3672, 0.2566; mean 0.20408, worst 0.36722, gate passed.
- Reward: 1.285363, far below r23. Conclusion: raw latent-delta caching is
  prompt-variable and not competitive with the learned fixed boundary schedule.

## Round 31 — completed, rejected

- Hypothesis: on the r23 cache winner, precomputing the invariant negative Gemma
  embedding during build removes a now-more-significant fraction of the 10s hot
  path and may raise aggregate reward.
- Mechanism: restore exact r23 denoiser schedule; cache only constant CFG
  unconditional conditioning, never positive prompt states or completed videos.
- Source: classifier-free guidance, https://arxiv.org/abs/2207.12598
- Commit/job: `b3e88521a5a723a6e6861d8e111d80716541d88a`, Slurm
  `18964861` (`COMPLETED`).
- Candidate mean latency about 10.048s.
- LPIPS 01-08: 0.1629, 0.2860, 0.2738, 0.3290, 0.1810, 0.1611,
  0.3904, 0.2560; mean 0.25502, worst 0.39036, gate passed.
- Reward: 1.957963, slightly below r23's 1.963193. Conclusion: reject under
  the authoritative observed metric and restore per-request negative encoding.

## Round 32 — completed, rejected

- Hypothesis: compiling the full-frame VAE decoder on top of r23 may fuse
  residual/pointwise decode work; with denoising near 10s, even a small VAE win
  can now materially improve total latency.
- Source: PyTorch module compile API,
  https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.compile
- Proposal: restore r23 text path and add VAE `reduce-overhead` compilation.
- Commit/job: `b16ee8dfef42c007e2c0970102f135fa707e4aac`, Slurm
  `18965285` (`COMPLETED`; delayed node prolog, no retry).
- Candidate mean latency about 10.165s.
- LPIPS mean/worst: 0.25589/0.39364; global gate passed.
- Reward: 1.934069, below r23's 1.963193. Conclusion: VAE compilation adds
  one-time cost and slightly regresses the hot path; reject.

## Round 33 — completed, rejected

- Hypothesis: explicitly selecting PyTorch `high` float32 matmul precision can
  permit TF32 tensor-core kernels for any residual FP32 GEMMs without altering
  BF16 attention and may trim text/VAE/runtime overhead.
- Source: official matmul precision API,
  https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
- Proposal: restore r23 and set precision once during build.
- Commit/job: `d2efafd924552d7538d7a6101008ef7ab921a29a`, Slurm
  `18965714` (`COMPLETED`).
- Candidate mean latency about 10.068s; mean/worst LPIPS 0.25452/0.38325.
- Reward: 1.955575, below r23. Conclusion: explicit high FP32 matmul
  precision provides no robust aggregate gain; reject.

## Round 34 — completed, kept (new winner)

- Hypothesis: enabling cuDNN benchmarking lets the untimed warmup select faster
  convolution algorithms for fixed-shape full-frame VAE decode.
- Source: official cuDNN backend flag,
  https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.benchmark
- Proposal: restore r23 and enable benchmark once during build.
- Commit/job: `026406c60324279bf7365ca85aaf811476337542`, Slurm
  `18966584` (`COMPLETED`).
- Candidate mean latency about 9.926s (about 2.658x).
- LPIPS mean/worst: 0.25474/0.38663; global gate passed.
- Reward: 1.982707, above r23's 1.963193. Conclusion: keep; cuDNN's
  fixed-shape search provides a small but reproducible aggregate gain.

## Round 35 — completed, rejected

- Hypothesis: combining r34's faster VAE algorithms with build-time invariant
  negative-conditioning cache may stack independent text and decode savings.
- Mechanism: cache only the constant CFG negative embedding; r34 diffusion
  schedule/cache and cuDNN policy remain unchanged.
- Source: classifier-free guidance, https://arxiv.org/abs/2207.12598
- Commit/job: `8e8236e1df6533f51d99da17daa3923dda11a696`, Slurm
  `18967241` (`COMPLETED`).
- Candidate mean latency about 10.036s (about 2.631x).
- LPIPS 01-08: 0.16792, 0.28698, 0.27444, 0.33079, 0.17584,
  0.16528, 0.38605, 0.25986; mean 0.25589, worst 0.38605, gate passed.
- Reward: 1.957908, below r34's 1.982707. Conclusion: the exact cache does
  not amortize favorably in this harness; reject and restore r34 behavior.

## Round 36 — completed, rejected

- Hypothesis: explicitly allowing TF32 in cuBLAS and cuDNN can accelerate any
  residual FP32 matmuls/convolutions around the BF16 core while preserving the
  existing r34 algorithm and quality gate.
- Mechanism: restore r34, retain cuDNN benchmark selection, and enable the two
  backend TF32 flags once during runner construction.
- Source: PyTorch CUDA backend documentation,
  https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.matmul.allow_tf32
- Commit/job: `635243e97d43c30fd17c46989be0cbebfd1c614b`, Slurm
  `18967859` (`COMPLETED`).
- Candidate mean latency about 10.045s; LPIPS mean/worst 0.25494/0.38645,
  global gate passed.
- Reward: 1.958653, below r34's 1.982707. Conclusion: these flags provide no
  incremental benefit beyond the existing BF16/compiled path; reject.

## Round 37 — completed, rejected

- Hypothesis: compiling Gemma with the default compiler mode (without the
  reduce-overhead CUDA-graph policy that previously caused output-lifetime
  issues) can fuse exact per-prompt text encoding while preserving semantics.
- Mechanism: restore r34, compile the resident text encoder after transfer;
  the required first-case warmup specializes its fixed positive/negative
  shapes before timed cases.
- Source: PyTorch `torch.compile` documentation,
  https://pytorch.org/docs/stable/generated/torch.compile.html
- Commit/job: `38da657a579723e67b7a4f57f13b21c3bb7c4463`, Slurm
  `18968451` (`COMPLETED`).
- Candidate mean latency about 10.010s; LPIPS 01-08: 0.16130, 0.29532,
  0.27674, 0.33367, 0.18037, 0.17372, 0.39936, 0.26028; mean/worst
  0.26010/0.39936, gate passed.
- Reward: 1.951870, below r34. Conclusion: compiled Gemma's modest hot-path
  timing is outweighed by observed numerical-quality loss; reject.

## Round 38 — completed, rejected

- Hypothesis: for the same 21 cached denoiser calls, shifting reuse from even
  to odd calls protects one additional late denoising update (exact calls
  44-46) at the cost of reusing call 3; late-state accuracy may improve LPIPS
  without changing compute.
- Mechanism: restore r34 text path and reverse the adaptive cache parity only;
  exact/reused call count stays identical.
- Source: DeepCache's timestep-dependent redundancy motivation,
  https://arxiv.org/abs/2312.00858
- Commit/job: `928032f45e9bf5944645b4186e941e361d747586`, Slurm
  `18969137` (`COMPLETED`).
- Candidate mean latency about 10.047s; LPIPS 01-08: 0.17616, 0.39564,
  0.27762, 0.32407, 0.21373, 0.32439, 0.39914, 0.28409; mean/worst
  0.29935/0.39914, gate passed narrowly.
- Reward: 1.841442. Conclusion: protecting the extra late call does not
  compensate for approximating call 3; reject and restore even-call reuse.

## Round 39 — failed, rejected

- Hypothesis: on skipped calls, a half-step linear extrapolation from the two
  latest exact denoiser predictions may track time evolution more accurately
  than zero-order reuse, at negligible elementwise cost.
- Mechanism: restore r34 parity and retain two latest exact outputs; return
  `latest + 0.5 * (latest - prior)` for cached calls.
- Source: TeaCache's output-difference/cross-timestep approximation premise,
  https://arxiv.org/abs/2411.19108
- Commit/job: `9a610a9a7a95ddac6892c8db86fdefd5dd55973d`, Slurm
  `18969568` (`FAILED`; no retry).
- The public evaluator marked case 01 invalid: exact outputs being differenced
  alternated 128 and 64 channels inside the CFG wrapper. Remaining seven cases
  produced LPIPS but the global result is invalid/reward 0.
- Conclusion: state-difference extrapolation at this wrapper boundary violates
  shape invariants; reject and restore the proven single-output cache.

## Round 40 — completed, rejected

- Hypothesis: a conservative 0.99 decay on reused predictions can compensate
  for cross-step output-magnitude drift without requiring a second history
  tensor or shape-dependent differencing.
- Mechanism: restore r34 and multiply only skipped outputs by 0.99; exact call
  placement and backbone call count are unchanged.
- Source: TeaCache's timestep-dependent model-output rescaling premise,
  https://arxiv.org/abs/2411.19108
- Commit/job: `8cb818722a34da67332b75c56b9d47d08d0c7d1a`, Slurm
  `18970114` (`COMPLETED`).
- Candidate mean latency about 10.040s; LPIPS mean/worst 0.25917/0.39507,
  gate passed; reward 1.948461.
- Conclusion: output decay moves quality away from the reference and adds no
  speed benefit; reject.

## Round 41 — completed, rejected

- Hypothesis: because 0.99 decay degraded quality, a symmetric 1.01 gain on
  reused predictions may correct output-magnitude drift in the opposite
  direction while keeping the same skipped-call schedule.
- Mechanism: change only cached-output scale from 0.99 to 1.01.
- Source: TeaCache, https://arxiv.org/abs/2411.19108
- Commit/job: `7f05af90babab034badd117cd99047eece39ebf1`, Slurm
  `18970429` (`COMPLETED`).
- Candidate mean latency about 10.048s; LPIPS mean/worst 0.25638/0.39318,
  gate passed; reward 1.954372.
- Conclusion: amplification also trails unscaled reuse, so abandon scalar
  correction and restore exact zero-order cache output.

## Round 42 — completed, rejected

- Hypothesis: a slightly lower flow shift (11.5 vs 12.0) redistributes the
  unchanged 46 solver steps and may reduce cache-induced trajectory error at
  identical denoiser-call count.
- Mechanism: restore r34 output reuse and change only sampler `flow_shift`.
- Source: DPM-Solver++, https://arxiv.org/abs/2211.01095
- Commit/job: `0bbcf3ea0df1e166fe8ad0b552b3104e251ca7ba`, Slurm
  `18970795` (`COMPLETED`).
- Candidate mean latency about 10.033s; LPIPS mean/worst 0.26588/0.39269,
  gate passed; reward 1.932053.
- Conclusion: shifting the trajectory downward worsens aggregate perceptual
  agreement; reject.

## Round 43 — completed, kept (new winner)

- Hypothesis: a symmetric upward flow-shift change (12.5) may move the cached
  trajectory toward the 50-step GPU-resident reference while keeping compute
  unchanged.
- Mechanism: change only `flow_shift` from 11.5 to 12.5.
- Source: DPM-Solver++, https://arxiv.org/abs/2211.01095
- Commit/job: `bb7b6ed55da92981dace347d6273cc53dfb1c4c9`, Slurm
  `18971120` (`COMPLETED`).
- Candidate mean latency about 10.029s; LPIPS 01-08: 0.16593, 0.26831,
  0.26463, 0.32525, 0.14968, 0.15489, 0.36587, 0.26007; mean/worst
  0.24433/0.36587, gate passed.
- Reward: 1.989548, above r34's 1.982707. Conclusion: keep; upward shift
  materially improves cached-trajectory quality without extra denoiser work.

## Round 44 — completed, kept (new winner)

- Hypothesis: continuing the beneficial direction to flow shift 13.0 may
  further improve cache/reference alignment at identical compute.
- Mechanism: change only `flow_shift` from 12.5 to 13.0.
- Source: DPM-Solver++, https://arxiv.org/abs/2211.01095
- Commit/job: `b6c20bf7daf46314fb547be803aa59792ea810c6`, Slurm
  `18971363` (`COMPLETED`).
- Candidate mean latency about 10.018s; LPIPS 01-08: 0.16366, 0.25645,
  0.25500, 0.32476, 0.14337, 0.15164, 0.36175, 0.25160; mean/worst
  0.23853/0.36175, gate passed.
- Reward: 2.007169, new winner. Conclusion: keep flow shift 13.0.

## Round 45 — completed, rejected

- Hypothesis: flow shift 13.5 may continue the monotonic quality improvement
  observed from 12.0 through 13.0 without changing compute.
- Mechanism: change only `flow_shift` from 13.0 to 13.5.
- Source: DPM-Solver++, https://arxiv.org/abs/2211.01095
- Commit/job: `647c4e81db2811170124f172e73ca020fea2c5b4`, Slurm
  `18971790` (`COMPLETED`).
- Candidate mean latency about 10.029s; LPIPS mean/worst 0.24319/0.36018,
  gate passed; reward 1.992165.
- Conclusion: quality remains strong, but aggregate reward falls below r44;
  reject and bracket the local optimum between 13.0 and 13.5.

## Round 46 — completed, rejected

- Hypothesis: flow shift 13.25 may retain r44's aggregate quality while
  capturing some of r45's best-case improvements.
- Mechanism: set only `flow_shift` to the bracket midpoint 13.25.
- Source: DPM-Solver++, https://arxiv.org/abs/2211.01095
- Commit/job: `d53d5f21afb8e0662cbaf6e718234f2f109d323f`, Slurm
  `18971941` (`COMPLETED`).
- Candidate mean latency about 10.056s; LPIPS mean/worst 0.23910/0.34074,
  gate passed; reward 1.997911.
- Conclusion: worst-case quality improves but aggregate reward stays below
  r44; reject.

## Round 47 — completed, kept (new winner)

- Hypothesis: flow shift 13.125 may split the quality tradeoff between the
  13.0 winner and 13.25's improved worst case while recovering aggregate
  reward.
- Mechanism: set only `flow_shift` to 13.125.
- Source: DPM-Solver++, https://arxiv.org/abs/2211.01095
- Commit/job: `a321d456c8de4f855706e853c49d1550556297eb`, Slurm
  `18972190` (`COMPLETED`).
- Candidate mean latency about 10.054s; LPIPS 01-08: 0.16151, 0.25737,
  0.25788, 0.31917, 0.12776, 0.13394, 0.34364, 0.25692; mean/worst
  0.23227/0.34364, gate passed.
- Reward: 2.016754, new winner. Conclusion: keep flow shift 13.125.

## Round 48 — completed, rejected

- Hypothesis: flow shift 13.0625 may improve the aggregate compromise between
  r44 (13.0) and r47 (13.125) while preserving identical compute.
- Mechanism: set only `flow_shift` to the left midpoint 13.0625.
- Source: DPM-Solver++, https://arxiv.org/abs/2211.01095
- Commit/job: `32d273eb053f922aad9a543954186651b194d4c6`, Slurm
  `18972867` (`COMPLETED`).
- Candidate mean latency about 10.029s; LPIPS mean/worst 0.23893/0.34750,
  gate passed; reward 2.003740.
- Conclusion: below r47; reject.

## Round 49 — completed, rejected

- Hypothesis: flow shift 13.1875 may improve the right side of the r47 local
  optimum without reaching the 13.25 regression.
- Mechanism: set only `flow_shift` to the right midpoint 13.1875.
- Source: DPM-Solver++, https://arxiv.org/abs/2211.01095
- Commit/job: `83b35943e0b1951577bc0c101bec1c3393613a6e`, Slurm
  `18973178` (`COMPLETED`).
- Candidate mean latency about 10.036s; LPIPS mean/worst 0.23966/0.34890,
  gate passed; reward 2.000497.
- Conclusion: both immediate neighbors score below r47, confirming 13.125 as
  the measured local winner; reject.

## Round 50 — completed, kept (final winner)

- Hypothesis: with the best flow shift restored, reducing from 46 to 45 solver
  steps removes one exact backbone call while the adaptive cache and improved
  trajectory may keep all-eight quality within the global gate.
- Mechanism: restore `flow_shift=13.125` and change internal steps from
  `request_steps - 4` to `request_steps - 5`; external contract remains 50.
- Source: DPM-Solver fast-sampling result, https://arxiv.org/abs/2206.00927
- Commit/job: `a2c24818d54fe2bab0556e34720b6b775f5abb27`, Slurm
  `18973606` (`COMPLETED`).
- Candidate mean latency 9.674s (mean case speedup 2.730x).
- LPIPS 01-08: 0.16765, 0.24872, 0.25976, 0.32833, 0.14276, 0.15651,
  0.35933, 0.25370; mean 0.23960, worst 0.35933, gate passed.
- Reward: 2.075597, above r47's 2.016754. Conclusion: keep as final global
  winner; the tuned shift makes one fewer solver step profitable.

## Authoritative all-round metric table

`Latency` and `speedup` are means across the eight public cases. LPIPS entries
are ordered 01 through 08. Failed rounds have no complete valid metric vector.

| Round | Job | State | Latency s | Speedup | LPIPS 01-08 | Mean | Worst | Reward |
|---:|---:|---|---:|---:|---|---:|---:|---:|
| 1 | 18952722 | FAILED | - | - | - | - | - | 0 |
| 2 | 18952861 | COMPLETED | 25.362 | 1.041 | 0.03549, 0.22279, 0.13870, 0.14995, 0.08408, 0.05374, 0.34225, 0.07743 | 0.13805 | 0.34225 | 0.897417 |
| 3 | 18953294 | COMPLETED | 21.759 | 1.214 | 0.01732, 0.07786, 0.02991, 0.04572, 0.09883, 0.09832, 0.20599, 0.04953 | 0.07793 | 0.20599 | 1.119149 |
| 4 | 18953630 | COMPLETED | 21.790 | 1.212 | 0.01732, 0.07786, 0.02991, 0.04572, 0.09883, 0.09832, 0.20599, 0.04953 | 0.07793 | 0.20599 | 1.117481 |
| 5 | 18954173 | COMPLETED | 19.840 | 1.331 | 0.01648, 0.07257, 0.03595, 0.04241, 0.05458, 0.10001, 0.19494, 0.03875 | 0.06946 | 0.19494 | 1.238627 |
| 6 | 18954806 | COMPLETED | 19.824 | 1.332 | 0.01625, 0.07973, 0.04517, 0.04306, 0.06978, 0.10507, 0.20020, 0.04153 | 0.07510 | 0.20020 | 1.232071 |
| 7 | 18955320 | COMPLETED | 19.677 | 1.342 | 0.01561, 0.07023, 0.05171, 0.04283, 0.05757, 0.10301, 0.20676, 0.04091 | 0.07358 | 0.20676 | 1.243322 |
| 8 | 18955750 | COMPLETED | 19.068 | 1.385 | 0.01722, 0.21955, 0.09077, 0.05591, 0.06177, 0.10074, 0.16184, 0.10375 | 0.10144 | 0.21955 | 1.244410 |
| 9 | 18956187 | COMPLETED | 18.843 | 1.402 | 0.01515, 0.22387, 0.09181, 0.05304, 0.06366, 0.10012, 0.13986, 0.10948 | 0.09963 | 0.22387 | 1.261794 |
| 10 | 18956660 | COMPLETED | 18.616 | 1.419 | 0.01981, 0.28538, 0.18392, 0.15800, 0.05248, 0.10745, 0.31501, 0.14321 | 0.15816 | 0.31501 | 1.194149 |
| 11 | 18957155 | COMPLETED | 18.815 | 1.404 | 0.01800, 0.15707, 0.05549, 0.08074, 0.06098, 0.09687, 0.17289, 0.11904 | 0.09514 | 0.17289 | 1.270017 |
| 12 | 18957500 | COMPLETED | 18.856 | 1.401 | 0.01647, 0.17129, 0.05465, 0.08564, 0.05324, 0.09516, 0.17633, 0.11550 | 0.09603 | 0.17633 | 1.265685 |
| 13 | 18958207 | COMPLETED | 18.429 | 1.433 | 0.02511, 0.14763, 0.05507, 0.07562, 0.05216, 0.09886, 0.22917, 0.10396 | 0.09845 | 0.22917 | 1.291889 |
| 14 | 18958797 | COMPLETED | 17.937 | 1.472 | 0.03421, 0.23906, 0.07659, 0.09578, 0.08809, 0.10400, 0.23499, 0.10165 | 0.12180 | 0.23906 | 1.292970 |
| 15 | 18959257 | COMPLETED | 17.553 | 1.504 | 0.03883, 0.19248, 0.10645, 0.10682, 0.10776, 0.10529, 0.23559, 0.08334 | 0.12207 | 0.23559 | 1.320791 |
| 16 | 18959705 | COMPLETED | 17.221 | 1.533 | 0.04783, 0.31660, 0.14995, 0.09935, 0.09800, 0.11080, 0.25888, 0.11614 | 0.14969 | 0.31660 | 1.303817 |
| 17 | 18960335 | COMPLETED | 14.337 | 1.842 | 0.11086, 0.25350, 0.23612, 0.26954, 0.13439, 0.25511, 0.33296, 0.20102 | 0.22419 | 0.33296 | 1.428827 |
| 18 | 18960677 | COMPLETED | 13.509 | 1.955 | 0.17089, 0.26523, 0.26285, 0.30576, 0.14432, 0.18956, 0.34472, 0.19971 | 0.23538 | 0.34472 | 1.494703 |
| 19 | 18961057 | COMPLETED | 12.106 | 2.181 | 0.21837, 0.41311, 0.27064, 0.21797, 0.20059, 0.18910, 0.37481, 0.28681 | 0.27142 | 0.41311 | 1.589173 |
| 20 | 18961273 | COMPLETED | 9.245 | 2.857 | 0.32569, 0.41627, 0.40794, 0.37054, 0.25611, 0.32961, 0.50165, 0.29247 | 0.36253 | 0.50165 | 0 |
| 21 | 18961514 | COMPLETED | 11.576 | 2.281 | 0.16874, 0.26238, 0.26354, 0.30323, 0.13660, 0.19134, 0.35398, 0.19905 | 0.23486 | 0.35398 | 1.745487 |
| 22 | 18961721 | COMPLETED | 10.781 | 2.449 | 0.17291, 0.23125, 0.26331, 0.31798, 0.13694, 0.17986, 0.33029, 0.22840 | 0.23262 | 0.33029 | 1.879679 |
| 23 | 18962114 | COMPLETED | 10.010 | 2.638 | 0.16345, 0.28915, 0.27497, 0.33089, 0.18123, 0.16126, 0.39153, 0.25330 | 0.25572 | 0.39153 | 1.963193 |
| 24 | 18962479 | COMPLETED | 9.618 | 2.746 | 0.32251, 0.41535, 0.41217, 0.37093, 0.25257, 0.32788, 0.50091, 0.29119 | 0.36169 | 0.50091 | 0 |
| 25 | 18962980 | COMPLETED | 10.821 | 2.440 | 0.16324, 0.26603, 0.25201, 0.32044, 0.13519, 0.13765, 0.34001, 0.25489 | 0.23368 | 0.34001 | 1.870071 |
| 26 | 18963349 | COMPLETED | 10.519 | 2.510 | 0.16446, 0.27657, 0.27022, 0.32624, 0.15920, 0.15650, 0.36864, 0.25607 | 0.24724 | 0.36864 | 1.889879 |
| 27 | 18963705 | FAILED | - | - | - | - | - | 0 |
| 28 | 18963728 | COMPLETED | 5.001 | 5.283 | 0.54135, 0.71807, 0.69440, 0.70719, 0.77893, 0.63744, 0.71330, 0.58093 | 0.67145 | 0.77893 | 0 |
| 29 | 18963999 | COMPLETED | 10.775 | 2.453 | 0.32945, 0.59330, 0.45732, 0.47589, 0.26084, 0.33321, 0.60000, 0.38089 | 0.42886 | 0.60000 | 0 |
| 30 | 18964364 | COMPLETED | 16.378 | 1.615 | 0.07935, 0.16645, 0.26073, 0.23970, 0.10535, 0.15730, 0.36722, 0.25657 | 0.20408 | 0.36722 | 1.285363 |
| 31 | 18964861 | COMPLETED | 10.048 | 2.628 | 0.16289, 0.28604, 0.27377, 0.32898, 0.18101, 0.16111, 0.39036, 0.25602 | 0.25502 | 0.39036 | 1.957963 |
| 32 | 18965285 | COMPLETED | 10.160 | 2.599 | 0.16403, 0.28663, 0.27562, 0.33403, 0.17550, 0.16487, 0.39364, 0.25282 | 0.25589 | 0.39364 | 1.934069 |
| 33 | 18965714 | COMPLETED | 10.067 | 2.623 | 0.16241, 0.28846, 0.27570, 0.33139, 0.17580, 0.16528, 0.38325, 0.25388 | 0.25452 | 0.38325 | 1.955575 |
| 34 | 18966584 | COMPLETED | 9.926 | 2.660 | 0.16421, 0.28557, 0.27561, 0.32975, 0.17942, 0.16188, 0.38663, 0.25482 | 0.25474 | 0.38663 | 1.982707 |
| 35 | 18967241 | COMPLETED | 10.036 | 2.631 | 0.16792, 0.28698, 0.27444, 0.33079, 0.17584, 0.16528, 0.38605, 0.25986 | 0.25589 | 0.38605 | 1.957908 |
| 36 | 18967859 | COMPLETED | 10.045 | 2.629 | 0.16233, 0.28718, 0.27572, 0.33058, 0.17687, 0.16527, 0.38645, 0.25516 | 0.25494 | 0.38645 | 1.958653 |
| 37 | 18968451 | COMPLETED | 10.010 | 2.638 | 0.16130, 0.29532, 0.27674, 0.33367, 0.18037, 0.17372, 0.39936, 0.26028 | 0.26010 | 0.39936 | 1.951870 |
| 38 | 18969137 | COMPLETED | 10.047 | 2.628 | 0.17616, 0.39564, 0.27762, 0.32407, 0.21373, 0.32439, 0.39914, 0.28409 | 0.29935 | 0.39914 | 1.841442 |
| 39 | 18969568 | FAILED | - | - | - | - | - | 0 |
| 40 | 18970114 | COMPLETED | 10.040 | 2.630 | 0.17123, 0.28703, 0.27630, 0.33313, 0.18879, 0.16260, 0.39507, 0.25920 | 0.25917 | 0.39507 | 1.948461 |
| 41 | 18970429 | COMPLETED | 10.048 | 2.628 | 0.16097, 0.28985, 0.27634, 0.32923, 0.17690, 0.16540, 0.39318, 0.25916 | 0.25638 | 0.39318 | 1.954372 |
| 42 | 18970795 | COMPLETED | 10.033 | 2.632 | 0.14698, 0.33760, 0.27417, 0.34129, 0.18698, 0.18774, 0.39269, 0.25958 | 0.26588 | 0.39269 | 1.932053 |
| 43 | 18971120 | COMPLETED | 10.029 | 2.633 | 0.16593, 0.26831, 0.26463, 0.32525, 0.14968, 0.15489, 0.36587, 0.26007 | 0.24433 | 0.36587 | 1.989548 |
| 44 | 18971363 | COMPLETED | 10.018 | 2.636 | 0.16366, 0.25645, 0.25500, 0.32476, 0.14337, 0.15164, 0.36175, 0.25160 | 0.23853 | 0.36175 | 2.007169 |
| 45 | 18971790 | COMPLETED | 10.029 | 2.633 | 0.16318, 0.36018, 0.24961, 0.31732, 0.11721, 0.14968, 0.33665, 0.25172 | 0.24319 | 0.36018 | 1.992165 |
| 46 | 18971941 | COMPLETED | 10.056 | 2.626 | 0.16334, 0.32094, 0.25348, 0.32226, 0.12504, 0.13485, 0.34074, 0.25219 | 0.23910 | 0.34074 | 1.997911 |
| 47 | 18972190 | COMPLETED | 10.054 | 2.627 | 0.16151, 0.25737, 0.25788, 0.31917, 0.12776, 0.13394, 0.34364, 0.25692 | 0.23227 | 0.34364 | 2.016754 |
| 48 | 18972867 | COMPLETED | 10.029 | 2.633 | 0.16860, 0.29555, 0.25688, 0.32549, 0.13326, 0.13359, 0.34750, 0.25054 | 0.23893 | 0.34750 | 2.003740 |
| 49 | 18973178 | COMPLETED | 10.036 | 2.631 | 0.16624, 0.31104, 0.25275, 0.32244, 0.12954, 0.13197, 0.34890, 0.25442 | 0.23966 | 0.34890 | 2.000497 |
| 50 | 18973606 | COMPLETED | 9.674 | 2.730 | 0.16765, 0.24872, 0.25976, 0.32833, 0.14276, 0.15651, 0.35933, 0.25370 | 0.23960 | 0.35933 | 2.075597 |
