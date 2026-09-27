# Draft verification status

Historical status from the initial native-baseline draft. The H3 baseline and
measurement contract changed on 2026-09-19; see VERIFICATION.md. Prior H3
adapter-import checks and calibration claims do not validate the new Diffusers
baseline.

## Passed

- Seven standard-library unit tests: aggregate 12 GPU-hour cap, positive explicit
  allocation, configuration updates, timing boundaries, warmup exclusion,
  non-finite timing rejection and empty submission rejection.
- H3 persistent adapter import with Diffusers absent, CPU Slurm job 18815378.
- SANA persistent adapter import with Diffusers absent, CPU Slurm job 18815379.
- Official static checks: layout, agent Docker references/isolation, baseline
  and evaluator presence, canonical paths, network policy, GPU types, dependency
  declarations, Docker sanity, submission contract, canonical instruction notice,
  canaries and separate verifier environment.
- Followup static checks: validation/test JSON output contract and integrity
  manifests for both tasks, CPU Slurm job 18815754.

## Intentionally unresolved (official static suite is NOT fully green)

- Agent GPU-hour limit and timer: no per-task budget allocation yet.
- Aggregate reward definition and baseline metadata: quality criteria and at
  least three measured calibration runs per split are pending; no invented scores.
- Author and organization metadata: unspecified.

Full initial report: `reports/static.md`; followup report:
`reports/static-followup.md`. Initial output-contract findings are superseded
by the passing followup checks.

## Not performed / not claimed

- No GPU jobs or agent trials for these new packages.
- No GPU acceptance of the persistent CPU-resident/offloaded adapters.
- No measured hot baseline, quality acceptance or final score.
- No assets copy, new container build, registry upload or end-to-end Harbor run.
- No adversarially hardened verifier. The current harness is diagnostic only.

The shared research budget remains unallocated: 43,200 GPU seconds total,
no launch reservations. Existing standalone video acceptance does not substitute
for validation of the new persistent interfaces.
