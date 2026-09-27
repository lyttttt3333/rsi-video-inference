# Anti-cheat hardening record

The CW report for `be82fb5` found five bypasses in the earlier measurement
protocol. The verifier implementation now addresses them as follows:

| Earlier bypass | Current control |
| --- | --- |
| Teacher tensors were readable | Teacher and aggregate tensors are kept in root-only directories; candidate staging is separate and copied with owner and `O_NOFOLLOW` regular-file checks. |
| A staging symlink could redirect `reward.json` | Reports use a root-created temporary file, `fsync`, and atomic `os.replace`; the final path is never opened through an attacker-controlled symlink. |
| Work could be deferred to `items()` during save | The runtime requires a plain, finite tensor of the exact output shape and serializes it before the supervisor reports completion. Tensor materialization is inside the timed interval. |
| Candidate stdout could forge `generated` | Candidate code runs in a UID/GID 65534 child with stdout/stderr at `/dev/null`; only the root supervisor emits control records, and it independently observes the sealed output file. |
| The private request could be reused as warmup | Every private one-case worker warms the first public request, never its private request. The held-out request is sent only after warmup and is deleted after that worker exits. |

Static checks and the CPU tensor-shim smoke completed successfully (Slurm CPU
jobs `19371546`, `19371334`, and `19371377`; honest `invalid=0`, report mode
`0700`, forged/deferred probes rejected). These jobs do not load SANA weights
or run H100, LPIPS, Modal networking, or the full scoring path. The
owner-controlled CW VM adversarial suite must be rerun against this revision
before recording formal anti-cheat validation.

The Modal image smoke and the fixed-candidate verifier smoke now also pass. The
smoke uses one public and one held-out case on an H100 and returned
`invalid=0`; the GPU-resident candidate took 26.682 s versus 26.662 s for the
trusted baseline, with zero LPIPS and zero Excess-tLP in both smoke cases. The
verifier work tree is placed under a traversable random `/tmp` directory while
reports and aggregate tensors remain root-only under `/logs/verifier`. This is
an environment/integration smoke, not the owner-controlled CW adversarial
certification.
