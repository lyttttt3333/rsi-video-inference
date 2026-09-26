# SANA held-out and anti-cheat review

This task has eight public optimization cases and eight private held-out cases.
The public JSON is available to the agent through the environment image. The
held-out JSON is copied only into the verifier image at
`/opt/private-eval/heldout-cases.json`, with a root-only directory (`0700`) and
file (`0600`). It is not copied into the agent image or the submission bundle.

The verifier runs a trusted parent process. Candidate code executes as UID
65534 in a worker and receives one request at a time. The parent owns the
monotonic timer, validates and stores tensors, and writes the reward report;
candidate code cannot edit those files. The private baseline, teacher tensors,
metric weights, scoring code, and held-out case list are root-only. The public
worker runs before the trusted baseline, and each private request is evaluated
before its temporary one-request file is deleted.

This is an integrity/generalization control, not a claim that metric reward
hacking is impossible. The LPIPS, Excess-tLP and quality gates remain explicit
task metrics. The short malicious candidate trial is recorded in
`adversarial_trial.json`: it was rejected with `invalid=1` after the report
directory permission fix.
