# Verification contract

Both models run at 960x544, 24 FPS and 50 steps on one H100 with batch size
one. MiniMax-H3 returns 124 video frames plus stereo 32 kHz audio; SANA returns
121 video frames. Initial model loading, compilation and warmup are excluded
from latency. Text encoding, denoising, video decoding, recurring transfers and
H3 audio decoding are included.

MiniMax-H3 receives one shared first-case warmup, followed by exactly one timed
request per case. SANA's task wrapper defines its repeat policy. CUDA is
synchronized immediately before and after every timed `generate` call.

Spatial similarity uses LPIPS/AlexNet over aligned decoded RGB frames.
Temporal flicker uses positive excess adjacent-frame LPIPS relative to the
teacher, reported as `Excess-tLPx100`.

```text
case_reward = speedup * max(0, 1 - LPIPS) * max(0, 1 - Excess-tLPx100)
```

MiniMax-H3 requires split mean LPIPS <= 0.25, worst-case LPIPS <= 0.30 and
mean Excess-tLPx100 <= 0.10. SANA requires mean LPIPS <= 0.30, worst-case LPIPS
<= 0.50 and mean Excess-tLPx100 <= 0.30. A failed split scores zero.

Three MiniMax-H3 prompts are public. Four additional prompts and their teacher
assets are supplied only to the private verifier. All eight SANA prompts are
visible. Teacher tensors, LPIPS weights and run outputs are never committed.

This local harness is suitable for reproducibility and debugging, but candidate
code currently shares a process boundary with measurement code. It is not a
hardened competitive sandbox until worker/evaluator isolation is supplied by
the deployment platform.

