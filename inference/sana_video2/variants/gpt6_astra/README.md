# GPT-6 Astra SANA-Video-2.0 candidate

This directory contains the source-only candidate produced by the GPT-6 Astra
Modal optimization run. It is kept separate from the default
`inference/sana_video2` runner so the resident baseline remains unchanged.

The benchmark entrypoint is `candidate.py`; it exposes:

```python
runner = candidate.build(pathlib.Path("/opt/weights"))
result = runner.generate(request)
```

The request/output contract is the one in
`tasks/sana-video2-hot-inference`: one H100, batch one, 960x544, 121 frames,
50 sampling steps, and a finite RGB tensor in SANA's `[-1, 1]` range.

The candidate includes the optimized PyTorch/Triton model modules, FP8 paths,
CUDA-graph capture, fused denoiser/VAE operations, and the validated
cross-step reuse schedule. It does not include weights, videos, tensor dumps,
agent trajectories, logs, reward files, or other run records.

To evaluate it, place this directory's `candidate.py` and `model/` under an
otherwise valid `/workspace/submission` bundle and run the task verifier:

```bash
/workspace/validation/val.sh
```

The default native CLI remains `inference/sana_video2/infer_native.py`; this
variant is the RSI benchmark `build()/generate()` implementation.
