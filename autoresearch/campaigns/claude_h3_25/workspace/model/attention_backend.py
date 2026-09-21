"""Fused-attention backend selection for the packed MiniMax-H3 sequence.

The baseline pins SDPA to `FLASH_ATTENTION`. On the packed sequence that kernel
runs at roughly a third of the H100's bf16 peak, and it is 59.5% of one
transformer forward, so which fused backend and which input layout is used is
worth more than every elementwise fusion in the block put together.

`select` benchmarks the backends that are actually built into this torch on the
real shape and stride pattern of the block's query/key/value, and pins the
winner for the rest of the process. It is called once, from the untimed warmup
generation, so no timed request pays for the search.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# (name, backend). Ordered by expectation, not by preference: the measurement
# decides. `EFFICIENT_ATTENTION` is the non-fused fallback and is only a useful
# candidate because it is the one backend guaranteed to exist.
_CANDIDATE_BACKENDS = [
    ('cudnn', getattr(SDPBackend, 'CUDNN_ATTENTION', None)),
    ('flash', SDPBackend.FLASH_ATTENTION),
    ('efficient', SDPBackend.EFFICIENT_ATTENTION),
]

# How the (batch, heads, seq, head_dim) views handed to SDPA are materialised.
# The block produces contiguous query/key out of the rope fusion and a strided
# `value` slice of the fused qkv projection, so 'value' and 'all' are distinct.
_CANDIDATE_LAYOUTS = ('none', 'value', 'all')

_PLAN = ('flash', SDPBackend.FLASH_ATTENTION, 'none')
_DEFAULT_PLAN = _PLAN
# A calibrated plan is only trusted on sequences at least this long. The token
# refiner runs this same processor over a ~200-row sequence, and calibration has
# no real activations from it; a backend that is silently wrong on a short,
# ragged shape would corrupt the round with nothing in the log to show it.
_PLAN_MIN_LENGTH = 0

# ...and only from this denoising step onward. Round 13 ran cuDNN over the whole
# schedule: 2.64x on case 02, and LPIPS 0.0951 -> 0.3030 against a 0.30 gate.
# The cache rounds had already measured that the first few steps of this
# schedule decide the trajectory and the rest only refine it -- reusing step 4
# costs 0.154 LPIPS and reusing steps 36-45 costs 0.0007 -- so the same split
# applies to a kernel that is 2.9e-4 different rather than 100% absent.
_PLAN_MIN_STEP = 0
_STEP = 0

# ...with one exception, which is what round 21 measures. `_PLAN_MIN_STEP`
# treats a denoising step as indivisible: either all fifty blocks run the
# baseline kernel or all fifty run the calibrated one. Nine of the fifteen
# forwards a request still computes fall on the protected side of that line and
# cost 2.91 s each against 2.28 s, so 5.7 s of a 47.8 s request is the price of
# protecting the early steps. The question this knob asks is whether the
# sensitivity that price buys is spread evenly through the stack or concentrated
# near its input: past `_PLAN_MIN_LAYER`, a protected step runs the calibrated
# kernel anyway. Round 19 established that *staleness* gets worse with depth;
# whether a 2.9e-4 kernel difference does is a different question, because it is
# injected rather than propagated from a stale argument.
_PLAN_MIN_LAYER = 10**9


def set_denoise_step(step: int) -> None:
    global _STEP
    _STEP = int(step)


def set_plan_min_step(step: int) -> None:
    global _PLAN_MIN_STEP
    _PLAN_MIN_STEP = int(step)


def set_plan_min_layer(layer: int) -> None:
    global _PLAN_MIN_LAYER
    _PLAN_MIN_LAYER = int(layer) if layer >= 0 else 10**9


# Set by `arm_calibration`; consumed by the first `attention` call that is long
# enough to be the real packed sequence.
_CALIBRATION: dict | None = None


def plan() -> tuple[str, str]:
    return _PLAN[0], _PLAN[2]


def arm_calibration(**settings) -> None:
    """Calibrate the backend on the next real block-0 query/key/value."""
    global _CALIBRATION
    _CALIBRATION = settings


def disarm_calibration() -> None:
    global _CALIBRATION
    _CALIBRATION = None


def _prepare(query, key, value, layout):
    query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    if layout == 'all':
        return query.contiguous(), key.contiguous(), value.contiguous()
    if layout == 'value':
        return query, key, value.contiguous()
    return query, key, value


_FALLBACK = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, layer=-1):
    """Full attention over `(batch, seq, heads, head_dim)` query/key/value.

    The plan is chosen on the block stack's shape; the token refiner runs the
    same processor on a two-hundred-row sequence, so a backend that rejects the
    short shape must not take the round down with it.
    """
    global _CALIBRATION
    if _CALIBRATION is not None and query.shape[1] >= _CALIBRATION.get('min_length', 4096):
        settings = _CALIBRATION
        _CALIBRATION = None
        try:
            calibrate(query, key, value, **settings)
        except Exception as error:
            print(f'H3_ATTENTION_CALIBRATE fail:{type(error).__name__}:{error}', flush=True)
            torch.cuda.empty_cache()
    calibrated = query.shape[1] >= _PLAN_MIN_LENGTH and (
        _STEP >= _PLAN_MIN_STEP or (layer >= 0 and layer >= _PLAN_MIN_LAYER))
    _, backend, layout = _PLAN if calibrated else _DEFAULT_PLAN
    query, key, value = _prepare(query, key, value, layout)
    try:
        with sdpa_kernel(backend):
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
            )
    except RuntimeError:
        with sdpa_kernel(_FALLBACK):
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
            )
    return hidden_states.transpose(1, 2)


def _time(function, repeats):
    function()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / repeats


@torch.inference_mode()
def diagnose(sequence_length, heads, head_dim, device, dtype=torch.bfloat16, rows=192):
    """Report each backend's error against a float32 reference at the real length.

    Round 3 measured cuDNN at 1.75x FlashAttention's throughput and, on case 02,
    an LPIPS of 0.508 against the dense reference. Round 4's version of this
    check said cuDNN and flash are equally accurate — but it ran at sequence
    length 4096 while the model runs at 19487, so it tested a shape the model
    never sees, and its verdict is incompatible with FirstBlockCache scoring
    0.094 at the same time.

    A float32 reference over the whole 19487-row sequence would materialise a
    19487^2 attention matrix (1.5 TB), so the reference is taken over three
    slices of query rows -- the head, the middle and, most importantly, the tail
    of the sequence -- against the full key/value. A kernel that mishandles the
    ragged last tile of an odd-length sequence shows up in the tail slice and
    nowhere else.
    """
    inner = heads * head_dim
    fused = torch.randn(1, sequence_length, 3 * inner, device=device, dtype=dtype)
    value = fused.chunk(3, dim=-1)[2].unflatten(-1, (heads, head_dim))
    query = torch.randn(1, sequence_length, heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(1, sequence_length, heads, head_dim, device=device, dtype=dtype)

    rows = min(rows, sequence_length // 3)
    middle = (sequence_length - rows) // 2
    slices = {'head': slice(0, rows),
              'mid': slice(middle, middle + rows),
              'tail': slice(sequence_length - rows, sequence_length)}

    k32 = key.transpose(1, 2).float()
    v32 = value.transpose(1, 2).float()
    references = {}
    for tag, window in slices.items():
        q32 = query[:, window].transpose(1, 2).float()
        with sdpa_kernel(SDPBackend.MATH):
            references[tag] = F.scaled_dot_product_attention(q32, k32, v32, dropout_p=0.0,
                                                             is_causal=False)
        del q32
    del k32, v32
    torch.cuda.empty_cache()

    report = []
    for name, backend in _CANDIDATE_BACKENDS:
        if backend is None:
            continue
        try:
            q, k, v = _prepare(query, key, value, 'none')
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            for tag, window in slices.items():
                reference = references[tag]
                error = (out[:, :, window].float() - reference).abs()
                scale = reference.abs().mean().clamp_min(1.0e-12)
                report.append(f'{name}/{tag}=mean{float(error.mean() / scale):.3e}'
                              f'/max{float(error.max() / scale):.3e}')
            del out
        except Exception as error:
            report.append(f'{name}=fail:{type(error).__name__}')
    del fused, value, query, key, references
    torch.cuda.empty_cache()
    print(f'H3_ATTENTION_ERROR seq={sequence_length} rows={rows} relative_to_float32 '
          + ' '.join(report), flush=True)


def _slice_errors(candidate, reference, window):
    error = (candidate[:, :, window].float() - reference).abs()
    scale = reference.abs().mean().clamp_min(1.0e-12)
    return float(error.mean() / scale), float(error.max() / scale)


@torch.inference_mode()
def calibrate(query, key, value, tolerance=1.5, rows=96, repeats=3, min_length=4096):
    """Pick a backend using the *model's own* query/key/value, not random data.

    Round 3 ran cuDNN's fused attention, which is 1.75x FlashAttention on this
    shape because FA2 does not use Hopper's asynchrony, and case 02 came back at
    LPIPS 0.508 -- not a numerically slightly different video, a wrong one.
    Round 4's check then reported cuDNN and flash as equally accurate, and round
    11 confirmed that at the real sequence length too (2.274e-3 against
    2.271e-3). Both of those checks ran on `torch.randn`. Random keys make every
    softmax row nearly uniform; real ones are dominated by a few entries, and a
    kernel that mishandles that has nothing to show on random input.

    So this measures the thing that actually matters: each backend's error
    against a float32 reference computed from the real activations of block 0,
    over the head, middle and tail rows of the real sequence. A backend is
    eligible only if its worst relative error is within `tolerance` of the
    FlashAttention backend's -- flash is the baseline's own kernel, so matching
    it is by definition enough -- and the fastest eligible backend wins. If
    cuDNN is broken on real activations, this rejects it and the round costs
    nothing; if it is not, the round is worth 1.75x on 59.7% of every computed
    forward.

    A full float32 reference would need a 19487^2 attention matrix per head, so
    the reference is taken over `rows` query rows at each of three positions
    against the full key/value.
    """
    sequence_length = query.shape[1]
    heads = query.shape[2]
    rows = max(8, min(int(rows), sequence_length // 3))
    middle = (sequence_length - rows) // 2
    slices = {'head': slice(0, rows),
              'mid': slice(middle, middle + rows),
              'tail': slice(sequence_length - rows, sequence_length)}

    k32 = key.transpose(1, 2).float()
    v32 = value.transpose(1, 2).float()
    references = {}
    for tag, window in slices.items():
        q32 = query[:, window].transpose(1, 2).float()
        with sdpa_kernel(SDPBackend.MATH):
            references[tag] = F.scaled_dot_product_attention(q32, k32, v32, dropout_p=0.0,
                                                             is_causal=False)
        del q32
    del k32, v32
    torch.cuda.empty_cache()

    def execute(backend, layout):
        q, k, v = _prepare(query, key, value, layout)
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    # FlashAttention is the baseline's own kernel: its output *is* the reference
    # trajectory, and its error against float32 is the budget everything else
    # gets measured in.
    flash_out = execute(SDPBackend.FLASH_ATTENTION, 'none')
    reference_error = max(_slice_errors(flash_out, references[tag], window)[1]
                          for tag, window in slices.items())
    flash_scale = flash_out.abs().mean().clamp_min(1.0e-12)

    measured = {}
    report = []
    for name, backend in _CANDIDATE_BACKENDS:
        if backend is None:
            continue
        for layout in _CANDIDATE_LAYOUTS:
            try:
                out = execute(backend, layout)
                worst = max(_slice_errors(out, references[tag], window)[1]
                            for tag, window in slices.items())
                # Against flash over the *whole* output, not three slices of it.
                # This is the sensitive one: it sees every row and every head,
                # and it is what a kernel that is subtly different rather than
                # broken would show up in.
                divergence = float((out.float() - flash_out.float()).abs().mean() / flash_scale)
                del out
                milliseconds = _time(lambda: execute(backend, layout), repeats)
            except Exception as error:
                report.append(f'{name}/{layout}=fail:{type(error).__name__}')
                continue
            measured[(name, layout)] = (milliseconds, worst, divergence)
            report.append(f'{name}/{layout}={milliseconds:.2f}ms'
                          f'/err{worst:.3e}/div{divergence:.3e}')
    del references, flash_out
    torch.cuda.empty_cache()

    budget = tolerance * reference_error
    eligible = [(milliseconds, name, layout)
                for (name, layout), (milliseconds, worst, divergence) in measured.items()
                if worst <= budget and divergence <= budget]
    chosen = min(eligible) if eligible else None
    if chosen is not None:
        _, name, layout = chosen
        globals()['_PLAN'] = (name, dict(_CANDIDATE_BACKENDS)[name], layout)
        # Every case's packed sequence is within 1% of this one's length, and
        # the refiner's is two orders of magnitude shorter, so half of this
        # length separates them with room to spare.
        globals()['_PLAN_MIN_LENGTH'] = sequence_length // 2
    print(f'H3_ATTENTION_CALIBRATE seq={sequence_length} heads={heads} rows={rows} '
          f'tolerance={tolerance} flash_err={reference_error:.3e} chosen={_PLAN[0]}/{_PLAN[2]} '
          f'min_length={_PLAN_MIN_LENGTH} ' + ' '.join(report), flush=True)


@torch.inference_mode()
def select(sequence_length, heads, head_dim, device, dtype=torch.bfloat16, repeats=3,
           allowed=None):
    """Benchmark every available backend/layout pair and pin the fastest.

    `allowed` restricts the search to named backends, so a round can hold the
    numerics fixed while still picking the best layout.
    """
    global _PLAN
    inner = heads * head_dim
    # `value` is a chunk of the fused qkv projection: same element stride, three
    # times the row stride. Reproduce that rather than benchmarking a layout the
    # block never produces.
    fused = torch.randn(1, sequence_length, 3 * inner, device=device, dtype=dtype)
    value = fused.chunk(3, dim=-1)[2].unflatten(-1, (heads, head_dim))
    query = torch.randn(1, sequence_length, heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(1, sequence_length, heads, head_dim, device=device, dtype=dtype)
    scale = head_dim**-0.5

    results = []
    for name, backend in _CANDIDATE_BACKENDS:
        if backend is None or (allowed is not None and name not in allowed):
            continue
        for layout in _CANDIDATE_LAYOUTS:
            try:
                def run(backend=backend, layout=layout):
                    q, k, v = _prepare(query, key, value, layout)
                    with sdpa_kernel(backend):
                        F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                                       scale=scale)

                milliseconds = _time(run, repeats)
            except Exception as error:
                results.append((float('inf'), name, layout, f'{type(error).__name__}'))
                continue
            results.append((milliseconds, name, layout, 'ok'))

    results.sort()
    usable = [row for row in results if row[0] != float('inf')]
    if usable:
        milliseconds, name, layout, _ = usable[0]
        backend = dict((n, b) for n, b in _CANDIDATE_BACKENDS)[name]
        _PLAN = (name, backend, layout)
    del fused, value, query, key
    torch.cuda.empty_cache()
    summary = ' '.join(f'{name}/{layout}={"fail:" + note if note != "ok" else f"{value:.2f}ms"}'
                       for value, name, layout, note in results)
    print(f'H3_ATTENTION_SELECT seq={sequence_length} heads={heads} head_dim={head_dim} '
          f'chosen={_PLAN[0]}/{_PLAN[2]} {summary}', flush=True)
    return _PLAN[0], _PLAN[2]
