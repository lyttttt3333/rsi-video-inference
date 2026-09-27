"""Single-GPU Triton fusions ported from Sol-Engine's MiniMax-H3 runtime."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qknorm_partial_rope_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    head_dim,
    rotary_dim,
    half_dim,
    heads,
    seq,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    token = (pid // heads) % seq
    cols = tl.arange(0, BLOCK)
    mask = cols < head_dim
    base = pid * head_dim

    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / head_dim
    inv_rms = tl.math.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = x * inv_rms * weight

    in_rotary = cols < rotary_dim
    in_first_half = cols < half_dim
    partner = tl.where(in_first_half, cols + half_dim, cols - half_dim)
    partner_value = tl.load(x_ptr + base + partner, mask=in_rotary, other=0.0).to(tl.float32)
    partner_weight = tl.load(weight_ptr + partner, mask=in_rotary, other=0.0).to(tl.float32)
    partner_normed = partner_value * inv_rms * partner_weight
    rotated = tl.where(in_first_half, -partner_normed, partner_normed)

    cos = tl.load(cos_ptr + token * rotary_dim + cols, mask=in_rotary, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptr + token * rotary_dim + cols, mask=in_rotary, other=0.0).to(tl.float32)
    out = tl.where(in_rotary, normed * cos + rotated * sin, normed)
    tl.store(out_ptr + base + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def fused_qknorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse per-head RMSNorm and MiniMax-H3's partial rotary embedding."""
    batch, seq, heads, head_dim = x.shape
    rotary_dim = cos.shape[-1]
    if rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError((rotary_dim, head_dim))
    flat = x.reshape(-1, head_dim).contiguous()
    out = torch.empty_like(flat)
    _qknorm_partial_rope_kernel[(flat.shape[0],)](
        out,
        flat,
        weight,
        cos,
        sin,
        head_dim,
        rotary_dim,
        rotary_dim // 2,
        heads,
        seq,
        eps,
        BLOCK=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return out.view(batch, seq, heads, head_dim)


@triton.jit
def _rmsnorm_modulate_kernel(
    out_ptr, x_ptr, weight_ptr, scale_ptr, shift_ptr, index_ptr,
    n_cols, n_index, eps, stride_row, stride_table_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offset = row * stride_row + cols
    table_offset = tl.load(index_ptr + (row % n_index)) * stride_table_row + cols

    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = x * tl.math.rsqrt(variance + eps) * weight
    scale = tl.load(scale_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    out = normed * (1.0 + scale) + shift
    tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _residual_gate_rmsnorm_modulate_kernel(
    hidden_out_ptr, normed_out_ptr,
    residual_ptr, branch_ptr, weight_ptr,
    gate_ptr, scale_ptr, shift_ptr, index_ptr,
    n_cols, n_index, eps, stride_row, stride_table_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offset = row * stride_row + cols
    table_offset = tl.load(index_ptr + (row % n_index)) * stride_table_row + cols

    residual = tl.load(residual_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    hidden = residual + gate * branch
    tl.store(hidden_out_ptr + offset, hidden.to(hidden_out_ptr.dtype.element_ty), mask=mask)

    variance = tl.sum(hidden * hidden, axis=0) / n_cols
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = hidden * tl.math.rsqrt(variance + eps) * weight
    scale = tl.load(scale_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    out = normed * (1.0 + scale) + shift
    tl.store(normed_out_ptr + offset, out.to(normed_out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_kernel(
    out_ptr, x_ptr, n_cols, stride_in_row, stride_out_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    value = tl.load(x_ptr + row * stride_in_row + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(x_ptr + row * stride_in_row + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    out = value * (gate * tl.sigmoid(gate))
    tl.store(out_ptr + row * stride_out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _warps_for(block: int) -> int:
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    return 4


def _row_addressable(table: torch.Tensor) -> torch.Tensor:
    return table if table.stride(-1) == 1 else table.contiguous()


def fused_rmsnorm_modulate(x, weight, scale, shift, index, eps):
    cols = x.shape[-1]
    flat = x.reshape(-1, cols).contiguous()
    scale, shift = _row_addressable(scale), _row_addressable(shift)
    out = torch.empty_like(flat)
    block = _next_pow2(cols)
    _rmsnorm_modulate_kernel[(flat.shape[0],)](
        out, flat, weight, scale, shift, index,
        cols, index.numel(), eps, flat.stride(0), scale.stride(0),
        BLOCK=block, num_warps=_warps_for(block),
    )
    return out.view_as(x)


def fused_residual_gate_rmsnorm_modulate(residual, branch, gate, weight, scale, shift, index, eps):
    cols = residual.shape[-1]
    res_flat = residual.reshape(-1, cols).contiguous()
    br_flat = branch.reshape(-1, cols).contiguous()
    gate, scale, shift = (_row_addressable(tensor) for tensor in (gate, scale, shift))
    hidden = torch.empty_like(res_flat)
    normed = torch.empty_like(res_flat)
    block = _next_pow2(cols)
    _residual_gate_rmsnorm_modulate_kernel[(res_flat.shape[0],)](
        hidden, normed, res_flat, br_flat, weight, gate, scale, shift, index,
        cols, index.numel(), eps, res_flat.stride(0), gate.stride(0),
        BLOCK=block, num_warps=_warps_for(block),
    )
    return hidden.view_as(residual), normed.view_as(residual)


def fused_swiglu(x):
    cols = x.shape[-1] // 2
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    out = x.new_empty((flat.shape[0], cols))
    block = _next_pow2(cols)
    _swiglu_kernel[(flat.shape[0],)](
        out, flat, cols, flat.stride(0), out.stride(0),
        BLOCK=block, num_warps=_warps_for(block),
    )
    return out.view(*x.shape[:-1], cols)
