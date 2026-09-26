"""Fuse BF16-rounded SwiGLU with row-wise FP8 activation quantization."""
import torch
import os
SHARED_QUANT=os.environ.get("SHARED_MLP_QUANT","0")=="1"
import triton
import triton.language as tl

@triton.jit
def _swiglu_quant(G,U,Q,S,C:tl.constexpr,STRIDE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK)
    g=tl.load(G+row*STRIDE+i,i<C,0).to(tl.float32)
    u=tl.load(U+row*STRIDE+i,i<C,0).to(tl.float32)
    a=(g*tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    x=(a*u).to(tl.bfloat16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(x),0)/448.,1e-12)
    tl.store(Q+row*C+i,x/scale,i<C)
    tl.store(S+row,scale)

def swiglu_quant(g,u):
    shape=g.shape
    c=shape[-1];rows=g.numel()//c
    q=torch.empty((rows,c),device=g.device,dtype=torch.float8_e4m3fn)
    s=torch.empty((rows,1),device=g.device,dtype=torch.float32)
    _swiglu_quant[(rows,)](g,u,q,s,c,g.stride(-2),triton.next_power_of_2(c),num_warps=8)
    return q,s

def fused_mlp_forward(self,x,**kwargs):
    from .fast_ops import swiglu, swiglu_packed
    if hasattr(self,"gate_up_proj"):
        packed=self.gate_up_proj(x)
        g,u=packed.chunk(2,dim=-1)
    elif SHARED_QUANT and self.gate_proj.enabled and self.up_proj.enabled:
        from .fp8 import quantize
        q,s=quantize(x.reshape(-1,x.shape[-1]).contiguous())
        gate,up=self.gate_proj,self.up_proj
        g=torch._scaled_mm(q,gate.qweight,scale_a=s,scale_b=gate.scale_weight,bias=gate.bias,out_dtype=torch.bfloat16,use_fast_accum=True).reshape(*x.shape[:-1],gate.out_features)
        u=torch._scaled_mm(q,up.qweight,scale_a=s,scale_b=up.scale_weight,bias=up.bias,out_dtype=torch.bfloat16,use_fast_accum=True).reshape(*x.shape[:-1],up.out_features)
    else:
        g,u=self.gate_proj(x),self.up_proj(x)
    down=self.down_proj
    if down.enabled:
        q,s=swiglu_quant(g,u)
        out=torch._scaled_mm(q,down.qweight,scale_a=s,scale_b=down.scale_weight,bias=down.bias,out_dtype=torch.bfloat16,use_fast_accum=True)
        return out.reshape(*g.shape[:-1],down.out_features)
    return down(swiglu_packed(packed) if hasattr(self,"gate_up_proj") else swiglu(g,u))

def install(model):
    import types
    from .fp8 import FP8Linear
    for b in model.blocks:
        if isinstance(b.mlp.down_proj,FP8Linear):
            b.mlp.forward=types.MethodType(fused_mlp_forward,b.mlp)
