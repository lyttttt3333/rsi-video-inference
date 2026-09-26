"""Optional dynamic FP8 GEMM experiment; original BF16 parameters are retained."""
import torch
from torch import nn
import triton
import triton.language as tl

@triton.jit
def _quant(X,Q,S,N:tl.constexpr,STRIDE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0); i=tl.arange(0,BLOCK)
    x=tl.load(X+row*STRIDE+i,i<N,0).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(x),0)/448.,1e-12)
    tl.store(Q+row*N+i,x/scale,i<N)
    tl.store(S+row,scale)

def quantize(x):
    q=torch.empty(x.shape,device=x.device,dtype=torch.float8_e4m3fn)
    s=torch.empty((x.numel()//x.shape[-1],1),device=x.device,dtype=torch.float32)
    _quant[(s.numel(),)](x,q,s,x.shape[-1],x.stride(-2),triton.next_power_of_2(x.shape[-1]))
    return q,s

class FP8Linear(nn.Module):
    def __init__(self,linear):
        super().__init__()
        self.enabled=True
        self.weight=linear.weight
        self.bias=linear.bias
        self.in_features=linear.in_features
        self.out_features=linear.out_features
        q,s=quantize(linear.weight)
        self.register_buffer('qweight',q.t(),persistent=False)
        self.register_buffer('scale_weight',s.t().contiguous(),persistent=False)
    def forward(self,x):
        if not self.enabled:return torch.nn.functional.linear(x,self.weight,self.bias)
        shape=x.shape
        x=x.reshape(-1,shape[-1]).contiguous()
        q,s=quantize(x)
        out=torch._scaled_mm(q,self.qweight,scale_a=s,scale_b=self.scale_weight,bias=self.bias,out_dtype=torch.bfloat16,use_fast_accum=True)
        return out.reshape(*shape[:-1],self.out_features)

def install_fp8(model,mode):
    for block in model.blocks:
        groups={'mlp':[block.mlp],'self':[block.attn],'cross':[block.cross_attn],'attention':[block.attn,block.cross_attn],'up':[block.mlp],'down':[block.mlp],'up_down':[block.mlp]}.get(mode,[block.mlp,block.attn,block.cross_attn])
        for parent in groups:
            for name,child in list(parent.named_children()):
                if mode=='up' and name!='up_proj':continue
                if mode=='down' and name!='down_proj':continue
                if mode=='up_down' and name not in ('up_proj','down_proj'):continue
                if isinstance(child,nn.Linear) and child.in_features%16==0 and child.out_features%16==0:
                    setattr(parent,name,FP8Linear(child))
