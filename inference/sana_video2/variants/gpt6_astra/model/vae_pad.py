"""Exact temporal replication padding directly into channels-last storage."""
import torch
import triton
import triton.language as tl

@triton.jit
def _pad(X,Y,N:tl.constexpr,C:tl.constexpr,T:tl.constexpr,H:tl.constexpr,W:tl.constexpr,TO:tl.constexpr,LEFT:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,S4:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    if N>=2147483648:pid=pid.to(tl.int64)
    i=pid*BLOCK+tl.arange(0,BLOCK)
    c=i%C;w=(i//C)%W;h=(i//(C*W))%H;t=(i//(C*W*H))%TO;b=i//(C*W*H*TO)
    ti=tl.minimum(tl.maximum(t-LEFT,0),T-1)
    x=tl.load(X+b*S0+c*S1+ti*S2+h*S3+w*S4,i<N,0)
    tl.store(Y+i,x,i<N)

def pad(x,left,right):
    b,c,t,h,w=x.shape
    if left==right==0:return x.contiguous(memory_format=torch.channels_last_3d)
    y=torch.empty((b,c,t+left+right,h,w),device=x.device,dtype=x.dtype,memory_format=torch.channels_last_3d)
    _pad[(triton.cdiv(y.numel(),1024),)](x,y,y.numel(),c,t,h,w,t+left+right,left,*x.stride(),1024)
    return y

def forward(self,x,causal=True):
    k=self.kernel_size[0]
    left=k-1 if causal else (k-1)//2
    right=0 if causal else (k-1)//2
    return self.conv(pad(x,left,right))

def install():
    from .vae import LTX2VideoCausalConv3d
    LTX2VideoCausalConv3d.forward=forward
