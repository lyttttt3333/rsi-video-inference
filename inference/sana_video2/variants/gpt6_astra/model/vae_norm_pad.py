"""Fused BF16 RMS normalization, SiLU, and temporal replication padding."""
import types
import torch
import triton
import triton.language as tl

@triton.jit
def _norm_pad(X,Y,C:tl.constexpr,T:tl.constexpr,H:tl.constexpr,W:tl.constexpr,TO:tl.constexpr,LEFT:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,S4:tl.constexpr,EPS:tl.constexpr,BC:tl.constexpr,BP:tl.constexpr):
    b=tl.program_id(1).to(tl.int64)
    p=tl.program_id(0).to(tl.int64)*BP+tl.arange(0,BP)
    c=tl.arange(0,BC)
    t=p//(H*W);h=(p//W)%H;w=p%W
    ti=tl.minimum(tl.maximum(t-LEFT,0),T-1)
    valid=(p[:,None]<TO*H*W)&(c[None,:]<C)
    addr=b*S0+c[None,:]*S1+ti[:,None]*S2+h[:,None]*S3+w[:,None]*S4
    x=tl.load(X+addr,valid,0).to(tl.float32)
    sq=(x*x).to(tl.bfloat16).to(tl.float32)
    mean=(tl.sum(sq,1)/C).to(tl.bfloat16).to(tl.float32)
    var=(mean+EPS).to(tl.bfloat16).to(tl.float32)
    rms=tl.sqrt(var).to(tl.bfloat16).to(tl.float32)
    z=(x/rms[:,None]).to(tl.bfloat16).to(tl.float32)
    y=z*tl.sigmoid(z)
    out=b*C*TO*H*W+p[:,None]*C+c[None,:]
    tl.store(Y+out,y,valid)

def norm_pad(x,eps,left,right):
    b,c,t,h,w=x.shape;to=t+left+right
    y=torch.empty((b,c,to,h,w),device=x.device,dtype=x.dtype,memory_format=torch.channels_last_3d)
    bp=4 if c>=512 else 16
    _norm_pad[(triton.cdiv(to*h*w,bp),b)](x,y,c,t,h,w,to,left,*x.stride(),eps,triton.next_power_of_2(c),bp,enable_fp_fusion=False)
    return y

def conv_norm(x,norm,conv,causal):
    k=conv.kernel_size[0];left=k-1 if causal else (k-1)//2;right=0 if causal else (k-1)//2
    return conv.conv(norm_pad(x,norm.eps,left,right))

def forward(self,inputs,temb=None,generator=None,causal=True):
    x=conv_norm(inputs,self.norm1,self.conv1,causal)
    x=conv_norm(x,self.norm2,self.conv2,causal)
    if self.norm3 is not None:inputs=self.norm3(inputs.movedim(1,-1)).movedim(-1,1)
    if self.conv_shortcut is not None:inputs=self.conv_shortcut(inputs)
    return x+inputs

def install(vae):
    from .vae import LTX2VideoResnetBlock3d
    for m in vae.decoder.modules():
        if isinstance(m,LTX2VideoResnetBlock3d) and m.scale_shift_table is None and m.per_channel_scale1 is None and m.per_channel_scale2 is None and (not m.training or m.dropout.p==0):
            m.forward=types.MethodType(forward,m)
    d=vae.decoder
    if d.time_embedder is None:
        norm=d.norm_out
        d.norm_out.forward=lambda x,channel_dim=None:x
        d.conv_act.forward=lambda x:x
        def final_conv(self,x,causal=True):return conv_norm(x,norm,self,causal)
        d.conv_out.forward=types.MethodType(final_conv,d.conv_out)
