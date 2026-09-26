"""Single-block masked text cross attention for short conditioning sequences."""
import torch
import os
LATE_ONLY=os.environ.get("CROSS_FP8_ONLY","0")=="1"
import triton
import triton.language as tl

@triton.jit
def _cross(Q,K,V,M,O,N:tl.constexpr,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
           Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,
           K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,
           V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,
           M0:tl.constexpr,SM:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr):
    block=tl.program_id(0);bh=tl.program_id(1);b=bh//H;h=bh%H
    qi=block*BQ+tl.arange(0,BQ);ki=tl.arange(0,BK);d=tl.arange(0,D)
    q=tl.load(Q+b*Q0+qi[:,None]*Q1+h*Q2+d[None,:],qi[:,None]<N,0)
    k=tl.load(K+b*K0+ki[None,:]*K1+h*K2+d[:,None],ki[None,:]<T,0)
    logits=tl.dot(q,k)*(SM*1.4426950408889634)
    mask=tl.load(M+b*M0+ki,ki<T,0)
    logits=tl.where(mask[None,:]>0,logits,-float('inf'))
    p=tl.exp2(logits-tl.max(logits,1)[:,None]);denom=tl.sum(p,1)
    v=tl.load(V+b*V0+ki[:,None]*V1+h*V2+d[None,:],ki[:,None]<T,0)
    out=tl.dot(p.to(v.dtype),v)/denom[:,None]
    tl.store(O+((b*N+qi[:,None])*H+h)*D+d[None,:],out,qi[:,None]<N)

def cross_attention(q,k,v,mask):
    b,n,h,d=q.shape;t=k.shape[1]
    o=torch.empty_like(q)
    _cross[(triton.cdiv(n,64),b*h)](q,k,v,mask,o,n,t,h,d,*q.stride()[:3],*k.stride()[:3],*v.stride()[:3],mask.stride(0),d**-.5,64,triton.next_power_of_2(t),num_warps=4)
    return o

def forward(self,x,cond,mask=None):
    from torch.nn.functional import scaled_dot_product_attention
    b,n,c=x.shape
    q=self.q_norm(self.q_linear(x)).view(b,n,self.num_heads,self.head_dim)
    k,v=self._prepared_kv
    if (not LATE_ONLY or getattr(self.q_linear,"enabled",False)) and mask is not None and mask.ndim==2 and k.shape[1]<=128:
        x=cross_attention(q,k,v,mask)
    else:
        q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
        if mask is not None and mask.ndim==2:mask=((1-mask.to(q.dtype))*-10000.)[:,None,None]
        x=scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2)
    return self.proj_drop(self.proj(x.reshape(b,n,c)))

def install(model):
    import types
    for block in model.blocks:block.cross_attn.forward=types.MethodType(forward,block.cross_attn)
