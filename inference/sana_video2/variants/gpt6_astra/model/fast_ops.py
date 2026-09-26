"""Inference kernels; caches in this module contain only shape-specialized code."""
import torch
import triton
import triton.language as tl

@triton.jit
def _rms(X, W, Y, N:tl.constexpr, S:tl.constexpr, EPS:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    i=tl.arange(0,BLOCK)
    x=tl.load(X+row*S+i,i<N,0).to(tl.float32)
    w=tl.load(W+i,i<N,0).to(tl.float32)
    z=x*tl.rsqrt(tl.sum(x*x,0)/N+EPS)*w
    tl.store(Y+row*N+i,z,i<N)

def rms(x, weight, eps):
    out=torch.empty_like(x, memory_format=torch.contiguous_format)
    _rms[(x.numel()//x.shape[-1],)](x,weight,out,x.shape[-1],x.stride(-2),eps,triton.next_power_of_2(x.shape[-1]))
    return out

@triton.jit
def _rope(X,F,Y,N:tl.constexpr,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,DOUBLE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    pair=i%(D//2); head=i//(D//2)%H; token=i//(D//2*H)%T; batch=i//(D//2*H*T)
    base=batch*S0+token*S1+head*S2+2*pair*S3
    a=tl.load(X+base,i<N,0).to(tl.float64 if DOUBLE else tl.float32)
    b=tl.load(X+base+S3,i<N,0).to(tl.float64 if DOUBLE else tl.float32)
    c=tl.load(F+token*D+2*pair,i<N,0).to(tl.float64 if DOUBLE else tl.float32)
    s=tl.load(F+token*D+2*pair+1,i<N,0).to(tl.float64 if DOUBLE else tl.float32)
    tl.store(Y+2*i,a*c-b*s,i<N)
    tl.store(Y+2*i+1,a*s+b*c,i<N)

def rope_token(x, freqs, double=True):
    b,t,h,d=x.shape
    out=torch.empty((b,t,h,d),device=x.device,dtype=x.dtype)
    _rope[(triton.cdiv(x.numel()//2,256),)](x,torch.view_as_real(freqs),out,x.numel()//2,t,h,d,*x.stride(),double,256,enable_fp_fusion=False)
    return out

def rope_channel(x,freqs,double=True):
    return rope_token(x.permute(0,3,1,2),freqs,double).permute(0,2,3,1)

@triton.jit
def _depth_logits(X,W0,W1,W2,L,N:tl.constexpr,C:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0); i=tl.arange(0,BLOCK)
    x=tl.load(X+row*C+i,i<C,0).to(tl.float32)
    sq=(x*x).to(tl.bfloat16).to(tl.float32)
    mean=(tl.sum(sq,0)/C).to(tl.bfloat16).to(tl.float32)
    var=(mean+1.e-6).to(tl.bfloat16).to(tl.float32)
    inv=tl.rsqrt(var).to(tl.bfloat16).to(tl.float32)
    key=(x*inv).to(tl.bfloat16).to(tl.float32)
    w0=tl.load(W0+i,i<C,0).to(tl.float32)
    w1=tl.load(W1+i,i<C,0).to(tl.float32)
    w2=tl.load(W2+i,i<C,0).to(tl.float32)
    tl.store(L+row,tl.sum(key*w0,0))
    tl.store(L+N+row,tl.sum(key*w1,0))
    tl.store(L+2*N+row,tl.sum(key*w2,0))

def depth_logits(x, projections):
    x=x.contiguous()
    rows=x.numel()//x.shape[-1]
    out=torch.empty((3,rows),device=x.device,dtype=x.dtype)
    _depth_logits[(rows,)](x,*projections,out,rows,x.shape[-1],triton.next_power_of_2(x.shape[-1]),enable_fp_fusion=False)
    return out

@triton.jit
def _depth_sum(V,L,P,PL,Y, ROWS:tl.constexpr,C:tl.constexpr,COUNT:tl.constexpr,KIND:tl.constexpr,HAS_PARTIAL:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0); cols=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    # Small, statically unrolled softmax over residual-depth sources.
    maximum=tl.full((),float('-inf'),tl.float32)
    for s in tl.static_range(COUNT):
        z=tl.load(L+s*3*ROWS+KIND*ROWS+row).to(tl.float32)
        maximum=tl.maximum(maximum,z)
    if HAS_PARTIAL:
        pz=tl.load(PL+KIND*ROWS+row).to(tl.float32)
        maximum=tl.maximum(maximum,pz)
    total=tl.full((),0,tl.float32)
    for s in tl.static_range(COUNT):
        z=tl.load(L+s*3*ROWS+KIND*ROWS+row).to(tl.float32)
        total+=tl.exp(z-maximum)
    if HAS_PARTIAL: total+=tl.exp(pz-maximum)
    result=tl.full((BLOCK,),0,tl.float32)
    for s in tl.static_range(COUNT):
        z=tl.load(L+s*3*ROWS+KIND*ROWS+row).to(tl.float32)
        w=(tl.exp(z-maximum)/total).to(tl.bfloat16).to(tl.float32)
        v=tl.load(V+s*ROWS*C+row*C+cols,cols<C,0).to(tl.float32)
        result+=w*v
    if HAS_PARTIAL:
        w=(tl.exp(pz-maximum)/total).to(tl.bfloat16).to(tl.float32)
        v=tl.load(P+row*C+cols,cols<C,0).to(tl.float32)
        result+=w*v
    tl.store(Y+row*C+cols,result,cols<C)

def depth_sum(values,logits,count,partial,partial_logits,kind):
    if count==1 and partial is None: return values[0]
    out=torch.empty_like(values[0])
    rows=out.numel()//out.shape[-1]
    _depth_sum[(rows,triton.cdiv(out.shape[-1],512))](values,logits,partial if partial is not None else values,partial_logits if partial_logits is not None else logits,out,rows,out.shape[-1],count,kind,partial is not None,512,enable_fp_fusion=False)
    return out

@triton.jit
def _o_norm_gate(X,W,G,Y,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,EPS:tl.constexpr,BD:tl.constexpr,BT:tl.constexpr):
    bh=tl.program_id(0); block=tl.program_id(1)
    b=bh//H; h=bh%H
    d=tl.arange(0,BD); t=block*BT+tl.arange(0,BT)
    x=tl.load(X+bh*D*T+d[:,None]*T+t[None,:],(d[:,None]<D)&(t[None,:]<T),0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS)
    w=tl.load(W+d,d<D,0).to(tl.float32)
    z=(x*inv[None,:]*w[:,None]).to(tl.bfloat16).to(tl.float32)
    addr=b*T*H*D+t[None,:]*H*D+h*D+d[:,None]
    g=tl.load(G+addr,(d[:,None]<D)&(t[None,:]<T),0).to(tl.float32)
    gate=tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+addr,z*gate,(d[:,None]<D)&(t[None,:]<T))

def o_norm_gate(x,w,eps,g):
    b,h,d,t=x.shape
    y=torch.empty_like(g)
    _o_norm_gate[(b*h,triton.cdiv(t,32))](x,w,g,y,t,h,d,eps,triton.next_power_of_2(d),32,enable_fp_fusion=False)
    return y

@triton.jit
def _swiglu(G,U,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    g=tl.load(G+i,i<N,0).to(tl.float32)
    u=tl.load(U+i,i<N,0).to(tl.float32)
    a=(g*tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,a*u,i<N)

def swiglu(g,u):
    y=torch.empty_like(g)
    _swiglu[(triton.cdiv(g.numel(),1024),)](g,u,y,g.numel(),1024)
    return y

@triton.jit
def _layer_mod(X,SCALE,SHIFT,Y,N:tl.constexpr,C:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0); group=row//N
    i=tl.arange(0,BLOCK)
    x=tl.load(X+row*C+i,i<C,0).to(tl.float32)
    mean=tl.sum(x,0)/C
    centered=tl.where(i<C,x-mean,0)
    var=tl.sum(centered*centered,0)/C
    z=(centered*tl.rsqrt(var+EPS)).to(tl.bfloat16).to(tl.float32)
    scale=tl.load(SCALE+group*C+i,i<C,0).to(tl.float32)
    shift=tl.load(SHIFT+group*C+i,i<C,0).to(tl.float32)
    scale=(1+scale).to(tl.bfloat16).to(tl.float32)
    z=(z*scale).to(tl.bfloat16).to(tl.float32)+shift
    tl.store(Y+row*C+i,z,i<C)

def layer_mod(x,shift,scale,eps):
    out=torch.empty_like(x)
    c=x.shape[-1]
    groups=shift.numel()//c
    rows=x.numel()//c
    _layer_mod[(rows,)](x,scale.contiguous(),shift.contiguous(),out,rows//groups,c,eps,triton.next_power_of_2(c),enable_fp_fusion=False)
    return out

@triton.jit
def _norm_rope(Q,K,WQ,WK,F,OQ,OK,T:tl.constexpr,C:tl.constexpr,D:tl.constexpr,STRIDE:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0); token=row%T
    i=tl.arange(0,BLOCK)
    qe=tl.load(Q+row*STRIDE+2*i,2*i<C,0).to(tl.float32)
    qo=tl.load(Q+row*STRIDE+2*i+1,2*i+1<C,0).to(tl.float32)
    ke=tl.load(K+row*STRIDE+2*i,2*i<C,0).to(tl.float32)
    ko=tl.load(K+row*STRIDE+2*i+1,2*i+1<C,0).to(tl.float32)
    qi=tl.rsqrt((tl.sum(qe*qe,0)+tl.sum(qo*qo,0))/C+EPS)
    ki=tl.rsqrt((tl.sum(ke*ke,0)+tl.sum(ko*ko,0))/C+EPS)
    qwe=tl.load(WQ+2*i,2*i<C,0).to(tl.float32); qwo=tl.load(WQ+2*i+1,2*i+1<C,0).to(tl.float32)
    kwe=tl.load(WK+2*i,2*i<C,0).to(tl.float32); kwo=tl.load(WK+2*i+1,2*i+1<C,0).to(tl.float32)
    qe=(qe*qi*qwe).to(tl.bfloat16).to(tl.float64); qo=(qo*qi*qwo).to(tl.bfloat16).to(tl.float64)
    ke=(ke*ki*kwe).to(tl.bfloat16).to(tl.float64); ko=(ko*ki*kwo).to(tl.bfloat16).to(tl.float64)
    c=tl.load(F+token*D+2*(i%(D//2))).to(tl.float64)
    s=tl.load(F+token*D+2*(i%(D//2))+1).to(tl.float64)
    tl.store(OQ+row*C+2*i,qe*c-qo*s,2*i<C)
    tl.store(OQ+row*C+2*i+1,qe*s+qo*c,2*i+1<C)
    tl.store(OK+row*C+2*i,ke*c-ko*s,2*i<C)
    tl.store(OK+row*C+2*i+1,ke*s+ko*c,2*i+1<C)

def norm_rope(q,k,qnorm,knorm,freqs,heads):
    b,t,c=q.shape;d=c//heads
    oq=torch.empty((b,t,heads,d),device=q.device,dtype=q.dtype)
    ok=torch.empty_like(oq)
    _norm_rope[(b*t,)](q,k,qnorm.weight,knorm.weight,torch.view_as_real(freqs),oq,ok,t,c,d,q.stride(-2),qnorm.eps,triton.next_power_of_2(c//2),enable_fp_fusion=False)
    return oq,ok

@triton.jit
def _vae_rms(X,Y,C:tl.constexpr,P:tl.constexpr,CHANNEL_LAST:tl.constexpr,EPS:tl.constexpr,BC:tl.constexpr,BP:tl.constexpr):
    b=tl.program_id(1); block=tl.program_id(0)
    c=tl.arange(0,BC); p=block*BP+tl.arange(0,BP)
    if CHANNEL_LAST:
        addr=b*C*P+p[:,None]*C+c[None,:]
    else:
        addr=b*C*P+c[None,:]*P+p[:,None]
    valid=(c[None,:]<C)&(p[:,None]<P)
    x=tl.load(X+addr,valid,0).to(tl.float32)
    sq=(x*x).to(tl.bfloat16).to(tl.float32)
    mean=(tl.sum(sq,1)/C).to(tl.bfloat16).to(tl.float32)
    var=(mean+EPS).to(tl.bfloat16).to(tl.float32)
    rms=tl.sqrt(var).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+addr,x/rms[:,None],valid)

def vae_rms(x,eps):
    cl=x.stride(1)==1
    memory_format=torch.channels_last_3d if cl else torch.contiguous_format
    x=x.contiguous(memory_format=memory_format)
    y=torch.empty_like(x)
    b,c,t,h,w=x.shape;p=t*h*w
    _vae_rms[(triton.cdiv(p,16),b)](x,y,c,p,cl,eps,triton.next_power_of_2(c),16,enable_fp_fusion=False)
    return y

@triton.jit
def _swiglu_packed(X,Y,N:tl.constexpr,C:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//C; col=i%C
    g=tl.load(X+row*2*C+col,i<N,0).to(tl.float32)
    u=tl.load(X+row*2*C+C+col,i<N,0).to(tl.float32)
    a=(g*tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,a*u,i<N)

def swiglu_packed(x):
    shape=(*x.shape[:-1],x.shape[-1]//2)
    y=torch.empty(shape,device=x.device,dtype=x.dtype)
    _swiglu_packed[(triton.cdiv(y.numel(),1024),)](x,y,y.numel(),y.shape[-1],1024)
    return y
