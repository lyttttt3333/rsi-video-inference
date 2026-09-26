import os
import torch
from . import attention
from .norm import RMSNorm
from .fast_ops import rms, rope_token, rope_channel

def install(model, vae=None):
    if os.environ.get('SDPA_BACKEND','') == 'cudnn':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(True)
    if os.environ.get('MERGE_MLP','0') == '1':
        install_merged_mlp(model)
    if os.environ.get('FP8','0') != '0':
        from .fp8 import install_fp8
        install_fp8(model,os.environ['FP8'])
        if 'FP8_START' in os.environ:
            install_fp8_window(model)
    if os.environ.get('FP8_SWIGLU','0') == '1':
        from .fp8_fusion import install as install_fp8_swiglu
        install_fp8_swiglu(model)
    if os.environ.get('CUDNN_BENCH','0') == '1':
        torch.backends.cudnn.benchmark=True
    if vae is not None and os.environ.get('VAE_NORM','0') == '1':
        from .vae import PerChannelRMSNorm
        from .fast_ops import vae_rms
        PerChannelRMSNorm.forward=lambda self,x,channel_dim=None: vae_rms(x,self.eps)
    if vae is not None and os.environ.get('VAE_FULL','0') == '1':
        vae.disable_tiling()
    if vae is not None and os.environ.get('VAE_CHANNELS_LAST','0') == '1':
        from .vae import LTX2VideoCausalConv3d
        vae.decoder.to(memory_format=torch.channels_last_3d)
        original_conv=LTX2VideoCausalConv3d.forward
        def channels_last_conv(self,x,causal=True):
            return original_conv(self,x.contiguous(memory_format=torch.channels_last_3d),causal)
        LTX2VideoCausalConv3d.forward=channels_last_conv
    if vae is not None and os.environ.get('VAE_PAD','0')=='1':
        from .vae_pad import install as install_vae_pad
        install_vae_pad()
    if vae is not None and os.environ.get('VAE_NORM_PAD','0')=='1':
        from .vae_norm_pad import install as install_vae_norm_pad
        install_vae_norm_pad(vae)
    if os.environ.get('CONDITION_CACHE','0') == '1':
        install_condition_cache(model)
    if os.environ.get('CROSS_FLASH','0') == '1':
        from .cross_flash import install as install_cross_flash
        install_cross_flash(model)
    double = os.environ.get('ROPE_DOUBLE','1') == '1'
    attention._apply_rope_channel_first = lambda x,f: rope_channel(x,f,double)
    attention._apply_rope_token_first = lambda x,f: rope_token(x,f,double)
    original = RMSNorm.forward
    def fast_norm(self,x):
        if self.norm_dim == -1 and x.stride(-1)==1:
            return rms(x,self.weight,self.eps)
        return original(self,x)
    RMSNorm.forward=fast_norm
    if os.environ.get('COMPILE_BLOCK','0') == '1':
        for b in model.blocks:
            b.forward_attn_sublayer = torch.compile(b.forward_attn_sublayer, dynamic=False, mode=os.environ.get("COMPILER_MODE","default"))
            b.forward_mlp_sublayer = torch.compile(b.forward_mlp_sublayer, dynamic=False, mode=os.environ.get("COMPILER_MODE","default"))
    if os.environ.get('FAST_RESIDUAL','0') == '1':
        import types
        model._forward_attn_res_inference=types.MethodType(fast_residual,model)
    if int(os.environ.get("GUIDANCE_CACHE","1"))>1:
        install_guidance_cache(model)
    depth_cache=int(os.environ.get("CACHE_DEPTH","0"))
    if depth_cache:
        model._deep_cache={}
        model._cache_depth=depth_cache
    if os.environ.get("CORE_GRAPH","0")=="1" and not depth_cache:
        install_core_graph(model)
    if int(os.environ.get("CACHE_INTERVAL","1"))>1 and not depth_cache:
        if os.environ.get("CACHE_TARGET", "hidden")=="flow":
            install_flow_cache(model)
        else:
            install_step_cache(model)

    from .prediction_cache import install as install_prediction_cache
    install_prediction_cache(model)

@torch.no_grad()
def fast_residual(self,x,y,t0,condition_mask,rope_linear,rope_softmax):
    from .fast_ops import depth_logits, depth_sum
    cache=getattr(self,'_deep_cache',None)
    if cache is not None:
        step=cache.get('step',0)
        cache['step']=step+1
        start=int(os.environ.get('CACHE_START','4'))
        stop=int(os.environ.get('CACHE_STOP','45'))
        interval=int(os.environ.get('CACHE_INTERVAL','3'))
        full=step<start or step>=stop or (step-start)%interval==0
        extrap=float(os.environ.get('CACHE_EXTRAP','0'))
    count=(self.depth+self.attn_res_block_size-1)//self.attn_res_block_size
    values=torch.empty((count+1,*x.shape),device=x.device,dtype=x.dtype)
    logits=torch.empty((count+1,3,x.numel()//x.shape[-1]),device=x.device,dtype=x.dtype)
    projections=[self.attn_res.attn_proj.weight,self.attn_res.mlp_proj.weight,self.attn_res.final_proj.weight]
    values[0]=x
    logits[0]=depth_logits(x,projections)
    active=1
    partial=None
    pl=None
    for start in range(0,self.depth,self.attn_res_block_size):
        for index in range(start,min(start+self.attn_res_block_size,self.depth)):
            hidden=depth_sum(values,logits,active,partial,pl,0)
            delta=self.blocks[index].forward_attn_sublayer(hidden,y,t0,mask=condition_mask,THW=(self.f,self.h,self.w),rotary_emb=self._rope_for_layer(index,rope_linear,rope_softmax))
            partial=delta if partial is None else partial+delta
            pl=depth_logits(partial,projections)
            hidden=depth_sum(values,logits,active,partial,pl,1)
            partial=partial+self.blocks[index].forward_mlp_sublayer(hidden,t0,THW=(self.f,self.h,self.w))
            pl=depth_logits(partial,projections)
            if cache is not None and index+1==self._cache_depth:
                anchor=depth_sum(values,logits,active,partial,pl,2)
                if not full:
                    res=cache['res']
                    if extrap and 'diff' in cache: res=res+cache['diff']*((step-cache['last'])*extrap)
                    return (anchor.float()+res).to(x.dtype)
        values[active]=partial
        logits[active]=pl
        active+=1
        partial=None
        pl=None
    out=depth_sum(values,logits,active,None,None,2)
    if cache is not None:
        res=out.float()-anchor.float()
        if 'res' in cache:cache['diff']=(res-cache['res'])/(step-cache['last'])
        cache['res']=res
        cache['last']=step
    return out

def install_condition_cache(model):
    import types
    from .blocks import MultiHeadCrossAttention
    original_prepare = model._prepare_condition
    def prepare(self,y,mask):
        if self._prepared_condition is not None:
            return self._prepared_condition
        return original_prepare(y,mask)
    def begin(self,y,mask):
        self._prepared_condition=original_prepare(y,mask)
        cond,cmask=self._prepared_condition
        limit=int(os.environ.get('TEXT_LIMIT','0'))
        if limit and cmask is not None and not bool(cmask[:,limit:].any()):
            cond=cond[:,:,:limit].contiguous()
            cmask=cmask[:,:limit].contiguous()
            self._prepared_condition=(cond,cmask)
        for block in self.blocks:
            a=block.cross_attn
            first_dim=1 if a.use_xformers else y.shape[0]
            kv=a.kv_linear(cond).view(first_dim,-1,2,a.d_model)
            k,v=kv.unbind(2)
            k=a.k_norm(k).view(first_dim,-1,a.num_heads,a.head_dim)
            v=v.view(first_dim,-1,a.num_heads,a.head_dim)
            a._prepared_kv=(k,v)
    def end(self):
        self._prepared_condition=None
        for block in self.blocks: block.cross_attn._prepared_kv=None
    model._prepared_condition=None
    model._prepare_condition=types.MethodType(prepare,model)
    model.begin_request=types.MethodType(begin,model)
    model.end_request=types.MethodType(end,model)
    for block in model.blocks: block.cross_attn._prepared_kv=None
    # Keep all backend-specific mask behavior identical to the baseline.
    import inspect
    source=inspect.getsource(MultiHeadCrossAttention.forward)
    import textwrap
    source=textwrap.dedent(source)
    old='''kv = self.kv_linear(cond).view(first_dim, -1, 2, C)
    k, v = kv.unbind(2)
    q = self.q_norm(q).view(first_dim, -1, self.num_heads, self.head_dim)
    k = self.k_norm(k).view(first_dim, -1, self.num_heads, self.head_dim)
    v = v.view(first_dim, -1, self.num_heads, self.head_dim)'''
    new='''q = self.q_norm(q).view(first_dim, -1, self.num_heads, self.head_dim)
    k, v = self._prepared_kv'''
    assert old in source
    namespace=dict(MultiHeadCrossAttention.forward.__globals__)
    exec(source.replace(old,new),namespace)
    MultiHeadCrossAttention.forward=namespace['forward']

def install_step_cache(model):
    original=model._forward_attn_res_inference
    interval=int(os.environ.get('CACHE_INTERVAL','1'))
    start=int(os.environ.get('CACHE_START','4'))
    stop=int(os.environ.get('CACHE_STOP','40'))
    extrap=float(os.environ.get('CACHE_EXTRAP','0'))
    fp32=os.environ.get('CACHE_FLOAT','0')=='1'
    time_extrap=os.environ.get('CACHE_TIME','0')=='1'
    output_mode=os.environ.get('CACHE_MODE','residual')=='output'
    def clock(step):
        if not time_extrap:return float(step)
        u=.999*(1-step/50.)
        return 12*u/(1+11*u)
    model._step_cache={}
    def forward(x,*args):
        cache=model._step_cache
        step=cache.get('step',0)
        cache['step']=step+1
        full=step<start or step>=stop or (step-start)%interval==0
        if full:
            out=original(x,*args)
            res=(out.float()-x.float()) if fp32 else out-x
            if output_mode:res=out.float() if fp32 else out
            if 'res' in cache:
                cache['diff']=(res-cache['res'])/(clock(step)-clock(cache['last']))
            cache['res']=res
            cache['last']=step
            return out
        res=cache['res']
        if extrap and 'diff' in cache: res=res+cache['diff']*((clock(step)-clock(cache['last']))*extrap)
        if output_mode:return res.to(x.dtype)
        return ((x.float()+res).to(x.dtype) if fp32 else x+res)
    model._forward_attn_res_inference=forward

def install_core_graph(model):
    original=model._forward_attn_res_inference
    state={}
    def run(x,y,t0,mask,rl,rs):
        inputs=(x,y,t0,mask,rl,rs)
        # Condition projections are request-local; copy them into graph-owned
        # buffers so a new prompt never reuses a previous prompt's values.
        kv=[b.cross_attn._prepared_kv for b in model.blocks] if hasattr(model,'begin_request') else []
        if not state:
            static=[v.clone() if v is not None else None for v in inputs]
            static_kv=[tuple(v.clone() for v in pair) for pair in kv]
            for b,p in zip(model.blocks,static_kv): b.cross_attn._prepared_kv=p
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2): out=original(*static)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): out=original(*static)
            state.update(inputs=static,kv=static_kv,graph=graph,out=out)
            for b,p in zip(model.blocks,kv):b.cross_attn._prepared_kv=p
        for dst,src in zip(state['inputs'],inputs):
            if dst is not None:dst.copy_(src)
        for dst,src in zip(state['kv'],kv):
            for d,s in zip(dst,src):d.copy_(s)
        state['graph'].replay()
        return state['out']
    model._forward_attn_res_inference=run

def install_flow_cache(model):
    original=model.forward
    interval=int(os.environ.get('CACHE_INTERVAL','3'))
    start=int(os.environ.get('CACHE_START','4'))
    stop=int(os.environ.get('CACHE_STOP','45'))
    extrap=float(os.environ.get('CACHE_EXTRAP','0'))
    time_extrap=os.environ.get('CACHE_TIME','0')=='1'
    manual=os.environ.get('CACHE_STEPS','')
    refresh=set(map(int,manual.split(','))) if manual else None
    model._step_cache={}
    model._refresh_steps=refresh
    def clock(step):
        if not time_extrap:return float(step)
        u=.999*(1-step/50.)
        return 12*u/(1+11*u)
    def run(*args,**kwargs):
        cache=model._step_cache
        step=cache.get('step',0);cache['step']=step+1
        refresh=model._refresh_steps
        full=(step in refresh) if refresh is not None else (step<start or step>=stop or (step-start)%interval==0)
        if full:
            out=original(*args,**kwargs)
            value=out.float()
            if 'value' in cache:cache['diff']=(value-cache['value'])/(clock(step)-clock(cache['last']))
            cache['value']=value
            cache['last']=step
            cache['dtype']=out.dtype
            return out
        value=cache['value']
        if extrap and 'diff' in cache:value=value+cache['diff']*((clock(step)-clock(cache['last']))*extrap)
        return value.to(cache['dtype'])
    model.forward=run

def install_guidance_cache(model):
    original=model.forward
    interval=int(os.environ.get('GUIDANCE_CACHE','2'))
    start=int(os.environ.get('GUIDANCE_START','4'))
    model._guidance_cache={}
    def run(x,t,y,mask=None,**kwargs):
        cache=model._guidance_cache
        step=cache.get('step',0);cache['step']=step+1
        full=step<start or step%interval==0 or getattr(model,'_step_cache',{}).get('step',0)==50
        if full:
            out=original(x,t,y,mask=mask,**kwargs)
            u,c=out.chunk(2)
            cache['delta']=c.float()-u.float()
            return out
        cond=model._prepared_condition
        kv=[b.cross_attn._prepared_kv for b in model.blocks]
        model._prepared_condition=(cond[0][1:],cond[1][1:] if cond[1] is not None else None)
        for b,p in zip(model.blocks,kv):b.cross_attn._prepared_kv=(p[0][1:],p[1][1:])
        try:
            c=original(x[1:],t[1:],y[1:],mask=mask[1:] if mask is not None else None,**kwargs)
        finally:
            model._prepared_condition=cond
            for b,p in zip(model.blocks,kv):b.cross_attn._prepared_kv=p
        u=(c.float()-cache['delta']).to(c.dtype)
        return torch.cat([u,c])
    model.forward=run

def merged_mlp_forward(self,x,**kwargs):
    from .fast_ops import swiglu_packed
    return self.down_proj(swiglu_packed(self.gate_up_proj(x)))

def install_merged_mlp(model):
    import types
    from torch import nn
    for b in model.blocks:
        m=b.mlp
        linear=nn.Linear(m.gate_proj.in_features,m.gate_proj.out_features*2,bias=m.gate_proj.bias is not None,device='meta',dtype=torch.bfloat16)
        linear.weight=nn.Parameter(torch.cat([m.gate_proj.weight,m.up_proj.weight]),requires_grad=False)
        if linear.bias is not None:linear.bias=nn.Parameter(torch.cat([m.gate_proj.bias,m.up_proj.bias]),requires_grad=False)
        m.gate_up_proj=linear
        m.forward=types.MethodType(merged_mlp_forward,m)


def install_fp8_window(model):
    from .fp8 import FP8Linear
    modules=[m for m in model.modules() if isinstance(m,FP8Linear)]
    model._fp8_modules=modules
    for m in modules:m.enabled=False
    start=int(os.environ.get('FP8_START','0'))
    stop=int(os.environ.get('FP8_STOP','49'))
    original=model.forward
    model._fp8_step=0
    def forward(*args,**kwargs):
        if os.environ.get('CACHE_TARGET','hidden')=='flow' and hasattr(model,'_step_cache'):
            step=model._step_cache.get('step',1)-1
        else:
            step=model._fp8_step
        model._fp8_step+=1
        enabled=getattr(model,'_fp8_start',start)<=step<getattr(model,'_fp8_stop',stop)
        allow=getattr(model,'_fp8_allow',None)
        for m in modules:m.enabled=enabled and (allow is None or id(m) in allow)
        return original(*args,**kwargs)
    model.forward=forward
