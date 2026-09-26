"""Replay the fixed sampler graph with freshly copied request inputs."""
import torch

def sample_graph(runner,sampler,noise,condition,uncondition,mask,steps):
    model=runner.model
    actual_condition=model._prepared_condition
    actual_kv=[b.cross_attn._prepared_kv for b in model.blocks]
    def reset():
        if hasattr(sampler,'_derived_cache'):sampler._derived_cache.clear()
        model._step_cache.clear()
        if hasattr(model,'_fp8_step'):model._fp8_step=0
        for m in getattr(model,'_fp8_modules',[]):m.enabled=False
    state=getattr(runner,'_sampler_graph',None)
    if state is None:
        static_noise=noise.clone()
        static_condition=tuple(v.clone() if v is not None else None for v in actual_condition)
        static_kv=[tuple(v.clone() for v in pair) for pair in actual_kv]
        timesteps=sampler.get_time_steps(skip_type='time_uniform_flow',t_T=1.,t_0=.001,N=steps,device=noise.device,shift=12.)
        sampler.get_time_steps=lambda **kw:timesteps
        model._prepared_condition=static_condition
        for b,pair in zip(model.blocks,static_kv):b.cross_attn._prepared_kv=pair
        def run():
            return sampler.sample(static_noise,steps=steps,order=2,skip_type='time_uniform_flow',method='multistep',flow_shift=12.)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):reset();out=run()
        torch.cuda.current_stream().wait_stream(stream)
        reset()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=run()
        state=dict(noise=static_noise,condition=static_condition,kv=static_kv,text=(condition,uncondition,mask),sampler=sampler,graph=graph,out=out)
        runner._sampler_graph=state
        reset()
        model._prepared_condition=actual_condition
        for b,pair in zip(model.blocks,actual_kv):b.cross_attn._prepared_kv=pair
    state['noise'].copy_(noise)
    for dst,src in zip(state['condition'],actual_condition):
        if dst is not None:dst.copy_(src)
    for dst,src in zip(state['kv'],actual_kv):
        for d,s in zip(dst,src):d.copy_(s)
    for dst,src in zip(state['text'],(condition,uncondition,mask)):
        if dst.data_ptr()!=src.data_ptr():dst.copy_(src)
    state['graph'].replay()
    # All static request inputs are overwritten before replay, then cleared.
    state['noise'].zero_()
    for v in state['condition']:
        if v is not None:v.zero_()
    for pair in state['kv']:
        for v in pair:v.zero_()
    for v in state['text']:v.zero_()
    return state['out']
