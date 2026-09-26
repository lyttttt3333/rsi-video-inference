"""Optional reuse of derived sampler predictions within a single request."""
import os
from .sampler import DPM_Solver

def install(model):
    model._prediction_target=os.environ.get('CACHE_PREDICTION','flow')
    original_model=DPM_Solver.model_fn
    original_noise=DPM_Solver.noise_prediction_fn
    def reuse(sampler,original,x,t):
        cache=getattr(sampler,'_derived_cache',None)
        if cache is None:
            cache={}
            sampler._derived_cache=cache
        step=cache.get('step',0)
        cache['step']=step+1
        if step in model._refresh_steps:
            # Keep precision selection aligned to the original sampler index.
            model._step_cache['step']=step
            cache['value']=original(sampler,x,t)
        return cache['value']
    def model_fn(sampler,x,t):
        if model._prediction_target=='x0':return reuse(sampler,original_model,x,t)
        return original_model(sampler,x,t)
    def noise_fn(sampler,x,t):
        if model._prediction_target=='noise':return reuse(sampler,original_noise,x,t)
        return original_noise(sampler,x,t)
    DPM_Solver.model_fn=model_fn
    DPM_Solver.noise_prediction_fn=noise_fn
