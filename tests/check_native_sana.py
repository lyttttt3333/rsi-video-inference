import os
import sys
import torch
from accelerate import init_empty_weights
from model.transformer import SanaVideo2_5B
from model.vae import AutoencoderKLLTX2Video
import infer_native

reference_source = os.environ.get('SANA_REFERENCE_SOURCE')
if not reference_source:
    raise SystemExit('Set SANA_REFERENCE_SOURCE to an upstream Sana source checkout')
sys.path.insert(0, reference_source)
from diffusion.model.nets.sana_video2 import SanaVideo2_5B as ReferenceSana
from diffusers import AutoencoderKLLTX2Video as ReferenceVae

for ours, original in [(SanaVideo2_5B, ReferenceSana), (AutoencoderKLLTX2Video, ReferenceVae)]:
    with init_empty_weights():
        a, b = ours(), original()
    ak = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    bk = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert ak == bk, (ours.__name__, set(ak) ^ set(bk))
    print('PASS weight schema', ours.__name__)
