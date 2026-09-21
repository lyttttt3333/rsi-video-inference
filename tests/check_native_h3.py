"""CPU parity checks against the pinned reference; not shipped in runtime."""
import sys
import torch
from model.scheduler import MiniMaxH3Scheduler
from model.transformer import MiniMaxH3Transformer3DModel
from model.video_vae import AutoencoderKLMiniMaxH3
from model.audio_vae import AutoencoderKLMiniMaxH3Audio
import infer_native
from diffusers import MiniMaxH3Scheduler as ReferenceScheduler

local, reference = MiniMaxH3Scheduler(12), ReferenceScheduler(12)
local.set_timesteps(50)
reference.set_timesteps(50)
torch.testing.assert_close(local.timesteps, reference.timesteps, rtol=0, atol=0)
x = torch.randn(3, 4)
velocity = torch.randn_like(x)
torch.testing.assert_close(local.step(velocity, local.timesteps[0], x)[0],
                           reference.step(velocity, reference.timesteps[0], x)[0], rtol=0, atol=0)
print('PASS scheduler parity')
from accelerate import init_empty_weights
from diffusers import MiniMaxH3Transformer3DModel as ReferenceTransformer
from diffusers import AutoencoderKLMiniMaxH3 as ReferenceVideo
from diffusers import AutoencoderKLMiniMaxH3Audio as ReferenceAudio
for ours, upstream in [(MiniMaxH3Transformer3DModel, ReferenceTransformer),
                       (AutoencoderKLMiniMaxH3, ReferenceVideo),
                       (AutoencoderKLMiniMaxH3Audio, ReferenceAudio)]:
    with init_empty_weights():
        a, b = ours(), upstream()
    ak = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    bk = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert ak == bk, (ours.__name__, set(ak) ^ set(bk))
    print('PASS weight schema', ours.__name__)

params = dict(num_attention_heads=2, attention_head_dim=16, hidden_size=32,
              num_layers=1, num_refiner_layers=1, ffn_dim=64, in_channels=4,
              audio_in_channels=4, patch_size=(1, 1, 1), text_dim=16,
              freq_dim=16, time_embed_hidden_dim=32, time_embed_dim=16, rope_freq_dim=2)
a, b = MiniMaxH3Transformer3DModel(**params).eval(), ReferenceTransformer(**params).eval()
b.load_state_dict(a.state_dict())
inputs = dict(hidden_states=torch.randn(1, 2, 4), audio_hidden_states=torch.randn(1, 2, 4),
              encoder_hidden_states=torch.randn(1, 2, 16), timestep=torch.tensor([0.5]),
              timestep_indices=torch.zeros(6, dtype=torch.long),
              token_tags=torch.tensor([0, 0, 2, 2, 1, 1]), position_ids=torch.randn(6, 3),
              video_indices=torch.tensor([0, 1]), audio_indices=torch.tensor([2, 3]),
              text_indices=torch.tensor([4, 5]), return_dict=False)
with torch.inference_mode():
    local_out, reference_out = a(**inputs), b(**inputs)
for x, y in zip(local_out, reference_out):
    torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
print('PASS transformer numerical parity')

from types import SimpleNamespace
from model.geometry import MiniMaxH3ModularPipeline
from model.before_denoise import MiniMaxH3NoKeyframeAnchorsStep, MiniMaxH3PrepareLayoutStep, MiniMaxH3PrepareLatentsStep, MiniMaxH3SetTimestepsStep
from model.denoise import MiniMaxH3DenoiseStep
from model.decoders import MiniMaxH3AfterDenoiseStep
c = MiniMaxH3ModularPipeline()
c._execution_device = torch.device('cpu')
c.transformer = a
c.vae = SimpleNamespace(spatial_compression_ratio=16, tokens_chunk_size=5,
                        config=SimpleNamespace(latent_channels=4, clip_length=17))
c.audio_vae = SimpleNamespace(config=SimpleNamespace(latent_channels=4))
c.scheduler, c.audio_scheduler = MiniMaxH3Scheduler(12), MiniMaxH3Scheduler(3)
s = SimpleNamespace(height=32, width=32, num_frames=124, num_inference_steps=2,
                    generator=torch.Generator().manual_seed(0), text_token_tags=torch.ones(2, dtype=torch.long),
                    prompt_embeds=torch.randn(1, 2, 16), attention_kwargs=None)
for cls in (MiniMaxH3NoKeyframeAnchorsStep, MiniMaxH3PrepareLayoutStep, MiniMaxH3PrepareLatentsStep, MiniMaxH3SetTimestepsStep):
    cls()(c, s)
s.denoiser_input_fields = {name: getattr(s, name) for name in ('position_ids', 'token_tags', 'video_indices', 'audio_indices', 'text_indices')}
MiniMaxH3DenoiseStep()(c, s)
MiniMaxH3AfterDenoiseStep()(c, s)
assert s.latents.shape == (1, 4, 37, 2, 2), s.latents.shape
print('PASS explicit denoising stages')
