"""SANA-Video 2.0 text-to-video using local PyTorch model definitions."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer

from model.transformer import SanaVideo2_5B
from model.text_encoder import Gemma2ForCausalLM
from model.vae import AutoencoderKLLTX2Video
from model.sampler import DPM_Solver, NoiseScheduleFlow, model_wrapper
from model.prompt import CHI_PROMPT, NEGATIVE_PROMPT


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path('weights'))
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/sana-video2-native.mp4'))
    parser.add_argument('--height', type=int, default=736)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--frames', type=int, default=193)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--cfg-scale', type=float, default=8.0)
    parser.add_argument('--flow-shift', type=float, default=12.0)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('Exactly one CUDA GPU is required')
    if args.height % 32 or args.width % 32 or (args.frames - 1) % 8:
        raise SystemExit('Height/width must be multiples of 32; frames must be 8n+1')
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    print('Encoding text with local Gemma2', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.weights / 'gemma', local_files_only=True)
    tokenizer.padding_side = 'right'
    encoder = Gemma2ForCausalLM.from_pretrained(args.weights / 'gemma', torch_dtype=dtype,
                                              local_files_only=True).get_decoder().eval().to('cuda')
    length = len(tokenizer.encode(CHI_PROMPT)) + 300 - 2
    tokens = tokenizer([CHI_PROMPT + args.prompt + ' motion score: 20.'], max_length=length,
                       padding='max_length', truncation=True, return_tensors='pt').to('cuda')
    select = [0] + list(range(-299, 0))
    condition = encoder(tokens.input_ids, tokens.attention_mask)[0][:, None, select]
    mask = tokens.attention_mask[:, select]
    negative = tokenizer([NEGATIVE_PROMPT], max_length=300, padding='max_length',
                         truncation=True, return_tensors='pt').to('cuda')
    uncondition = encoder(negative.input_ids, negative.attention_mask)[0][:, None]
    mask = torch.cat([negative.attention_mask, mask]) if args.cfg_scale > 1 else mask
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    print('Loading local SANA transformer', flush=True)
    with init_empty_weights(include_buffers=False):
        model = SanaVideo2_5B(input_size=args.height // 32)
    checkpoint = torch.load(args.weights / 'SANA_Video_2.0_5B_720p.pth', map_location='cpu', weights_only=True, mmap=True)
    state = checkpoint.get('state_dict', checkpoint)
    state.pop('pos_embed', None)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if set(missing) - {'pos_embed'} or unexpected:
        raise ValueError((missing, unexpected))
    model.eval().requires_grad_(False).to(device='cuda', dtype=dtype)
    for module in model.modules():
        module.fp32_attention = True
    del checkpoint, state
    noise = torch.randn(1, 128, (args.frames - 1) // 8 + 1, args.height // 32, args.width // 32,
                        generator=torch.Generator(device='cuda').manual_seed(args.seed), device='cuda')
    schedule = NoiseScheduleFlow(schedule='discrete_flow')
    wrapped = model_wrapper(model, schedule, model_type='flow', condition=condition,
                            unconditional_condition=uncondition, guidance_scale=args.cfg_scale,
                            guidance_type='classifier-free',
                            model_kwargs={'data_info': {'img_hw': torch.tensor([[args.height, args.width]], device='cuda')}, 'mask': mask})
    sampler = DPM_Solver(wrapped, schedule, algorithm_type='dpmsolver++')
    latents = sampler.sample(noise, steps=args.steps, order=2, skip_type='time_uniform_flow',
                             method='multistep', flow_shift=args.flow_shift)
    del model, wrapped, sampler
    gc.collect()
    torch.cuda.empty_cache()
    print('Decoding with local LTX VAE', flush=True)
    vae_path = args.weights / 'vae'
    config = json.loads((vae_path / 'config.json').read_text())
    overrides = {}
    if 'upsample_type' not in config and 'decoder_upsample_type' in config:
        overrides['upsample_type'] = tuple(reversed(config['decoder_upsample_type']))
    vae = AutoencoderKLLTX2Video.from_pretrained(vae_path, torch_dtype=dtype, **overrides).to('cuda')
    vae.enable_tiling(tile_sample_min_height=512, tile_sample_min_width=512,
                      tile_sample_stride_height=448, tile_sample_stride_width=448)
    vae.use_framewise_decoding = True
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
    latents = (latents * std / vae.config.scaling_factor + mean).to(dtype)
    video = vae.decode(latents, temb=None, return_dict=False)[0]
    frames = ((video[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.output, frames, fps=24, macro_block_size=1)
    print(f'Video saved: {args.output.absolute()}', flush=True)
    assert not any(name == 'diffusers' or name.startswith('diffusers.') for name in sys.modules)


if __name__ == '__main__':
    main()
