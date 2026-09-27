"""Explicit MiniMax-H3 single-GPU inference using local model definitions."""
import argparse
import gc
import os
import sys
import subprocess
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

from model.text_encoder import Qwen3VLForConditionalGeneration
from model.transformer import MiniMaxH3Transformer3DModel
from model.video_vae import AutoencoderKLMiniMaxH3
from model.audio_vae import AutoencoderKLMiniMaxH3Audio
from model.scheduler import MiniMaxH3Scheduler
from model.runtime import Config, VideoProcessor
from model.geometry import MiniMaxH3ModularPipeline
from model.encoders import MiniMaxH3TextEncoderStep
from model.before_denoise import (MiniMaxH3NoKeyframeAnchorsStep, MiniMaxH3PrepareLayoutStep,
                                 MiniMaxH3PrepareLatentsStep, MiniMaxH3SetTimestepsStep)
from model.denoise import MiniMaxH3DenoiseStep
from model.decoders import MiniMaxH3AfterDenoiseStep, MiniMaxH3VideoDecodeStep, MiniMaxH3AudioDecodeStep


def encode_video(video, audio, sample_rate, output):
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        video_path, audio_path = Path(temporary) / 'video.mp4', Path(temporary) / 'audio.wav'
        imageio.mimwrite(video_path, (np.asarray(video) * 255).round().clip(0, 255).astype(np.uint8), fps=24, macro_block_size=1)
        samples = (audio.detach().float().cpu().numpy().T.clip(-1, 1) * 32767).astype('<i2')
        with wave.open(str(audio_path), 'wb') as file:
            file.setnchannels(samples.shape[1])
            file.setsampwidth(2)
            file.setframerate(sample_rate)
            file.writeframes(samples.tobytes())
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(video_path), '-i', str(audio_path),
                        '-c:v', 'copy', '-c:a', 'aac', str(output)], check=True)
    print(f'Video saved: {output}', flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path('weights'))
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--height', type=int, default=544)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--frames', '--num-frames', type=int, default=124)
    parser.add_argument('--steps', '--num-inference-steps', type=int, default=50)
    parser.add_argument('--workflow', choices=['t2va'], default='t2va')
    parser.add_argument('--offline', action='store_true', help='All weights are loaded locally regardless of this compatibility flag.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='outputs/minimax-h3-native.mp4')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('Exactly one CUDA GPU is required')
    components = MiniMaxH3ModularPipeline()
    components._execution_device = torch.device('cuda')
    components.config = Config(canvas_short_edge=768, canvas_max_pixels=1032192)
    components.tokenizer = Qwen2TokenizerFast.from_pretrained(args.weights / 'tokenizer', local_files_only=True)
    components.processor = Qwen3VLProcessor.from_pretrained(args.weights / 'processor', local_files_only=True)
    print('Loading text encoder', flush=True)
    components.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.weights / 'text_encoder', dtype=torch.bfloat16, local_files_only=True,
    ).eval().to('cuda')
    state = SimpleNamespace(prompt=args.prompt, height=args.height, width=args.width,
                            num_frames=args.frames, num_inference_steps=args.steps,
                            generator=torch.Generator().manual_seed(args.seed), output_type='np', attention_kwargs=None)
    MiniMaxH3TextEncoderStep()(components, state)
    del components.text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    print('Loading transformer', flush=True)
    components.transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        args.weights / 'transformer', torch_dtype=torch.bfloat16,
    ).to('cuda')
    components.scheduler = MiniMaxH3Scheduler.from_pretrained(args.weights / 'scheduler')
    components.audio_scheduler = MiniMaxH3Scheduler.from_pretrained(args.weights / 'audio_scheduler')
    for cls in (MiniMaxH3NoKeyframeAnchorsStep, MiniMaxH3PrepareLayoutStep, MiniMaxH3PrepareLatentsStep, MiniMaxH3SetTimestepsStep):
        cls()(components, state)
    state.denoiser_input_fields = {name: getattr(state, name) for name in
                                  ('position_ids', 'token_tags', 'video_indices', 'audio_indices', 'text_indices')}
    MiniMaxH3DenoiseStep()(components, state)
    MiniMaxH3AfterDenoiseStep()(components, state)
    del components.transformer
    gc.collect()
    torch.cuda.empty_cache()
    print('Decoding video', flush=True)
    components.video_processor = VideoProcessor()
    components.vae = AutoencoderKLMiniMaxH3.from_pretrained(args.weights / 'vae').to('cuda')
    MiniMaxH3VideoDecodeStep()(components, state)
    del components.vae
    gc.collect()
    torch.cuda.empty_cache()
    print('Decoding audio', flush=True)
    components.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(args.weights / 'audio_vae', torch_dtype=torch.bfloat16).to('cuda')
    MiniMaxH3AudioDecodeStep()(components, state)
    encode_video(state.videos[0], state.audio[0], state.sampling_rate, args.output)
    assert not any(name == 'diffusers' or name.startswith('diffusers.') for name in sys.modules)


if __name__ == '__main__':
    main()
