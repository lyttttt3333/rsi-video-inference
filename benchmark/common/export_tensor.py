"""Export one decoded benchmark tensor as an MP4 preview on a CPU Slurm node."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tensor', type=Path, required=True, action='append')
    parser.add_argument('--model', choices=['sana_video2', 'minimax_h3'], required=True)
    parser.add_argument('--output', type=Path, required=True, action='append')
    parser.add_argument('--fps', type=int, default=24)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run video encoding on a CPU Slurm allocation')
    if len(args.tensor) != len(args.output):
        raise ValueError('Provide one --output for every --tensor')

    import imageio.v2 as imageio
    import torch
    from verify import metric_frame

    torch.set_num_threads(2)
    for tensor_path, output_path in zip(args.tensor, args.output):
        if output_path.exists():
            raise FileExistsError(output_path)
        data = torch.load(tensor_path, weights_only=True, map_location='cpu', mmap=True)
        video = data.get('video') if isinstance(data, dict) else None
        if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[:2] != (1, 3):
            raise ValueError('Expected decoded RGB video tensor [1, 3, T, H, W]')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temporary:
            video_path = Path(temporary) / 'video.mp4'
            with imageio.get_writer(video_path, fps=args.fps, macro_block_size=1,
                                    ffmpeg_params=['-threads', '2']) as writer:
                for frame in range(video.shape[2]):
                    rgb = metric_frame(video, frame, args.model)[0]
                    pixels = ((rgb + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
                    writer.append_data(pixels)
            if args.model == 'minimax_h3':
                audio = data.get('audio')
                if not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[:2] != (1, 2):
                    raise ValueError('Expected decoded stereo audio tensor [1, 2, samples]')
                audio_path = Path(temporary) / 'audio.wav'
                samples = (audio[0].float().clamp(-1, 1).T.numpy() * 32767).astype('<i2')
                with wave.open(str(audio_path), 'wb') as wav:
                    wav.setnchannels(2)
                    wav.setsampwidth(2)
                    wav.setframerate(32000)
                    wav.writeframes(samples.tobytes())
                subprocess.run(
                    ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(video_path), '-i', str(audio_path),
                     '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(output_path)],
                    check=True,
                )
            else:
                import shutil
                shutil.copyfile(video_path, output_path)
        print(output_path.resolve(), flush=True)


if __name__ == '__main__':
    main()
