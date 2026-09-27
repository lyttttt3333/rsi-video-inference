"""CPU-only MP4 previews; scoring always uses the original decoded tensors."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import wave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run video encoding on a CPU compute node')
    import torch
    import imageio.v2 as imageio
    from verify import metric_frame
    torch.set_num_threads(2)
    manifest = json.loads((args.reference / 'manifest.json').read_text())
    for case_id, row in manifest['cases'].items():
        output = args.reference / f'{case_id}.mp4'
        if output.exists():
            raise FileExistsError(output)
        data = torch.load(args.reference / row['tensor'], weights_only=True, map_location='cpu', mmap=True)
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / 'video.mp4'
            with imageio.get_writer(video, fps=24, macro_block_size=1,
                                    ffmpeg_params=['-threads', '2']) as writer:
                for frame in range(data['video'].shape[2]):
                    rgb = metric_frame(data['video'], frame, manifest['model'])[0]
                    writer.append_data(((rgb + 1) * 127.5).round().byte().permute(1,2,0).numpy())
            if manifest['model'] == 'minimax_h3':
                audio = Path(tmp) / 'audio.wav'
                samples = (data['audio'][0].float().clamp(-1,1).T.numpy() * 32767).astype('<i2')
                with wave.open(str(audio), 'wb') as wav:
                    wav.setnchannels(2)
                    wav.setsampwidth(2)
                    wav.setframerate(32000)
                    wav.writeframes(samples.tobytes())
                subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-i',str(audio),
                                '-c:v','copy','-c:a','aac','-shortest',str(output)], check=True)
            else:
                import shutil
                shutil.copyfile(video, output)
        print(output, flush=True)


if __name__ == '__main__':
    main()
