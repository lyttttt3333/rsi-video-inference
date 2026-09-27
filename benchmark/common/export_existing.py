"""Export existing first-repeat teachers only; no inference or calibration claim."""
import json
import os
from pathlib import Path
import subprocess
import sys

if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('CPU Slurm allocation required')
here = Path(__file__).resolve().parent
outputs = here.parent / 'outputs/harbor'
import torch
from measure import validate_output
torch.set_num_threads(2)
reports = []
for model, run in [('sana_video2', 'calibrate-all-N9NNdaXs'),
                   ('minimax_h3', 'calibrate-all-89U4qEgH')]:
    cases = json.loads((here / 'reference-specs' / f'{model}.json').read_text())
    target = outputs / 'teacher-videos' / model
    target.mkdir(parents=True, exist_ok=False)
    entries = {}
    for index, request in enumerate(cases):
        source = outputs / model / run / 'results/measurement' / f'output-0-{index}.pt'
        data = torch.load(source, map_location='cpu', mmap=True, weights_only=True)
        validate_output(data, request, model, torch)
        del data
        name = request['id'] + '.pt'
        os.link(source, target / name)
        entries[request['id']] = {'tensor': name}
    (target / 'manifest.json').write_text(json.dumps(dict(model=model, cases=entries,
        purpose='MP4 export only; not a calibrated scoring manifest'), indent=2))
    subprocess.run([sys.executable, str(here / 'export_teachers.py'), '--reference', str(target)], check=True)
    for request in cases:
        video = target / (request['id'] + '.mp4')
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
            '-show_entries','stream=codec_type,width,height,nb_read_frames,sample_rate,channels:format=duration',
            '-of','json',str(video)], text=True))
        stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        assert (stream['width'],stream['height'],int(stream['nb_read_frames'])) == (
            request['width'],request['height'],request['frames'])
        if model == 'minimax_h3':
            audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            assert audio['channels'] == 2 and int(audio['sample_rate']) == 32000
        reports.append(dict(model=model, file=str(video.relative_to(outputs)), probe=probe))
(outputs / 'teacher-videos/export-report.json').write_text(json.dumps(reports, indent=2))
print('All 16 existing teachers exported and frame/audio properties verified.', flush=True)
