"""Materialize explicitly selected weights for a portable assets image.

Large copies and checksums MUST run on a Slurm compute node. This tool never
uploads assets or downloads new weights, and refuses existing destinations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

COMPONENTS = {
    'minimax_h3': {'modular_model_index.json', 'transformer', 'vae', 'audio_vae',
                   'scheduler', 'audio_scheduler', 'text_encoder', 'tokenizer', 'processor'},
    'sana_video2': {'gemma', 'vae', 'SANA_Video_2.0_5B_720p.pth'},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=COMPONENTS, required=True)
    p.add_argument('--manifest', type=Path, required=True, help='JSON object: component name -> absolute source path')
    p.add_argument('--output', type=Path, required=True, help='New directory for assets-image build context')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run large asset staging in a Slurm CPU allocation, not on a login node')
    sources = json.loads(args.manifest.read_text())
    if set(sources) != COMPONENTS[args.model]:
        raise SystemExit(f'Required component keys: {sorted(COMPONENTS[args.model])}')
    resolved = {}
    for name, source in sources.items():
        path = Path(source)
        if not path.is_absolute() or not path.exists():
            raise SystemExit(f'Expected an existing absolute source: {name}')
        resolved[name] = path.resolve()
        if resolved[name] in {Path('/'), Path.home()}:
            raise SystemExit('Refusing a filesystem root or home directory as a weight component')
        if resolved[name].is_dir() and args.output.resolve().is_relative_to(resolved[name]):
            raise SystemExit('Output must not be inside a source component')
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / 'weights'
    destination.mkdir()
    for name, source in resolved.items():
        target = destination / name
        if source.is_dir():
            shutil.copytree(source, target, symlinks=False)
        else:
            shutil.copy2(source, target)
    # Record actual bytes, rather than claiming a local path is a pinned source.
    with (args.output / 'weights.sha256').open('w') as manifest:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as file:
                    for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                manifest.write(f'{digest.hexdigest()}  {path.relative_to(args.output)}\n')
    (args.output / 'Dockerfile').write_text('''ARG NATIVE_IMAGE
FROM ${NATIVE_IMAGE}
USER root
ENTRYPOINT []
COPY weights/ /opt/weights/
COPY weights.sha256 /opt/weights.sha256
RUN cd /opt && sha256sum -c weights.sha256
WORKDIR /workspace
CMD ["/bin/bash"]
''')
    (args.output / '.dockerignore').write_text('source-provenance.json\n')
    (args.output / 'source-provenance.json').write_text(json.dumps({
        'model': args.model, 'sources': {name: str(path) for name, path in resolved.items()},
        'redistribution_license_review': 'REQUIRED before upload'}, indent=2) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
