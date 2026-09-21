"""Run inside either native image; no weights or GPU allocation required."""
import ast
import importlib.util
from pathlib import Path
import sys

assert importlib.util.find_spec('diffusers') is None
import infer

root = Path(infer.__file__).parent
assert (root / 'model').is_dir()
assert not (root / 'Sana').exists()
for source in (root / 'model').glob('*.py'):
    for node in ast.walk(ast.parse(source.read_text())):
        names = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                 else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
        assert not any(n.split('.')[0] in {'diffusers', 'sglang', 'sana', 'diffusion'} for n in names), source
assert not any(n == 'diffusers' or n.startswith('diffusers.') for n in sys.modules)
print('PASS standalone imports, local model sources, no Diffusers installed')
