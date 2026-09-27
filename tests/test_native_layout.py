"""Lightweight source/layout checks; runtime checks run on compute nodes."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeLayoutTests(unittest.TestCase):
    def test_local_architectures_without_external_pipeline_imports(self):
        for package in ('minimax_h3', 'sana_video2'):
            package_root = ROOT / 'inference' / package
            sources = list((package_root / 'model').glob('*.py'))
            self.assertTrue(sources)
            for source in sources + [package_root / 'infer_native.py']:
                for node in ast.walk(ast.parse(source.read_text())):
                    imports = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                               else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                    self.assertFalse(any(n.split('.')[0] in {'diffusers', 'sglang', 'sana', 'diffusion'}
                                         for n in imports), str(source))

    def test_default_launchers_select_native_images(self):
        for package in ('minimax_h3', 'sana_video2'):
            launcher = (ROOT / 'inference' / package / 'launch_shell.sh').read_text()
            self.assertIn('IMAGE', launcher)
            self.assertIn('--gpus=1', launcher)
            self.assertIn('--no-container-mount-home', launcher)

    def test_docker_context_is_local_only(self):
        for package in ('minimax_h3', 'sana_video2'):
            package_root = ROOT / 'inference' / package
            docker = (package_root / 'Dockerfile').read_text()
            self.assertIn('COPY model ./model', docker)
            self.assertNotIn('git clone', docker)
            self.assertNotIn('COPY . ', docker)
            requirements = (package_root / 'requirements.txt').read_text()
            self.assertNotIn('diffusers', requirements.lower())
            self.assertNotIn('sglang', requirements.lower())


if __name__ == '__main__':
    unittest.main()
