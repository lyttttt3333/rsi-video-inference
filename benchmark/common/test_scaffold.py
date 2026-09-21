"""Small standard-library tests; no model imports or GPU allocation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measure = load('measure')
campaign = load('campaign')


class ScaffoldTests(unittest.TestCase):
    def test_budget_sum_not_per_task(self):
        campaign.validate_allocation({'h3': 8 * 3600, 'sana': 4 * 3600})
        with self.assertRaises(ValueError):
            campaign.validate_allocation({'h3': 12 * 3600, 'sana': 12 * 3600})
        with self.assertRaises(ValueError):
            campaign.validate_allocation({'h3': 0, 'sana': 12 * 3600})

    def test_allocation_requires_explicit_positive_value(self):
        self.assertEqual(campaign.seconds('1.5'), 5400)
        with self.assertRaises(ValueError):
            campaign.seconds('0')

    def test_budget_configuration_preserves_verifier_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'task.toml'
            path.write_text('[agent]\n[environment.env]\nTASK_BUDGET_SECS = "UNALLOCATED"\n[verifier]\ntimeout_sec = 3600\n')
            campaign.configure_task(path, 100)
            campaign.configure_task(path, 200)
            text = path.read_text()
            self.assertEqual(text.count('campaign-agent-budget'), 1)
            self.assertIn('TASK_BUDGET_SECS = "200"', text)
            self.assertIn('[verifier]\ntimeout_sec = 3600', text)

    def test_timing_boundaries_warmup_and_export(self):
        events = []
        ticks = iter([10., 12., 20., 23., 30., 34.])
        class Runner:
            def generate(self, request):
                events.append(('generate', request['prompt'], request['seed']))
                return {'output': True}
        def sync(): events.append('sync')
        def clock():
            events.append('clock')
            return next(ticks)
        def validate(*args): events.append('validate')
        def save(*args): events.append('save')
        requests = [{'prompt': 'A', 'seed': 1}, {'prompt': 'B', 'seed': 2},
                    {'prompt': 'C', 'seed': 3}]
        report = measure.hot_measure(Runner(), requests, sync, 1, 1, validate, save, clock,
                                     warmup_scope='first')
        self.assertEqual(report['latency_mean_seconds'], 3.)
        self.assertEqual(len(report['samples']), 3)
        generated = [event for event in events if isinstance(event, tuple) and event[0] == 'generate']
        self.assertEqual(generated, [('generate', 'A', 1), ('generate', 'A', 1),
                                     ('generate', 'B', 2), ('generate', 'C', 3)])
        self.assertEqual(events[3:10], ['sync', 'clock', ('generate', 'A', 1),
                                        'sync', 'clock', 'validate', 'save'])

    def test_refuses_invalid_warmup_scope(self):
        with self.assertRaises(ValueError):
            measure.hot_measure(None, [{}], lambda: None, 0, 1, lambda *a: None,
                                warmup_scope='invalid')

    def test_nan_time_is_rejected(self):
        class Runner:
            def generate(self, request): return {}
        ticks = iter([0., float('nan')])
        with self.assertRaises(ValueError):
            measure.hot_measure(Runner(), [{}], lambda: None, 1, 1, lambda *a: None,
                                clock=lambda: next(ticks))

    def test_empty_submission_fails_before_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = subprocess.run([sys.executable, str(HERE / 'measure.py'),
                '--submission', str(root / 'missing'), '--weights', str(root / 'absent'),
                '--cases', str(root / 'cases.json'), '--model', 'minimax_h3',
                '--output', str(root / 'logs')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            result = json.loads((root / 'logs/reward.json').read_text())
            self.assertEqual(result['invalid'], 1)
            self.assertEqual(result['reward'], 0)
            self.assertTrue(result['scoring_pending'])


if __name__ == '__main__':
    unittest.main()
