"""Small deterministic contract tests; no GPU or model loading."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('score', HERE / 'score.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


class VerificationTests(unittest.TestCase):
    def test_baseline_identity(self):
        self.assertEqual(score.case_score(10., 10., 0.)['reward'], 1.)

    def test_speed_quality_tradeoff(self):
        self.assertAlmostEqual(score.case_score(10., 5., .1)['reward'], 1.8)

    def test_temporal_quality_multiplier(self):
        scored = score.case_score(10., 5., .1, lpips_cutoff=None,
                                  excess_tlp_x100=.2)
        self.assertAlmostEqual(scored['reward'], 1.44)
        self.assertAlmostEqual(scored['temporal_quality'], .8)

    def test_lpips_above_one_clamped(self):
        self.assertEqual(score.case_score(10., 5., 1.5)['reward'], 0.)

    def test_lpips_hard_gate(self):
        self.assertGreater(score.case_score(10., 5., .249999, lpips_cutoff=.25)['reward'], 0.)
        gated = score.case_score(10., 5., .25, lpips_cutoff=.25)
        self.assertEqual(gated['reward'], 0.)
        self.assertFalse(gated['quality_gate_passed'])

    def test_failures_remain_in_denominator(self):
        self.assertEqual(score.aggregate([{'reward': 2}, {'reward': 0}]), 1.)

    def test_h3_split_quality_gate(self):
        passing = [
            {'invalid': 0, 'lpips': .20, 'excess_tlp_x100': .05},
            {'invalid': 0, 'lpips': .30, 'excess_tlp_x100': .15},
        ]
        gate = score.quality_gate(passing, mean_lpips_max=.25, worst_lpips_max=.30,
                                  mean_excess_tlp_x100_max=.10)
        self.assertTrue(gate['passed'])
        self.assertAlmostEqual(gate['mean_lpips'], .25)
        self.assertAlmostEqual(gate['worst_lpips'], .30)
        self.assertAlmostEqual(gate['mean_excess_tlp_x100'], .10)
        self.assertTrue(gate['temporal_passed'])
        self.assertFalse(score.quality_gate(
            [{'invalid': 0, 'lpips': .251}, {'invalid': 0, 'lpips': .251}], .25, .30)['passed'])
        self.assertFalse(score.quality_gate(
            [{'invalid': 0, 'lpips': .01}, {'invalid': 0, 'lpips': .301}], .25, .30)['passed'])

    def test_temporal_split_gate_uses_mean_not_worst(self):
        rows = [
            {'invalid': 0, 'lpips': .1, 'excess_tlp_x100': .0},
            {'invalid': 0, 'lpips': .1, 'excess_tlp_x100': .2},
        ]
        passing = score.quality_gate(rows, .25, .30, mean_excess_tlp_x100_max=.10)
        self.assertTrue(passing['passed'])
        self.assertAlmostEqual(passing['worst_excess_tlp_x100'], .2)
        rows[1]['excess_tlp_x100'] = .2001
        self.assertFalse(score.quality_gate(
            rows, .25, .30, mean_excess_tlp_x100_max=.10)['passed'])

    def test_h3_case_score_has_no_individual_cutoff(self):
        self.assertGreater(score.case_score(10., 5., .30, lpips_cutoff=None)['reward'], 0.)

    def test_invalid_metrics(self):
        for values in [(1,0,0), (1,1,float('nan')), (1,1,-1), (-1,1,0)]:
            with self.assertRaises(ValueError):
                score.case_score(*values)
        for excess in (-1, float('nan')):
            with self.assertRaises(ValueError):
                score.case_score(1, 1, 0, excess_tlp_x100=excess)

    def test_median_accepts_single_h3_measurement(self):
        self.assertEqual(score.median_seconds([7.]), 7.)
        self.assertEqual(score.median_seconds([1,100,2]), 2)
        with self.assertRaises(ValueError):
            score.median_seconds([])

    def test_public_cases_and_fixed_shapes(self):
        tasks = HERE.parents[1] / 'tasks'
        slug = 'sana-video2-hot-inference'
        public = json.loads(
            (tasks / slug / 'environment/validation/all-cases.json').read_text())
        self.assertEqual(len(public), 8)
        for r in public:
            self.assertEqual((r['width'], r['height'], r['frames'], r['steps']),
                             (960, 544, 121, 50))
        dockerfile = (tasks / slug / 'environment/Dockerfile').read_text()
        self.assertNotIn('COPY tests', dockerfile)

    def test_only_sana_harbor_task_is_published(self):
        tasks = HERE.parents[1] / 'tasks'
        self.assertTrue((tasks / 'sana-video2-hot-inference').is_dir())
        self.assertFalse((tasks / 'minimax-h3-hot-inference').exists())

    def test_missing_teacher_fails_closed_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cases.json').write_text(json.dumps([{'id':'example'}]))
            run = subprocess.run([sys.executable, str(HERE / 'verify.py'),
                '--model','sana_video2','--cases',str(root / 'cases.json'),
                '--reference',str(root / 'missing'), '--output',str(root / 'output')],
                capture_output=True, text=True)
            self.assertEqual(run.returncode,1)
            report = json.loads((root / 'output/reward.json').read_text())
            self.assertEqual(report['reward'],0)
            self.assertEqual(report['invalid'],1)
            self.assertIn('Teacher calibration',report['error'])

    def test_h3_rejects_more_than_one_timed_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = subprocess.run([sys.executable,str(HERE / 'verify.py'), '--model','minimax_h3',
                '--cases',str(root / 'missing'), '--reference',str(root / 'missing'),
                '--output',str(root / 'output'), '--repeats','3'], capture_output=True, text=True)
            self.assertEqual(run.returncode,1)
            report = json.loads((root / 'output/reward.json').read_text())
            self.assertIn('exactly 1 timed repeat', report['error'])


if __name__ == '__main__':
    unittest.main()
