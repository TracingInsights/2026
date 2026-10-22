"""Check that generated reports describe the run that produced the results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_v2 as reporting


class ReportingTests(unittest.TestCase):
    """Keep methodology text and public documentation references accurate."""

    def test_methodology_uses_configured_year_and_refit_interval(self):
        """Use manifest settings for both per-race and less frequent refits."""
        for year, every, cadence in [(2025, 1, 'after each race'), (2027, 5, 'every 5 races')]:
            with self.subTest(year=year, every=every):
                text = '\n'.join(reporting.methodology_lines(
                    dict(eval_year=year, tuning_races=12, refit_every=every)))
                self.assertIn(f'12 races before {year}', text)
                self.assertIn(f'completed {year} races', text)
                self.assertIn(f'model refits {cadence}', text)
                self.assertIn(f'{year} labels do not choose', text)
                self.assertNotIn('2026 labels', text)

    def test_generated_report_uses_manifest_and_shipped_readme(self):
        """Exercise report creation with a non-default archived evaluation."""
        manifest = dict(run_id='report-test', race_date='2025-10-19',
                        eval_year=2025, tuning_races=12, refit_every=5, prediction_modes=[])
        metrics = dict(winner_hit=.5, podium_hits=2., podium_set_hit=.25,
                       podium_brier=.1, ordered_podium_hit=.125)
        comparisons = {name: metrics for name in (
            'form_or_grid_baseline', 'uniform_core_ranker', 'selected_raw', 'selected_calibrated')}
        summary = {'pre': dict(selected=dict(name='test', temperature=1.),
                               eval_year=2025, eval_race_count=4, comparisons=comparisons,
                               paired_vs_baseline={})}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in [('run_manifest.json', manifest),
                               ('validation_summary.json', summary), ('pre_data_audit.json', [])]:
                (root / name).write_text(json.dumps(data), encoding='utf-8')
            with patch.object(reporting, 'ROOT', root), patch('builtins.print'):
                reporting.main()
            text = (root / 'RESULTS.txt').read_text(encoding='utf-8')
            self.assertIn('model refits every 5 races', text)
            self.assertIn('2025 labels do not choose', text)
            self.assertIn('See README.md for commands', text)
            self.assertNotIn('README_V2.txt', text)
            self.assertTrue((root / 'model_comparison.csv').exists())
            self.assertTrue((root / 'validation_comparison.png').exists())


if __name__ == '__main__':
    unittest.main()
