"""Regression tests for probability constraints and historical feature safety."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import cota_predictor_improved as model


def history():
    """Build four synthetic races with enough drivers for circuit statistics."""
    rows = []
    for rnd in range(1, 5):
        for d in range(12):
            rows.append(dict(season=2026, rnd=rnd, race_name='Test', circuitId='americas',
                             driverId=f'd{d}', code=f'D{d}', constructorId=f't{d // 2}',
                             grid=d + 1, position=(d + rnd) % 12 + 1, points=float(12-d),
                             dnf=float(d == 11), dnf_inc=0., dnf_mech=float(d == 11),
                             quali_pos=d+1, q1=90.+d, q2=89.+d, q3=88.+d))
    return pd.DataFrame(rows)


class PredictorTests(unittest.TestCase):
    """Check forecast arithmetic, chronology, and manual starting grids."""

    def test_probability_totals_and_nested_events(self):
        """Enforce event totals and nesting for several field sizes."""
        rng = np.random.default_rng(42)
        for n in (3, 20, 22):
            for _ in range(20):
                df = pd.DataFrame(dict(race_idx=0, p_win=rng.random(n),
                                       p_podium=rng.random(n), p_top10=rng.random(n)))
                df[['p_win', 'p_podium', 'p_top10']] = df[['p_win', 'p_podium', 'p_top10']].astype('float32')
                out = model.reconcile_probabilities(df)
                for col, total in [('p_win', 1), ('p_podium', 3), ('p_top10', min(10, n))]:
                    self.assertAlmostEqual(out[col].sum(), total, places=9)
                    self.assertTrue(out[col].between(0, 1).all())
                self.assertTrue((out.p_win <= out.p_podium + 1e-12).all())
                self.assertTrue((out.p_podium <= out.p_top10 + 1e-12).all())

    def test_future_results_do_not_change_prior_features(self):
        """Keep current/prior inputs stable when current/future outcomes change."""
        df = history()
        wx = pd.DataFrame([
            dict(season=2026, rnd=rnd, air_temp=25., humidity=60., wind=10.,
                 cloud=20., precip_mm=1., rain_prob=90., elevation=150., wet=1.)
            for rnd in range(1, 5)
        ])
        before = model.add_features(df, wx=wx)
        df.loc[df.rnd >= 3, ['position', 'points', 'dnf']] = [12, 0, 1]
        df.loc[(df.rnd >= 3) & (df.driverId == 'd0'), 'position'] = 1
        after = model.add_features(df, wx=wx)
        cols = model.get_features('post', False, True)
        env_cols = ['circuit_pole_win', 'circuit_dnf', 'circuit_grid_corr',
                    'drv_wet_gain', 'drv_wet_x']
        self.assertTrue(before.loc[before.rnd == 3, env_cols].notna().all().all())
        pd.testing.assert_frame_equal(before.loc[before.rnd <= 3, cols],
                                      after.loc[after.rnd <= 3, cols])
        self.assertFalse(before.loc[before.rnd == 4, env_cols].equals(
            after.loc[after.rnd == 4, env_cols]))

    def test_pit_lane_start_uses_actual_field_size(self):
        """Place pit-lane starters behind the actual synthetic field."""
        df = history()
        df.loc[df.driverId == 'd0', 'grid'] = 0
        out = model.add_features(df)
        self.assertTrue((out.loc[out.driverId == 'd0', 'grid'] == 12).all())

    def test_reject_training_on_prediction_race(self):
        """Reject fitting on the race whose outcome is being predicted."""
        df = model.add_features(history())
        with self.assertRaisesRegex(ValueError, 'strictly precede'):
            model.fit_predict(df, df, model.get_features('pre', False))

    def test_unknown_and_duplicate_grid_codes_rejected(self):
        """Reject malformed grids for their specific validation reason."""
        race = dict(round='5', raceName='United States Grand Prix')
        with patch.object(model, 'cota_race', return_value=race), patch.object(model, 'jget',
                return_value={'MRData': {'RaceTable': {'Races': []}}}):
            cases = [('D0,D0', 'Grid codes must be nonempty and unique'),
                     ('TYPO', 'Unknown grid driver codes: TYPO'),
                     ('D0,', 'Grid codes must be nonempty and unique'),
                     ('D0', 'Without qualifying results, --grid must specify every driver in order')]
            for codes, message in cases:
                with self.subTest(codes=codes), self.assertRaisesRegex(ValueError, message):
                    model.build_upcoming(history(), 2026, 'post', codes)

    def test_complete_manual_grid_without_qualifying_accepted(self):
        """Accept a full manual grid when qualifying results are unavailable."""
        race = dict(round='5', raceName='United States Grand Prix')
        codes = [f'D{d}' for d in reversed(range(12))]
        with patch.object(model, 'cota_race', return_value=race), patch.object(model, 'jget',
                return_value={'MRData': {'RaceTable': {'Races': []}}}):
            upcoming, _, _, mode = model.build_upcoming(history(), 2026, 'post', ','.join(codes))
        self.assertEqual(mode, 'post')
        self.assertEqual(upcoming.sort_values('grid').code.tolist(), codes)


if __name__ == '__main__':
    unittest.main()
