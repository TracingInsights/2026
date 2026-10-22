import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import cota_predictor_improved as model


def history():
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
    def test_probability_totals_and_nested_events(self):
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
        df = history()
        before = model.add_features(df)
        df.loc[df.rnd >= 3, ['position', 'points', 'dnf']] = [12, 0, 1]
        after = model.add_features(df)
        cols = model.get_features('post', False, False)
        pd.testing.assert_frame_equal(before.loc[before.rnd <= 3, cols],
                                      after.loc[after.rnd <= 3, cols])

    def test_pit_lane_start_uses_actual_field_size(self):
        df = history()
        df.loc[df.driverId == 'd0', 'grid'] = 0
        out = model.add_features(df)
        self.assertTrue((out.loc[out.driverId == 'd0', 'grid'] == 12).all())

    def test_reject_training_on_prediction_race(self):
        df = model.add_features(history())
        with self.assertRaisesRegex(ValueError, 'strictly precede'):
            model.fit_predict(df, df, model.get_features('pre', False))

    def test_unknown_and_duplicate_grid_codes_rejected(self):
        race = dict(round='5', raceName='United States Grand Prix')
        with patch.object(model, 'cota_race', return_value=race), patch.object(model, 'jget',
                return_value={'MRData': {'RaceTable': {'Races': []}}}):
            for codes in ('D0,D0', 'TYPO', 'D0,'):
                with self.assertRaises(ValueError):
                    model.build_upcoming(history(), 2026, 'post', codes)


if __name__ == '__main__':
    unittest.main()
