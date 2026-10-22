"""Compare the previous results-only model on the same held-out 2026 races."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cota_predictor_improved as old

ROOT = Path('v2_outputs')


def main():
    """Save walk-forward baseline predictions and metrics using V2 holdout folds."""
    old.XGB_REG['n_jobs'] = 2
    old.XGB_CLF['n_jobs'] = 2
    report = {}
    for mode in ['pre', 'post']:
        frame = pd.read_csv(ROOT / f'{mode}_features.csv')
        validation = json.loads((ROOT / f'{mode}_validation.json').read_text())
        ids = sorted(frame.loc[frame.position.notna() & frame.season.eq(validation['eval_year']), 'race_idx'].unique())
        parts = []
        step = validation['refit_every_races']
        for start in range(0, len(ids), step):
            block = ids[start:start+step]
            train = frame[frame.position.notna() & (frame.race_idx < min(block))]
            test = frame[frame.race_idx.isin(block)]
            parts.append(old.fit_predict(train, test, old.get_features(mode, False, False)))
            print(f'Previous {mode} model: {min(start+step, len(ids))}/{len(ids)} races', flush=True)
        predictions = pd.concat(parts, ignore_index=True)
        predictions.to_csv(ROOT / f'{mode}_previous_model_holdout.csv', index=False)
        rows = []
        for idx, race in predictions.groupby('race_idx'):
            actual = set(race.loc[race.position <= 3, 'driverId'])
            selected = set(race.nlargest(3, 'p_podium').driverId)
            winner = race.nlargest(1, 'p_win').position.iloc[0] == 1
            hits = len(actual & selected)
            rows.append(dict(race_idx=int(idx), winner_hit=float(winner), podium_hits=hits,
                             podium_set_hit=float(hits == 3),
                             podium_brier=float(np.mean((race.p_podium-race.position.le(3).astype(float))**2)),
                             win_brier=float(np.mean((race.p_win-race.position.eq(1).astype(float))**2))))
        metrics = pd.DataFrame(rows)
        metrics.to_csv(ROOT / f'{mode}_previous_model_races.csv', index=False)
        report[mode] = metrics.drop(columns='race_idx').mean().to_dict()
        report[mode]['evaluation_races'] = len(rows)
        report[mode]['eval_year'] = validation['eval_year']
        print(mode, report[mode], flush=True)
    (ROOT / 'previous_model_comparison.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
