"""Build a readable report and figures from completed v2 outputs."""
import collections
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('v2_outputs')
os.environ.setdefault('MPLCONFIGDIR', str((ROOT / 'matplotlib_cache').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    summary = json.loads((ROOT / 'validation_summary.json').read_text())
    manifest = json.loads((ROOT / 'run_manifest.json').read_text())
    old_path = ROOT / 'previous_model_comparison.json'
    previous = json.loads(old_path.read_text()) if old_path.exists() else {}
    rows = []
    lines = ['F1 Austin -- version 2 results', '',
             f"Run: {manifest['run_id']}; target race: {manifest['race_date']}.",
             f"Selection/calibration: last {manifest['tuning_races']} races before {manifest['eval_year']}. "
             f"Evaluation: completed {manifest['eval_year']} races.",
             'All forecasts train strictly on earlier races; model refits every three races.',
             '2026 labels do not choose feature variants, recency settings, or temperature.', '']
    labels = {'form_or_grid_baseline': 'Form/grid baseline', 'uniform_core_ranker': 'Uniform-history ranker',
              'selected_raw': 'Selected model, raw', 'selected_calibrated': 'Selected model, calibrated'}
    for mode, report in summary.items():
        chosen = report['selected']
        lines += [mode.upper(), f"  Selected: {chosen['name']}; temperature {chosen['temperature']:.3f}.",
                  f"  Evaluation: {report['eval_race_count']} races."]
        for name, metrics in report['comparisons'].items():
            rows.append(dict(mode=mode, model=labels[name], **metrics))
            lines.append(f"  {labels[name]:31} winner {metrics['winner_hit']:.1%}; "
                         f"podium {metrics['podium_hits']:.2f}/3; all three {metrics['podium_set_hit']:.1%}; "
                         f"podium Brier {metrics['podium_brier']:.4f}; ordered podium {metrics['ordered_podium_hit']:.1%}")
        if (mode in previous and previous[mode].get('eval_year') == report['eval_year']
                and previous[mode]['evaluation_races'] == report['eval_race_count']):
            metrics = previous[mode]
            rows.append(dict(mode=mode, model='Previous results-only model', **metrics))
            lines.append(f"  Previous results-only model     winner {metrics['winner_hit']:.1%}; "
                         f"podium {metrics['podium_hits']:.2f}/3; all three {metrics['podium_set_hit']:.1%}; "
                         f"podium Brier {metrics['podium_brier']:.4f}")
        for col, comparison in report['paired_vs_baseline'].items():
            lo, hi = comparison['ci95']
            lines.append(f"  Selected minus baseline, {col}: {comparison['mean']:+.4f}, race-bootstrap 95% [{lo:+.4f}, {hi:+.4f}].")
        audit = json.loads((ROOT / f'{mode}_data_audit.json').read_text())
        forecast_ok = sum(a['weather']['status'] == 'ok' for a in audit)
        practice_ok = sum(any(s.get('drivers', 0) >= 3 for s in a['sessions']) for a in audit)
        lines += [f'  Archived forecast coverage: {forecast_ok}/{len(audit)} weekends from 2024 onward.',
                  f'  Usable long-run/sprint summaries: {practice_ok}/{len(audit)} weekends.', '']
    pd.DataFrame(rows).to_csv(ROOT / 'model_comparison.csv', index=False)
    prediction_paths = [ROOT / f'{mode}_prediction_metadata.json' for mode in manifest['prediction_modes']]
    for path in prediction_paths:
        meta = json.loads(path.read_text())
        if meta.get('run_id') != manifest['run_id'] or meta['race_date'] != manifest['race_date']:
            raise ValueError('Prediction metadata does not match the current run manifest')
        mode = meta['mode']
        pred = pd.read_csv(ROOT / f'{mode}_prediction.csv')
        ordered = json.loads((ROOT / f'{mode}_ordered_podiums.json').read_text())
        lines += [f"CURRENT AUSTIN PREDICTION ({mode})", f"Training data through {meta['last_training_date']}.",
                  f"Weekend inputs used by selected model: {meta['selected']['extended']}.",
                  f"Calibration uses {meta['calibration_races']} earlier races, through {meta['calibration_last_date']}."]
        for _, driver in pred.head(7).iterrows():
            lines.append(f"  {driver.code:4} win {driver.p_win:.1%}; podium {driver.p_podium:.1%}")
        lines.append('Most likely ordered podiums (joint probabilities):')
        for podium in ordered[:5]:
            lines.append(f"  {podium['first']} / {podium['second']} / {podium['third']}: {podium['probability']:.2%}")
        weather = meta['data_audit'].get('weather', {})
        lines += [f"Austin forecast status: {weather.get('status')}; initialization {weather.get('run')}.",
                  'An ordered podium is one of thousands of possible combinations; its probability',
                  'is much lower than any individual driver podium probability.', '']
        fig, ax = plt.subplots(figsize=(9, 6))
        top = pred.head(10).iloc[::-1]
        ax.barh(top.code, top.p_podium*100, color='#dc3333', label='Podium')
        ax.barh(top.code, top.p_win*100, color='#202b3c', label='Win')
        ax.set_xlabel('Model probability (%)')
        stage = {'pre': 'before practice', 'practice': 'before qualifying', 'post': 'before race'}[mode]
        ax.set_title(f'Austin {meta.get("race_date", "2026")[:4]}: {stage}\nCalibrated race ranking model')
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(ROOT / f'{mode}_prediction.png', dpi=150)
        plt.close(fig)
    lines += ['INTERPRETATION AND LIMITATIONS',
              'A fitted calibrator does not guarantee that every percentage is accurate.',
              'Use the held-out Brier/reliability tables to judge probability quality.',
              'Brier score: lower is better. Podium hits: higher is better.',
              'An interval containing zero does not establish a clear difference from baseline.',
              'The held-out season is a small sample; inspect the per-race reports for counts.',
              'Source availability uses conservative eight-hour forecast and three-hour session',
              'buffers. These are assumptions, not verified historical publication timestamps.',
              'Practice pace cannot fully separate fuel loads, tyres, traffic, or car programs.',
              'Unavailable session inputs remain missing; no post-cutoff laps are used.',
              'Historical post-stage grids use qualifying order, excluding later grid changes.',
              'The joint ranking model does not model shared accidents or strategy directly.',
              'The revised model is experimental; a prospective unseen-weekend test is still needed.',
              'See README_V2.txt for commands and detailed methodology.']
    (ROOT / 'RESULTS.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    comparisons = pd.DataFrame(rows)
    modes = list(summary)
    fig, axes = plt.subplots(1, len(modes), figsize=(4.8*len(modes), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, modes):
        selected = comparisons[(comparisons['mode'] == mode) & comparisons.model.isin(
            ['Form/grid baseline', 'Uniform-history ranker', 'Selected model, calibrated'])]
        if selected.empty:
            continue
        ax.bar(['Baseline', 'Uniform', 'Selected'], selected.podium_hits,
               color=['#9098a6', '#497ea8', '#dc3333'])
        ax.set_title({'pre': 'Before practice', 'practice': 'Before qualifying', 'post': 'Before race'}[mode])
        ax.set_ylim(0, 3)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Correct podium drivers per race (out of 3)')
    fig.suptitle(f"Held-out {manifest['eval_year']} races: correct podium drivers")
    fig.tight_layout()
    fig.savefig(ROOT / 'validation_comparison.png', dpi=150)
    plt.close(fig)
    print('Saved v2_outputs/RESULTS.txt and figures.')


if __name__ == '__main__':
    main()
