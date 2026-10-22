# Experimental podium predictor

An Austin / Circuit of the Americas prediction example proposed by @fortikday,
built using TracingInsights session lap data, Jolpica race history, and archived
Open-Meteo forecasts. The implementation and tests were developed with AI assistance.

This is a research example for community development. Its current evaluation
does **not** establish a reliable accuracy improvement over the previous model
or simple baselines. It is not an official TracingInsights prediction service.

## What it does

- Fits race-grouped XGBoost rankings using strictly earlier races.
- Compares uniform history, 24/48-race half-lives, and a 72-race training window.
- Fits a temperature calibrator to chronological out-of-sample predictions.
- Produces consistent win, second-place, third-place and podium probabilities
  from a Plackett-Luce distribution, plus likely ordered podium combinations.
- Tests practice long-run features adjusted for tyre compound, tyre age and
  session time, and archived forecasts available before each prediction cutoff.
- Evaluates before practice, before qualifying and before the race. Additional
  feature sets are selected using earlier validation, not automatically assumed
  to improve accuracy.

## Install and test

Use Python 3.12. From the repository root:

```sh
cd examples/podium_predictor
python -m venv .venv
```

Activate the environment (`source .venv/bin/activate` on Linux/macOS, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell), then:

```sh
python -m pip install -r requirements.txt
python -m unittest -v test_predictor.py test_predictor_v2.py
```

The pinned dependencies are the versions used for the initial Windows/Python
3.12 evaluation. Cross-platform CI and a broader dependency matrix remain work
for review. Unit tests use synthetic or mocked data and require no downloads.

## Run

Before the 2026 Austin race, evaluate all three historical stages and create
the forecast for the stage currently available:

```sh
python cota_predictor_v2.py --season 2026 --eval-year 2026 --mode all
```

To reproduce the original historical evaluation independently of whether
Austin has already finished:

```sh
python cota_predictor_v2.py --season 2026 --eval-year 2026 --mode all --evaluate-only --through-date 2026-10-21
```

The initial download can be substantial. Later runs reuse local data. To repeat
an evaluation with cached data and reuse validation only when code, historical
features and settings match:

```sh
python cota_predictor_v2.py --season 2026 --eval-year 2026 --mode all --evaluate-only --through-date 2026-10-21 --offline --reuse-validation
```

For a manual current forecast refresh before Austin finishes:

```sh
python cota_predictor_v2.py --reuse-validation
```

Use `--mode practice` one hour before qualifying or later; use `--mode post`
two hours before the race. `--grid VER,ANT,RUS,...` can supply a confirmed grid
in post mode. Otherwise qualifying order is used. No scheduled jobs are created.

All commands above run from this example directory. The example downloads its
own inputs; it does not alter the archive's extraction scripts or data folders.

## Initial evaluation

Snapshot: race history through 21 October 2026. Selection/calibration used the
last 18 races of 2025. Choices were then evaluated by walking forward over the
17 completed 2026 races, refitting every three races on earlier results.

| Selected model stage | Winner correct | All podium drivers, any order | Exact podium order | Mean correct podium drivers |
| --- | ---: | ---: | ---: | ---: |
| Before practice | 5/17 (29.4%) | 0/17 | 0/17 | 1.24 / 3 |
| Before qualifying | 5/17 (29.4%) | 0/17 | 0/17 | 1.24 / 3 |
| Before race | 12/17 (70.6%) | 3/17 (17.6%) | 1/17 (5.9%) | 1.88 / 3 |

Before the race, the qualifying-order baseline averaged 1.82 correct podium
drivers; the previous results-only model averaged 1.88. Before practice, the
previous model averaged 1.29. The new model's differences from the simple
baselines are not clear in the paired race-bootstrap intervals. Calibration
improved the new ranking model's probability scores relative to its raw scores,
but does not establish perfectly calibrated probabilities or a predictive edge.

See `benchmarks_2026.json` for compact numerical comparisons. These are an initial
development benchmark, not a blind prospective experiment. Re-running with
changed source data or dependency versions may produce different results.

## Outputs and development

Runtime files are written under `.f1_cache/` and `v2_outputs/`; both are ignored.
Outputs include per-race holdout predictions, calibration tables, model-selection
reports, source/cutoff audits, current-run metadata, and readable results/charts.
`compare_v1.py` optionally compares the prior model on the same evaluation races;
run `summarize_v2.py` afterward to include that comparison in the readable report.

`cota_predictor_improved.py` contains the shared data/feature code and the previous
model retained for comparisons. The supported entry point for this example is
`cota_predictor_v2.py`; the previous script's separate weather backtest uses
realized weather and should not be confused with the new forecast-based path.

Future work is tracked in [ROADMAP.md](ROADMAP.md). Small changes with synthetic
regression tests and chronological baseline comparisons are welcome.

## Limits and sources

Practice fuel loads and traffic are unobserved. Long-run pace and within-stint
slopes remain noisy proxies. Forecast initialization precedes its publication;
we allow eight hours for availability, and three hours from session start for
session data. These are conservative assumptions, not verified publication logs.
Lap-end timestamps are also checked against the prediction cutoff. Historical
post-stage grids use qualifying order to exclude later penalty announcements.
Historical entry-list and result-publication revisions are not reconstructed.
Race-order strengths do not explicitly model shared accidents or strategy.
Position intervals are nominal, not empirically guaranteed coverage intervals.

- [TracingInsights session data](https://github.com/TracingInsights/2026)
- [Jolpica race-history API](https://github.com/jolpica/jolpica-f1)
- [Open-Meteo archived forecast runs](https://open-meteo.com/en/docs/single-runs-api)
- [XGBoost ranking documentation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)

This proposed contribution follows the repository's Apache-2.0 license.
Formula 1 and team marks belong to their respective owners; no affiliation is implied.
