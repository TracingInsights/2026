# Podium predictor development roadmap

The first objective is a reproducible, honestly evaluated example. Adding more
features is not itself evidence of better forecasts.

## Initial contribution

- [x] Grouped race ranking and coherent ordered-podium probabilities.
- [x] Chronological recency/feature selection and temperature calibration.
- [x] Practice, sprint and archived forecast inputs with cutoff audits.
- [x] Separate historical evaluation from live upcoming-race prediction.
- [x] Synthetic regression tests for chronology, probability arithmetic and data handling.
- [x] Publish the initial 17-race benchmark, including unsuccessful comparisons.

## Next milestones

- [ ] Record predictions before future sessions/races, then score them unchanged.
- [ ] Evaluate calibration and podium accuracy over a larger set of unseen races.
- [ ] Support selecting any circuit/round instead of only Austin.
- [ ] Split the shared data adapter from the older model into smaller modules.
- [ ] Add cross-platform offline CI and verify dependency compatibility.
- [ ] Pin source revisions/releases and record download hashes for reproducibility.
- [ ] Improve long-run comparisons using traffic and stint context where supported.
- [ ] Test shared retirement, weather and strategy risks in race simulations.

## How to show progress

For each change, record its data cutoff, features, training window, calibration
period and evaluation races. Compare with the same baseline on the same races.
Report winner accuracy, full unordered and ordered podium hit rates, mean correct
podium drivers, Brier scores and uncertainty. Keep development/selection results
separate from genuinely new prospective results. Avoid retuning to the same
small evaluation set and describing that as new independent evidence.
