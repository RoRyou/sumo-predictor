# Phase 1 Results — Route A (Structural / Tabular)

> Data: sumo-api 2015-01 .. 2024-11 (Makuuchi), 17,586 bouts, 679 rikishis.

## Headline numbers

| Setup | Val acc | Test acc | LogLoss | AUC |
|---|---:|---:|---:|---:|
| **Stacked XGB+LGBM+CatBoost** (T1+T3+T4+T5) | 60.7% | **59.4%** | 0.673 | 0.612 |
| Single XGB (n_est=600, lr=0.03) | – | 57.8% | 0.690 | – |
| Stacked + symmetric augment (T6) | 56.4% | 57.7% | 0.731 | 0.624 |
| **Walk-forward macro** (24 basho, single XGB) | – | 56.7% | 0.686 | 0.600 |

Test set = 6 basho (2024-01 .. 2024-11), 1,791 bouts.

Trivial baselines: east-win rate = 50.2%; "higher-ranked wins" ≈ 55-58% (estimate).

## Tier breakdown (test)

| rank-gap bin | n | acc |
|---|---:|---:|
| very-close (|Δ|≤50) | 1,241 | 57.8% |
| close (50<|Δ|≤200) | 262 | 60.7% |
| mid-gap (200<|Δ|≤500) | 288 | 65.3% |

Model exploits rank gap well; near-equal matchups remain near-random.

## What helps (feature importance, single-XGB)

Top 10: `winrate_diff_90`, `rank_diff`, `streak_A`, `career_winrate_A`, `te__shusshin_B`, `te__shusshin_A`, `career_winrate_B`, `winrate_A_90`, `career_bouts_A`, `streak_B`.

Notable: `h2h_winrate` (Bayesian-shrunk) is in the top 12 — confirms T2 pays off despite 80% of bouts having prior history.

## Tricks applied so far

| Trick | Applied | Effect |
|---|---|---|
| T1 KFold target encoding | ✅ | top-10 features include `te__shusshin_*` |
| T2 Bayesian h2h shrinkage | ✅ | reduces noise for sparse pairs |
| T3 Time-decay sample weight | ✅ | λ=0.05 per basho |
| T4 XGB+LGBM+CatBoost stacking | ✅ | +2pp over single XGB |
| T5 Isotonic calibration | ✅ | neutral on acc, fixes ECE |
| T6 Symmetric augmentation | ⚠️ | **hurts** — likely TE leakage between mirrors |

## Pending tricks

- T7 Kimarite克制 matrix (lookup table)
- T8 Focal loss for hard samples
- T15 Basho stage (day-binning) features
- Hyperparameter tuning with Optuna

## Backtest stability

Walk-forward 2021-01 .. 2024-11 (24 basho):
- macro accuracy 56.7%, range 47.4% .. 61.3%
- 2024 basho average ≈ 58% (improving with more data)

## Next steps

1. Investigate T6 augment regression (likely fix: augment **before** target encoding).
2. Add T7/T8/T15.
3. Run Optuna (50 trials, 20 min budget).
4. Move on to Phase 2 (video pose) once stacked ≥ 61% stable.
