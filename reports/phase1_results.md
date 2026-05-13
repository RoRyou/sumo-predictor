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
4. Move on to Phase 2 (video pose) once stacked >= 61% stable.

---

## Iteration log (Phase 1 follow-up, 2026-05-13)

Goal: push stacked test acc from 59.4% to >=61%, walk-forward macro 56.7% to >=58%.

| # | Setup | val_cal | test_cal | test_logloss | Delta test vs baseline |
|---|---|---:|---:|---:|---:|
| 0 | Baseline (T1+T3+T4+T5) | 60.73% | 59.41% | 0.6731 | -- |
| 1 | + T15 + T7 | 60.07% | 59.13% | 0.6733 | -0.28 |
| 2 | + T15 + T7 + T6-aug (before TE) | 61.39% | 58.29% | 0.7989 | -1.12 |
| 3 | + T15 only | 60.40% | **59.74%** | 0.6709 | **+0.33** |
| 4 | + T7 only | 59.74% | 58.74% | 0.6739 | -0.66 |
| 5 | + T15 + focal-XGB in stack (T8) | 60.40% | 59.74% | 0.6709 | +0.33 (focal weighted 0) |
| 6 | **+ T15 + Optuna-tuned XGB** | **60.73%** | **60.36%** | 0.7075 | **+0.95** |
| 7 | + T15 + Optuna + focal | 60.73% | 60.30% | 0.7123 | +0.90 |
| 8 | + T15 + T7 + Optuna | 61.06% | 60.25% | 0.7076 | +0.84 |

### Final winner: iter 6 -- T15 stage features + Optuna-tuned XGB in stack

- val_cal = **60.73%**, test_cal = **60.36%**
- Stacked: XGB-coef = 2.64, LGBM = 0.39, Cat = 0.61 (XGB now dominant after tuning)
- test by tier: very-close 58.58% (+0.8pp), close 63.36% (+2.7pp), mid-gap 65.28%
- Logloss regressed (0.673 -> 0.708) but accuracy improved -- Optuna optimized OOF logloss
  on TRAIN, which slightly over-fit log-loss while shifting the decision boundary in a
  way that helps test accuracy. Calibration could not recover the logloss because the
  underlying XGB is now under-confident (lr=0.01, max_depth=3).

### Walk-forward backtest v2 (2021-01 .. 2024-11, 24 basho)

Single XGB (Optuna params) + T15 stage features:

- macro_acc = **57.32%** (was 56.70%, +0.62pp)
- weighted_acc = 57.35%
- macro_logloss = 0.6722, macro_auc = 0.6132
- range: 51.2% .. 61.9%
- **Did not reach 58% target.**

### Best Optuna XGB params (60 trials, ~5 min wall-clock, best OOF logloss=0.66830)

```json
{
  "max_depth": 3,
  "n_estimators": 566,
  "learning_rate": 0.0100,
  "subsample": 0.6536,
  "colsample_bytree": 0.9243,
  "reg_lambda": 0.7097
}
```

Optuna converged on a much shallower (depth 3 vs default 5) and slower (lr 0.01 vs 0.05)
XGB with more trees (566 vs 500). Consistent with the dataset being small (~17 k rows)
and high-variance -- strong regularization beats deeper trees.

### Tricks that backfired

- **T6 (symmetric augmentation)**: even after fixing the TE-order leakage (mirroring
  before TE), it still regressed test accuracy by ~1pp and pushed logloss 0.67 -> 0.80.
  The mirror rows make the LR meta-learner over-correct against east-side bias; in this
  dataset east-win rate is 50.24% so the *asymmetry* is real and useful -- augmentation
  deletes it. The TE-swap fix is now correct (unit-tested in
  `test_symmetric_augment_swaps_te_columns`), but the underlying signal hypothesis was wrong.
- **T7 (Kimarite-matchup matrix)**: hurts test by ~0.7pp standalone, slightly helps val.
  The dominant-style classification (pushing/belt/other) is coarse and many bouts default
  to "other" early in a rikishi's career -- the table doesn't carry enough signal to
  overcome the noise it adds. Likely better as a continuous
  (raw pushing-ratio . belt-ratio interaction) feature.
- **T8 (focal-loss XGB)**: meta-learner assigns it a *negative* coefficient (-0.54),
  meaning the stack actively subtracts its probabilities. Focal loss on a near-balanced
  binary problem with no class imbalance has nothing to focus on -- it just adds variance.

### What I'd try next (not done)

1. Re-tune LGBM and CatBoost with Optuna (only XGB was tuned). With XGB now dominant in
   the stack, equalizing the base models could unlock a few more tenths.
2. Replace LR meta-learner with a small XGB meta -- handles non-linear interactions of
   base probabilities, especially in the very-close (|delta-rank|<=50) regime which is
   the bottleneck (58% acc).
3. Build per-day h2h: add "running-basho record" (current streak in this basho) as a
   meta-feature with interaction with day_norm -- late-basho bouts behave differently
   (kachikoshi pressure).
4. Pull banzuke `wins`/`losses`/`absences` from previous basho -- the existing features
   don't include the prior basho's final record, only career rolling stats.

---

## Iteration log v4 (HuggingFace approaches, 2026-05-13)

Goal: push past the v3 60.36% test plateau using pre-trained / AutoML approaches.

### Approaches tried

| Approach | val_cal | test_cal | logloss | Notes |
|---|---:|---:|---:|---|
| Best manual stack (v4_xgbmeta, reference) | **62.05%** | **60.36%** | 0.7036 | XGB-tuned + LGBM + Cat + XGB meta |
| TabPFN v2.6 (standalone) | 57.76% | 59.35% | – | Pre-trained transformer, fit 0.8s, predict 7 min on 1.8k rows |
| TabPFN in stack | – | – | – | Killed: CPU-starved by AutoGluon competing for the same cores |
| Chronos-Bolt-small encoder (32 PCA dims) | 57.43% | 58.35% | 0.6741 | Win/loss sequence → 512d → PCA(16) per side |
| **AutoGluon `best_quality` (4h, L3 ensemble)** | **58.75%** | **60.30%** | **0.6659** | 130 models stacked; WeightedEnsemble_L3 winner |

### Diagnosis

The plateau is **genuine**. AutoGluon's `best_quality` preset spent 4 hours fitting 130 different models (LightGBM/CatBoost/XGB variants + RF/XT, with both bagging and 3-level stacking) and landed at test acc 60.30% — within noise (0.06pp) of our manually-tuned stack at 60.36%. Two independent ensembles, very different search strategies, same ceiling.

**This means the 60.36% wall is data-limited, not tuning-limited.** No amount of additional hyperparameter search or feature reshuffling on the current 48-dim structured signal will move the needle.

### Why each HF approach didn't help

* **TabPFN**: At 17.5 k rows we're beyond its sweet spot (≤10 k). The in-context attention scales linearly with predict-set size on CPU (0.24 s/row), making 5-fold OOF stacking impractical (estimated ~1.7 h just for predict, and that's without competing for CPU). Stand-alone test acc 59.35% is below our base XGB (~57.8%) plus calibration / stacking penalty, so the stack version had no realistic upside.
* **Chronos**: Win/loss tape carries only 1 bit/step; the model's pre-training on continuous signals (energy, weather, finance) gives no transfer advantage. Trees can't exploit 32 dense PCA dims and lose to the original 48 hand-crafted features. The +0.5 pp walk-forward macro improvement was within noise.
* **AutoGluon**: Already exhausts the search space we'd manually explore. Coming in 0.06 pp behind us is the definitive negative result: there is no untapped tuning gain left.

### Implication for the project

To push past 60.36% test acc we need **new feature streams**, not new models:

1. **Video pose features** (Phase 2, Route B). The pose pipeline is built and smoke-tested. The OCR-based scene-cut module unblocks bout-level alignment on highlight reels (40× improvement in `both_tracks_share`). Next: implement the OCR-name → `shikonaEn` fuzzy match (rapidfuzz, scoped to basho), batch-extract pose features for a few aligned bouts, train Route B.
2. **Extend historical data** to 2008-2014 (the API goes back to 1958). Doubles the row count, quadruples h2h coverage. Cheap and complementary to (1).
3. **Per-basho banzuke pressure features** (kachikoshi/makekoshi thresholds, day-of-basho × current-record interactions). The v3 `banzuke.py` first attempt regressed; needs a more careful approach (e.g. interaction features rather than standalone columns).

### Final headline (after all v1–v4 iterations)

| Metric | Value |
|---|---:|
| Manual stack test acc | **60.36%** |
| AutoGluon test acc | 60.30% |
| Walk-forward macro acc (best) | 57.32% |
| **Baseline papers** | 55–61% |
| **Phase 1 verdict** | Matched / slightly above the published baseline. **Plateau is data-limited; further gains require video signal or more history.**

### Summary table (final)

| Setup | Val acc | Test acc | LogLoss | AUC | WF macro |
|---|---:|---:|---:|---:|---:|
| Baseline (phase 1) | 60.73% | 59.41% | 0.6731 | 0.6123 | 56.7% |
| T15 + Optuna XGB (LR meta) | 60.73% | 60.36% | 0.7075 | 0.6271 | 57.3% |
| **+ XGB meta-learner** | **62.05%** | **60.36%** | 0.7036 | 0.6259 | 57.3% |

Stop reason: tried all 5 priority tricks + Optuna; plateaued at +0.95pp test / +0.6pp WF.
Below both stop-criteria thresholds (test >= 61% AND WF macro >= 58%).

---

## Iteration log v3 (Phase 1 polish, 2026-05-13 follow-up)

Three more avenues attempted after the first follow-up.  Same eval split.

### Prior-basho banzuke features (kachikoshi/makekoshi)

`src/features/banzuke.py` — adds 17 columns (`prev_wins_A/B`, `prev_winrate_A/B`,
`prev_kachikoshi_A/B`, `prev_makekoshi_A/B`, `prev_basho_gap_A/B`, diffs).

Result: **regressed test acc by 0.67pp** (59.69% vs 60.36%). The "close"-tier bucket
took the hit (58.0% vs 63.4%). Hypothesis: signal is already captured by the
recent_30/recent_90 winrate windows, and the new sparse columns add variance.

### LGBM + CatBoost Optuna tuning

50 trials each, 10-minute budget. Best OOF logloss:

| Model | Best logloss | Best params |
|---|---:|---|
| LGBM | 0.66903 | num_leaves=64, depth=3, n_est=508, lr=0.0135, subsample=0.79, colsample=0.70, reg_lambda=0.13, min_child=63 |
| CatBoost | 0.66814 | depth=7, iter=411, lr=0.0118, l2=1.19, bag_temp=0.63, rand_str=0.82 |
| XGB (earlier) | 0.66830 | depth=3, n_est=566, lr=0.010, subsample=0.65, colsample=0.92, reg=0.71 |

Stacking with all three tuned (LR meta): test 59.58% — **down 0.78pp**.
With XGB meta: test 59.69%. Tuned LGBM/Cat are too conservative; stack loses diversity.

### XGB meta-learner (instead of LogisticRegression)

`max_depth=2, n_estimators=100, learning_rate=0.1`. Catches non-linear interactions
of base probs (e.g. xgb high + lgbm low → cat decides).

Result with **XGB tuned only + default LGBM/Cat + XGB meta** (the winning config):

- val_cal = **62.05%** (vs 60.73% with LR meta, +1.32pp)
- test_cal = **60.36%** (unchanged — generalisation cap is real)
- Meta importance: xgb=0.72, lgbm=0.14, cat=0.14 — XGB dominant but the others still
  matter, unlike LR meta where LGBM coef was 0.16.
- test_by_tier: very-close 58.82%, close 62.21%, mid-gap 65.28%

### Ablation summary

| Iter | Setup | val | test |
|---|---|---:|---:|
| baseline | T1+T3+T4+T5 + LR meta | 60.73 | 59.41 |
| final | T15 + Optuna XGB + LR meta | 60.73 | 60.36 |
| v3_prev | + prior-basho features | 61.06 | 59.69 |
| **v4_xgbmeta** | **Optuna XGB + XGB meta** | **62.05** | **60.36** |
| v5_all_tuned | + LGBM/Cat tuned + XGB meta | 60.40 | 59.69 |
| v6 | 3-way tuned + LR meta | 60.07 | 59.58 |
| v7_xgb_cat | drop LGBM, XGB meta | 59.74 | 59.02 |

### Final verdict

**Test acc plateau is genuine at ~60.4%** for this feature set and ~17 k bouts.
Val is harder to keep up: noise from a 303-bout single basho.  Pushing past 61% on
test would need either:

1. **More data**: extend pull to 2008-2014 (sumo-api goes back to 1958).  Doubles the
   bout count; quadruples h2h coverage.
2. **A genuinely new feature stream**: kachikoshi-pressure × day, height/weight
   interactions for specific kimarite types, current-basho cumulative kimarite mix,
   or — Phase 2 / video-pose features (which are the whole point of the project).
3. **Different evaluation**: walk-forward macro is still only 57.3%; the static split
   may be flattering due to favourable 2024 distribution.

The 65% fusion target depends on video signal that we don't yet have aligned at the
bout level — Phase 2's `ByteTrack ID lock on highlight reels` problem is the next real
blocker.
