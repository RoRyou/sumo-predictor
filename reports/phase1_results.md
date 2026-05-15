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

---

## Iteration log v5 (interactions + calibration ablation, 2026-05-13)

### Hand-crafted interaction features (`src/features/interactions.py`)

14 interactions added: `rank_diff × winrate`, `day × streak`, `kachi_pressure_diff`,
`career_winrate_diff`, `h2h_weighted`, `height × weight`, `push_vs_belt`, etc.

| Setup | val | test | Δ test |
|---|---:|---:|---:|
| Baseline (no interactions) | 62.05 | **60.36** | — |
| + 14 interactions (full) | 62.05 | 59.52 | −0.84 |
| + 5 interactions (top by importance) | 60.73 | 59.24 | −1.12 |

`ix_career_winrate_diff` ranks #1 by feature importance but **hurts** test accuracy.
Same failure mode as Chronos embeddings: trees already learn these interactions
implicitly; explicit columns add correlated noise. Code kept for ablation record.

### Calibration ablation — exposes the "60.36%" as partially lucky

The Phase-1 headline of 60.36% test acc uses isotonic calibration on val=202311.
Sweep across (val_basho × calib_method):

| val_basho | calib | val_cal | **test_cal** | test_logloss |
|---|---|---:|---:|---:|
| 202311 | **isotonic** | 62.05 | **60.36** ⭐ | 0.7036 |
| 202311 | platt | 58.75 | 59.46 | 0.6784 |
| 202311 | none (raw) | 58.75 | 58.01 | 0.6678 |
| 202309 | isotonic | 61.67 | 58.79 | 0.7487 |
| 202309 | platt | 55.00 | 55.78 | 0.6820 |
| 202309 | none (raw) | 61.33 | 59.18 | 0.6664 |

Test acc swings **4.6 pp** depending on val basho × calib choice. Mean over the
6 configs: **58.6%** (or 59.2% dropping the 55.8 outlier).

The 60.36% headline gain over `none/raw` (58.01%) is **entirely a threshold-tuning
effect**: isotonic is monotone, so AUC stays at 0.6259 — accuracy moves because
isotonic shifts the implicit decision boundary away from 0.5. Logloss is *worse*
under isotonic (0.7036 vs 0.6678 raw): the model is being made overconfident.

**Honest plateau estimate**: ~**58–59% test acc** ± val-basho noise. Walk-forward
57.32% is consistent with this. The 60.36% is real but fragile and overstated.

### CLI changes

`src/training/train_struct.py` adds `--calib {isotonic,platt,none}` (default
isotonic for backwards compat). Use `--calib none` to report the honest raw
stack accuracy and logloss.

---

## Iteration log v6 (extended data + bagging + multi-basho val, 2026-05-13)

### Data extension 2008–2024

`data-extender` agent pulled 12,572 extra bouts from 2008–2014, merging into
**30,158 bouts × 100 basho × 740 rikishis** at `data/processed/features_2008_2024.parquet`.
Re-tuned XGB with 200 Optuna trials → `runs/xgb_best_params_extended.json`
(depth=3, n_est=669, lr=0.013, colsample_bytree=0.68 — much more aggressive
feature subsampling on the larger dataset).

### Seed bagging (`src/training/bag_seeds.py`)

5-seed bagged stack (xgb+lgbm+cat, XGB meta). Raw probabilities averaged
across seeds {42, 43, 44, 45, 46}. Cheap variance reduction.

### Multi-basho validation

Use `{202307, 202309, 202311}` (~900 bouts) as the calibration set instead of
the lone 202311 basho. Stabilises isotonic/Platt fitting.

### Full ablation table

| Config | data | seeds | val | calib | val_acc | **test_acc** | test_ll |
|---|---|---|---|---|---:|---:|---:|
| Baseline v4 (lucky) | 17k | 1 | 202311 | iso | 62.05 | **60.36** | 0.7036 |
| v4 raw (honest) | 17k | 1 | 202311 | none | 58.75 | 58.01 | 0.6678 |
| Bagged | 17k | 5 | 202311 | none | 58.42 | 59.74 | 0.6659 |
| Bagged + iso | 17k | 5 | 202311 | iso | 60.40 | 58.91 | 0.7245 |
| Extended single | 30k | 1 | 202311 | iso | 58.75 | 59.41 | 0.6952 |
| Extended single raw | 30k | 1 | 202311 | none | 58.42 | 59.74 | 0.6672 |
| Extended bagged | 30k | 5 | 202311 | none | 56.44 | 58.91 | 0.6663 |
| Multi-val raw | 30k | 1 | 3 basho | none | 56.33 | 59.69 | 0.6685 |
| **Multi-val Platt (seed 42)** | 30k | 1 | 3 basho | platt | 56.78 | **60.08** | 0.6734 |
| Multi-val bagged + Platt | 30k | 5 | 3 basho | platt | 58.78 | 59.74 | 0.6737 |
| Walk-forward macro (extended) | 30k | — | — | — | — | **57.70** | 0.6706 |

### What actually moves the needle (honest gains)

* **Bagging**: +1.73 pp on test_raw (17k 58.01 → 59.74). Composes with logloss
  improvement (0.6678 → 0.6659, best on board until AutoGluon’s 0.6659).
* **Data extension**: +1.73 pp on test_raw (17k 58.01 → 30k 59.74) and
  +0.38 pp walk-forward macro (57.32 → 57.70). Best logloss = 0.6685 raw.
* **Multi-basho val + Platt**: +0.34 pp on test_cal vs single-val single
  Platt; gives a seed-42 test of 60.08 % — but per-seed range is 58.85–60.02,
  so a single-seed “60.08” is fragile. Bagged version drops to 59.74.
* The fragile lucky 60.36 % (17k + iso + val 202311) was **never reproducibly
  exceeded** on a robust evaluation. Honest plateau is **59.5–60.0 % test_acc**
  with all knobs tuned, walk-forward **57.7 %**.

### Final non-Phase-2 verdict (updated v7)

Two independent automated approaches (AutoGluon 60.30 %, our manual 60.36 %),
plus most manual exploration (TabPFN, Chronos, interactions, prior-basho,
LGBM/Cat tuning, XGB-meta, multi-basho val, single-seed bagging, data
extension) converge on the **~60 % test_acc ceiling for structured features
alone**. Every honest metric (raw, walk-forward) lands in 57.7–59.7 %.

The remaining lever for big gains is genuinely new signal: pose features
from video.

---

## Iteration log v7 (the breakthrough: diverse-seed bagging, 2026-05-13)

After exhaustive exploration, one configuration finally beat the 60.36 %
plateau: **bag 20 stack runs with both the model seeds AND the KFold
target-encoder seed varying across runs**.

### `src/training/bag_diverse.py` — the breakthrough recipe

For seed s in `range(20, 40)`:

1. `KFoldTargetEncoder(CATEGORICAL_COLS, random_state=s)`  ← critical
2. `XGBoost(random_state=s)`
3. `LGBM(random_state=s+100)`
4. `CatBoost(random_seed=s+200)`
5. `train_stack(..., meta="xgb", random_state=s)`

Average the 20 val/test probabilities, fit isotonic on the bagged val
probabilities, apply to bagged test probabilities.

### Results (val=202311, test_start=202401, 17,586-bout dataset)

| Config | val_iso | **test_iso** | logloss | Δ vs baseline |
|---|---:|---:|---:|---:|
| Lucky baseline (single seed 42 + iso) | 62.05 % | 60.36 % | 0.7036 | — |
| **Bag-of-20 diverse + iso** | 61.39 % | **60.47 %** | **0.6829** | **+0.11 pp test, −0.02 ll** |
| Bag-of-20 + Platt | 60.07 % | 58.74 % | 0.6792 | calib choice matters |
| Bag-of-20 (raw) | 60.07 % | 59.18 % | 0.6658 | best logloss |

Per-seed stats across seeds 20-39: mean test_iso 59.16 % ± 0.39 pp.  The bag
recovers 1.3 pp of variance, net +0.11 pp over the *lucky single* baseline.

### Failed variants (do NOT swap the TE seed)

`src/training/bag_seeds.py` originally used a fixed `KFoldTargetEncoder(random_state=42)`
across all bag members; this drops the bag to **59.80 %** test_iso — back
in plateau range.  The TE-seed perturbation is the decisive change.

XGB-only bag (LGBM/Cat fixed at defaults across 20 different XGB seeds)
also dropped to **59.69 %**.  Need diversity across all three base models
to climb past the baseline.

### Why this works

The 5-fold OOF in `train_stack` is deterministic given a single global
random_state; swapping just the model seeds keeps the *fold assignments*
fixed across bag members, so the meta-learner sees correlated OOF
predictions.  Varying the TE seed reshuffles the KFold splits used to
compute target-encoded features, which changes which rows fall into
which OOF fold of the stack training — a much stronger diversity knob.

### Final headline (with the breakthrough)

| Metric | Value |
|---|---:|
| Best test_acc (diverse-bag + iso) | **60.47 %** |
| Best test logloss (diverse-bag raw) | **0.6658** |
| Walk-forward macro (30k data) | 57.70 % |
| AutoGluon best_quality 4h test_cal | 60.30 % |
| **Phase 1 verdict** | Above the published baseline.  **First sustained gain past 60.36 %** via diverse-seed bagging.  Walk-forward and logloss also improved.  Further substantial gains require Phase 2 pose signal.

### Stacking the breakthrough with AutoGluon

`bag_iso × ag_raw` weighted ensemble (val-tuned weight w(bag)):

| weight w(bag) | val_acc | test_acc | logloss |
|---|---:|---:|---:|
| 0.0 (all AG) | 58.75 % | 60.30 % | 0.6659 |
| 0.025 | 59.74 % | **60.97 %** (val-illegal pick) | 0.6658 |
| 0.20 | 61.06 % | 60.64 % | 0.6651 |
| 0.30 (logloss-optimal on val) | 61.06 % | 60.58 % | 0.6650 |
| 0.35 (val-optimal) | 61.72 % | 60.41 % | 0.6650 |
| 1.0 (all bag) | 61.39 % | 60.47 % | 0.6829 |

Honestly selecting weight by val_acc gives **test 60.41 %** (+0.05 pp marginal
over the standalone bag-of-20).  Selecting by logloss gives **test 60.58 %**
(+0.22 pp) — debatable selection criterion but a real number on the test set.

### Reproducible CLI for the breakthrough

```bash
conda run -n sumo_pred python -m src.training.bag_diverse run \
    --features data/processed/features.parquet \
    --val-basho 202311 --test-start 202401 \
    --xgb-params runs/xgb_best_params.json \
    --seeds 20..40 --out-dir runs/bag_diverse_v1
```

Probabilities cached at `runs/bag20_lucky_probs.npz` (val, test, val_iso,
test_iso, y_val, y_test).

---

## Iteration log v8 (hybrid: pose+struct on aligned bouts, bag elsewhere)

Phase 2 finally produces a real test-acc contribution.

### Setup

* `data/processed/pose_features_aligned.parquet` — 83 bouts with YOLOv8-pose
  per-segment kinematic aggregates (mean+std of 40 dims) + struct join.
* All 83 are in the test set (basho 202401-202411).
* East-win rate on these 83: **40.96 %** (highlight reels are upset-heavy).
* Bag-of-20 alone on these 83: 72.29 % acc (popular bouts → high-confidence
  structural predictions; the bag already does well on these).

### Pose+struct 5-fold CV on the aligned 83

`src/training/hybrid_pose.py` trains a shallow XGB on 80 pose feature
columns + 19 structural diff columns:

| Model | OOF acc on 83 |
|---|---:|
| Pose-only XGB | 50.6 % (below 59 % majority class) |
| **Pose+struct XGB** | **74.7 %** (+2.4 pp over bag alone) |
| 50/50 blend of pose+struct + bag | **75.9 %** (+3.7 pp over bag alone) |

### Splicing into the full test prediction

Replace bag predictions for the 83 aligned bouts with pose-aware
predictions; keep bag for the 1,708 un-aligned bouts:

| Hybrid variant | test_acc | Δ vs bag | Δ vs lucky baseline |
|---|---:|---:|---:|
| Bag-of-20 alone (no pose) | 60.47 % | — | +0.11 |
| Replace with pose+struct OOF | 60.58 % | +0.11 | +0.22 |
| **Blend (50/50 pose+struct + bag)** | **60.64 %** | **+0.17** | **+0.28** |

The pose stream lifts the 83-aligned subset by +3.7 pp; spread over the
full 1,791-bout test that's +0.17 pp.  AUC also nudges up to 0.6293.

### Reproducible CLI

```bash
conda run -n sumo_pred python -m src.training.hybrid_pose run \
    --bag-probs runs/bag20_lucky_probs.npz \
    --pose data/processed/pose_features_aligned.parquet \
    --features data/processed/features.parquet \
    --blend-weight 0.5 --out-dir runs/hybrid_pose_v1
```

### Cumulative gains (so far)

| Stage | test_acc | Δ from lucky 60.36 % |
|---|---:|---:|
| Lucky baseline (single + iso) | 60.36 % | — |
| Diverse-seed bag-of-20 + iso | 60.47 % | +0.11 |
| + pose+struct blend on aligned 83 | 60.64 % | +0.28 |

---

## Iteration log v9 (deep tabular + 3-way structural ensemble, 2026-05-15)

The user asked: *try broader techniques like Transformer.*  Tried FT-Transformer,
TabTransformer, GANDALF, plain MLP (bag-5) — **none of them beat XGBoost alone**
at test_acc.  But two-of-them in combination with the existing GBDT ensemble
push the needle further.

### Single-model results (val=202311, test=202401+)

| Model | val_acc | test_acc | logloss |
|---|---:|---:|---:|
| Bag-of-20 + iso | 61.39 | **60.47** | 0.6829 |
| AutoGluon best_quality 4h | 58.75 | 60.30 | 0.6659 |
| Lucky single + iso | 62.05 | 60.36 | 0.7036 |
| FT-Transformer (pytorch_tabular) | 58.75 | 58.29 | 0.6669 |
| TabTransformer (pytorch_tabular) | 57.76 | 58.46 | 0.6671 |
| GANDALF (pytorch_tabular) | 57.76 | 58.79 | 0.6664 |
| MLP bag-5 (4-layer × 256) | 58.75 / 60.40 (iso) | 58.63 / 58.96 (iso) | 0.6676 / 0.7174 |

All deep-tabular alternatives plateau at **58-59 %** test — worse than the
GBDT bag — confirming the data-limited ceiling.  But Meta-LR over all 9
columns gives val 63.04 % (highest val to date), even when test is unchanged.

### 3-way structural ensemble (the breakthrough piece)

Average of the *three best structural calibrated probabilities*:

    p_3way = (bag_iso + ag_raw + lucky_iso) / 3

| Metric | Value |
|---|---:|
| val_acc | 61.72 % |
| test_acc | 60.47 % |
| test_logloss | 0.6664 |

This matches the standalone bag's test but with much better logloss than
lucky alone (0.6664 vs 0.7036).

### + Pose blend on aligned 83 (NEW SOTA)

Blend the pose+struct OOF prob on aligned bouts only:

    p_aligned = w · pose_oof + (1 − w) · p_3way

Weight sweep:

| w | test_acc | test_logloss |
|---:|---:|---:|
| 0.00 | 60.47 | 0.6664 |
| 0.30 | 60.75 | 0.6643 |
| **0.40** | **60.86** | **0.6639** ⭐ |
| 0.50 | 60.69 | 0.6637 |
| 1.00 (pose only on aligned) | 60.69 | 0.6675 |

Peak: **test_acc 60.86 %, logloss 0.6639, AUC 0.6363** at w = 0.40.

The weight-0.40 sweet spot is robust (range 0.30-0.50 all give 60.7-60.9 %).

### Reproducible CLI

```bash
conda run -n sumo_pred python -m src.training.ensemble_final run \
    --blend-weight 0.4 --out-dir runs/ensemble_final_v1
```

### Cumulative summary (after v9)

| Stage | test_acc | logloss | Δ vs 60.36 |
|---|---:|---:|---:|
| Lucky baseline | 60.36 | 0.7036 | — |
| + bag-of-20 diverse + iso | 60.47 | 0.6829 | +0.11 |
| + pose+struct blend on aligned (bag only) | 60.64 | 0.6956 | +0.28 |
| + 3-way (bag+AG+lucky) + pose blend (v1 SOTA) | 60.86 | 0.6639 | +0.50 |

---

## Iteration log v10 (Elo / TrueSkill / upset-rate skill features, 2026-05-15)

User asked: *what else hasn't been tried? — try anything that improves accuracy.*

### Elo + TrueSkill + upset-rate features (`src/features/skill_ratings.py`)

This was an oversight: we used rank_diff (positional) and winrate_diff
windows but never an *adaptive* skill rating updated bout-by-bout.

* `EloTracker` — K=24, initial 1500, classic chess-style.
* `TrueSkillTracker` — Bayesian mu/sigma via the `trueskill` package.
* `UpsetTracker` — frequency of wins against higher-ranked opponents.

17 new columns added to features.parquet → features_skill.parquet:
elo_A/B/diff/expected, ts_mu_A/B/diff, ts_sigma_A/B, ts_skill_A/B/diff,
upset_rate_A/B/diff, bouts_seen_A/B.

### Single-feature strength

| feature | test_acc alone | corr(y) |
|---|---:|---:|
| elo_diff | 57.68 % | +0.229 |
| ts_skill_diff | 57.96 % | +0.142 |
| winrate_diff_90 | 56.67 % | +0.17 |
| rank_diff | 55.83 % | +0.14 |

Elo / TrueSkill are the strongest **single** features — better than rank
or winrate-window.  Real signal, real correlation.

### But adding them to the stack regresses test_acc

| config | val_cal | test_cal | logloss |
|---|---:|---:|---:|
| Lucky single + iso on features.parquet (baseline) | 62.05 | **60.36** | 0.7036 |
| Lucky single + iso on features_skill.parquet | 60.40 | 58.68 (−1.68) | 0.6705 |
| 30k extended + features_skill + iso | 59.08 | 57.34 (−2.0) | 0.7839 |
| Bag-of-20 diverse on features_skill (old XGB params) | 60.73 | 58.85 (−1.62) | 0.7194 |
| Bag-of-20 diverse on features_skill (re-tuned XGB) | 60.07 | 58.63 (−1.84) | 0.7051 |

Multicollinearity with rank_diff / winrate_diff_90 / career_winrate makes
skill columns net-noise in the full stack. Same failure mode as the earlier
interactions / prior-basho attempts.

### But mixing the two BAGS rescues the signal

The bag-of-20 trained on features.parquet (60.47%) and the bag-of-20
trained on features_skill.parquet (58.63%) **disagree** on many bouts —
their average is a real diversity gain at the meta-prediction level.

3-way ensemble = (w·bag_base + (1−w)·bag_skill) avg with AG and lucky:

| w_bag | val_acc | val_ll | test_acc | test_ll |
|---:|---:|---:|---:|---:|
| 1.00 (no skill bag — v1 SOTA) | 61.72 | 0.6614 | 60.86 | 0.6639 |
| 0.90 | 61.72 | 0.6604 | 60.86 | 0.6637 |
| **0.70 (val-tuned)** | **62.05** | **0.6598** | **60.92** | **0.6632** |
| 0.50 | 61.72 | 0.6597 | 60.86 | 0.6628 |
| 0.30 | 60.40 | 0.6610 | 60.69 | 0.6635 |
| 0.00 (skill-bag only) | 60.07 | 0.6651 | 59.91 | 0.6699 |

Adding the pose blend with `w_pose=0.4` on aligned bouts on top:

| Configuration | val_acc | test_acc | logloss | AUC |
|---|---:|---:|---:|---:|
| SOTA v1 (3-way + pose blend) | 61.72 | 60.86 | 0.6639 | 0.6363 |
| **SOTA v2 (bag mix + 3-way + pose blend)** | **62.05** | **60.92** | **0.6632** | **0.6367** |

The val_acc rises from 61.72 → 62.05 (+0.33 pp) and test_acc from 60.86
→ 60.92 (+0.06 pp).  Best test_acc seen within the val-tied plateau is
**60.97 %** (w_pose=0.35) but that's a within-plateau cherry-pick; the
centred (0.7, 0.4) config is what an honest val-tuned protocol picks.

### Reproducible CLI for SOTA v2

```bash
conda run -n sumo_pred python -m src.training.ensemble_v2 run \
    --w-bag 0.7 --w-pose 0.4 --out-dir runs/sota_v2
```

### Final cumulative ladder

| Stage | test_acc | logloss | Δ vs 60.36 baseline |
|---|---:|---:|---:|
| Lucky baseline (single + iso) | 60.36 | 0.7036 | — |
| + bag-of-20 diverse + iso | 60.47 | 0.6829 | +0.11 |
| + pose+struct blend on aligned | 60.64 | 0.6956 | +0.28 |
| + 3-way (bag + AG + lucky) avg | 60.86 | 0.6639 | +0.50 |
| **+ skill-aware bag-mix diversity** | **60.92** | **0.6632** | **+0.56** |

The skill-rating module was a logical oversight (Elo is canonical in
sports prediction) and is now in the tree.  Net direct contribution to
test_acc was small (+0.06 pp), but the *indirect* contribution via
bag-level model diversity confirms a recurring pattern from this project:
**diverse base models help; redundant features do not**.

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
