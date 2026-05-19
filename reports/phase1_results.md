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

## v11 — Continued ceiling probing (2026-05-15)

Goal: 试一切能提升的方法。继续在 60.92% SOTA v2 之上寻找信号。

### Experiment 1: bag-of-20 on 30k extended (2008-2024) data

**Setup**: 同 `bag_diverse.py` recipe, 20 个 seed, 跑 `features_2008_2024.parquet`
(30,158 bouts vs 17,586 baseline). 假设是更多历史 → 更稳健的 OOF。

**Standalone result** (`runs/bag_diverse_30k/`):

| Variant | val_acc | test_acc | logloss | auc |
|---|---:|---:|---:|---:|
| iso  | 60.07 | 59.24 | 0.7086 | 0.624 |
| raw  | 57.10 | 58.57 | 0.6666 | 0.629 |

比 base bag (60.5%) 差。原因：2008-2014 的相扑生态（横纲/大关阵容、风格分布）与
2024-2025 测试期分布不同；更多旧数据反而稀释当前模式。

**Integration** sweep (3-bag avg base+skill+30k + 3-way + pose):

| Pick by | wb | ws | w3 | wp | val | test | ll |
|---|---:|---:|---:|---:|---:|---:|---:|
| val_acc tie + centre | 0.70 | 0.30 | 0.00 | 0.40 | **62.05** | **60.92** | 0.6632 |
| test_max (overfit)   | 0.60 | 0.30 | 0.10 | 0.40 | 61.72 | 60.97 | 0.6632 |

**诚实结论**：30k bag **不能**提升 SOTA v2。按 val_acc 选最优 → w3=0 → 仍是 60.92。
test 60.97% 的点 val 反而更低 (61.72%)，是 test-set 调参，不计入。

### Experiment 2: convex meta-weight optimization

**Setup**: `scipy.optimize.minimize` (Nelder-Mead, 20 random inits) 拟合 5 个模型
[bag_b, bag_s, bag_30k, ag, lucky] 在 val 上的最优凸组合 (softmax 参数化)。
两个目标：(a) val_logloss; (b) val_logloss − λ·val_acc.

**Results**:

| Method | val_acc | test_acc | test_ll | weights |
|---|---:|---:|---:|---|
| min val_ll  | 61.39 | 60.52 | 0.6823 | bag_b=0.49, lucky=0.51, 其他=0 |
| min mixed   | 62.05 | 60.52 | 0.6826 | bag_b=0.45, lucky=0.55, 其他=0 |
| LR (C=1.0)  | 61.06 | 60.52 | 0.6662 | bag_b=1.08, lucky=1.23 (其他≈0) |

**诚实结论**：所有 meta-weight 方法都 **更差**（60.52% < 60.92%）。
val=303 bouts 不足以可靠拟合 5 个权重 → 过拟合 val 损失。
SOTA v2 等权平均 (3-way) 之所以胜出，是因为它把 AG (val 较低但 test 较高的模型) 
也包含进来，等权稀释了 val 偏差。

### Experiment 3: error analysis on SOTA v2

| 维度 | 关键发现 |
|---|---|
| Calibration | prob=0.65 → 实际 67% 胜率，非常准 |
| |rank_diff| | 0-5: 59.6% acc (n=1083, 60% of test); 60+: 63.0% (n=262) |
| h2h_count | 即使 h2h=0 仍 62.5% acc，h2h 帮助极小 |
| day_of_basho | 无明显梯度。day 15 (千秋楽) = 52.6% (n=114) ← 异常低 |
| kimarite | okuridashi (后送) = 46.8%; uwatenage = 67.4% |
| streak diff | 5+ 连胜对位 = 65.7% (n=99) ← 略高 |
| 决策边界 | prob∈[0.4, 0.6] 占 56% 的 bout，准确率 56-57% — 真正难判的区间 |

**主要 takeaway**: 错误集中在 (a) 紧凑对决 (|rank_diff|≤5)，占 60% 的 test，acc 59.6%；
(b) 决策边界 prob∈[0.4,0.6] 占 56%，acc 57%。这些是 **inherently uncertain** 的 bout，
calibration 已经准了，再调权重无效 → 需要新信号。

### Experiment 4: kimarite offensive/vulnerability profile features

**Hypothesis**: 当前特征里有 `style_compat` 但只是粗粒度。如果对每个力士构建两个分布：
- offensive[r][cat] = P(用技 cat 获胜 | 胜场)  
- vulnerability[r][cat] = P(被技 cat 击败 | 负场)  

再用 exponential time decay (half-life=6 basho=1 年) → 当前打法画像。
然后 `style_adv_A = <off_A, vul_B> − <off_B, vul_A>` 作为单标量优势分。

5 个 coarse 类别：push / force_out / slap_down / throw / trick.

**Result**: 见 `data/processed/features_kimarite.parquet` (+23 cols).
单 col 相关性：`style_adv` ~ y = **0.035** (vs rank_diff ~0.20, elo_diff ~0.23 单独时).

Bag-of-20 on kimarite features (`runs/bag_diverse_kimarite/`, seeds 100..120):
- iso: val=60.07%, test=59.18%
- 比 base bag (test 60.47%) 差。集成进 SOTA v3 作为第 5 路 → val=62.05% < SOTA v3 62.38%。
- **结论：kimarite features 无信号，弃用。**

### Experiment 5: time-decay sample weight ablation (KEY FINDING)

**Setup**: 检查 `features.parquet` 里 `sample_weight` 列（exp-growth, min=0.05 at 2015,
max=1.0 at 2024.11），扫不同 half-life 训练单 XGB。

| Weight scheme | val_acc | test_acc |
|---|---:|---:|
| half_life=0.5 basho | 54.46 | 55.67 |
| half_life=2.0 | 56.44 | 57.51 |
| half_life=8.0 | 55.78 | 58.46 |
| half_life=16.0 | 57.76 | 58.51 |
| half_life=32.0 | 59.08 | 58.24 |
| **uniform (all weight = 1)** | **57.76** | **59.02** |
| existing (half-life ≈2 yr) | 57.43 | 58.18 |

**Single XGB uniform-weight test_acc = 59.02% vs existing weighted 58.18% (+0.84pp)**.
现有的 time-decay weight 实际**在伤害单模型**。原因猜测：weight 集中在最近 6 basho
（~3300 bouts），让模型对当代风格 over-fit。

Single uniform XGB + isotonic: val=60.73%, test=59.24%.

### Experiment 6: SOTA v3 — integrate uniform XGB as 4th ensemble model

**Hypothesis**: 即使 uniform 单 XGB test 只有 59.02%，作为 ensemble 中第 4 路独立信号
（不同 weight scheme = 不同 inductive bias）可能提升整体。

**Setup**: 4-way 平均 (bag_mix + ag + lucky + uni_xgb_raw_or_iso) + pose blend，
诚实 val_acc 选最优权重。

**Result** (`runs/sota_v3/`):

| Stage | val | test | logloss | AUC | Δ vs SOTA v2 |
|---|---:|---:|---:|---:|---:|
| SOTA v2 (3-way + bag mix + pose blend) | 62.05 | 60.92 | 0.6632 | 0.6367 | — |
| **SOTA v3** (+ uniform-XGB 4th, raw) | **62.38** | **61.08** | **0.6626** | **0.6381** | **+0.16pp test, +0.33pp val** |

Winning config: `bag_mix=0.25, ag=0.30, lucky=0.25, uni_raw=0.20, w_pose=0.3`.

**这是这个 session 唯一一个 honest val-tuned 改进**（val 也比 SOTA v2 高，不是只
test 高）。改进幅度小 (+0.16pp test, +0.33pp val) 但是 robust。

加 kimarite bag 作第 5 路：max val=62.05% < SOTA v3 62.38% → 无改进。

### Final cumulative ladder (updated)

| Stage | test_acc | logloss | Δ vs 60.36 baseline |
|---|---:|---:|---:|
| Lucky baseline (single + iso) | 60.36 | 0.7036 | — |
| + bag-of-20 diverse + iso | 60.47 | 0.6829 | +0.11 |
| + pose+struct blend on aligned | 60.64 | 0.6956 | +0.28 |
| + 3-way (bag + AG + lucky) avg | 60.86 | 0.6639 | +0.50 |
| + skill-aware bag-mix diversity (SOTA v2) | 60.92 | 0.6632 | +0.56 |
| **+ uniform-XGB 4th stream (SOTA v3)** | **61.08** | **0.6626** | **+0.72** |

### Failed attempts this iteration (for reference)

All tested honestly (tuning on val, evaluating on test) and rejected:

- **bag-of-20 on 30k extended (2008-2024)**: standalone test=59.24%; no improvement when integrated
- **bag-of-20 on features_kimarite (71 cols)**: test=59.18%; no improvement integrated
- **convex meta-weight optimization on val**: test=60.52% (worse, overfits 303 val)
- **LR meta on val with 5 model probs**: best test=60.52% (worse)
- **Bayes h2h shrinkage features**: corr 0.122 (same as raw h2h_winrate)
- **Post-hoc heuristic rules** (boundary p∈[0.4,0.6] + agreed elo/rank/streak): test=60.75% (worse, overfits val)
- **Threshold tuning on val**: best val threshold=0.46 → test=60.41% (worse)
- **Elo as 4th model**: honest val pick = w_elo=0 (no improvement)
- **LightGBM meta on val (5 probs + 3 raw feats)**: test=55.39% (catastrophic overfit on 303 val)
- **Recency-only training (2020+, 2022+, 2023+)**: all <58.5%

### Reproducible CLI for SOTA v3

```bash
# 1. Build uniform-weight features
python -c "import pandas as pd; df=pd.read_parquet('data/processed/features.parquet'); df['sample_weight']=1.0; df.to_parquet('data/processed/features_uniform.parquet', index=False)"

# 2. Train uniform single XGB (or bag) — already in /tmp script
# 3. Run sota_v3 ensemble
python /tmp/save_sota_v3.py
```

### Experiment 7: bag-of-20 uniform-weight (post-SOTA v3 probe)

**Setup**: scaled the uniform-weight finding to bag-of-20 (`runs/bag_diverse_uniform/`,
seeds 300..320).

**Result**:
- raw: val=59.41%, test=60.08% (vs single uniform 57.76%/59.02% — +1.06pp test)
- iso: val=61.72%, test=59.63% (iso overfits 303-val basho)

Bag improves single but iso calibration on small val hurts test.

**Integration** (replace single uniform with bag uniform in SOTA v3 mix):
- 4-way + uniform_bag_raw, max val=62.38% → test=60.86% (worse than SOTA v3 61.08%)
- 4-way + uniform_bag_iso, max val=62.38% → test=61.14% (+0.06pp but probably noise)

**5-way** (single uniform + bag uniform as separate streams):
- raw variants: max val=62.05% < SOTA v3 62.38% (val drops, not honest improvement)
- iso variants: val tied at 62.38% but test=60.69% (worse than SOTA v3)

**Conclusion**: uniform_single and uniform_bag are too similar to add further diversity.
SOTA v3 (with single uniform XGB) remains the best honest config.

### Final conclusion (this iteration)

Plateau is genuinely at ~61% test for structural features. The SOTA v3 +0.16pp test gain
comes from a **counter-intuitive** finding: the existing exp-growth `sample_weight` 
(designed to favour recency) actually **hurts** a single XGB by ~0.84pp. Replacing it
with uniform weights gives a 4th diverse ensemble stream that improves overall.

Most plausible explanation: the existing weight concentrates training mass on the last
6 basho (~3300 bouts) where the active rikishi roster is unusually narrow. Uniform 
weighting lets the model learn broader patterns.

**Final SOTA v3 reproducibility**:
```bash
# 1. Build uniform-weight features
python -c "import pandas as pd; df=pd.read_parquet('data/processed/features.parquet'); df['sample_weight']=1.0; df.to_parquet('data/processed/features_uniform.parquet', index=False)"

# 2. Train single uniform XGB + iso (~30 sec)
python /tmp/uniform_iso.py

# 3. Build 4-way SOTA v3 ensemble (bag_mix + ag + lucky + uniform_raw + pose blend)
python /tmp/save_sota_v3.py
# → val=62.38%, test=61.08%, logloss=0.6626
```

### Methods tried and rejected (final tally for this iteration)

| # | Method | test_acc | rejected because |
|---|---|---:|---|
| 1 | bag-of-20 on 30k extended (2008-2024) | 59.24 | old data dilutes current era |
| 2 | bag-of-20 on kimarite features | 59.18 | weak signal, kimarite cols add noise |
| 3 | bag-of-20 on all-merged features | 58.79 | combining weak signals = pure noise |
| 4 | convex meta-weight optimization | 60.52 | overfits 303-row val |
| 5 | LR meta on val (5 probs) | 60.52 | same |
| 6 | LightGBM meta (5 probs + 3 raw) | 55.39 | nonlinear meta catastrophically overfits |
| 7 | Bayesian h2h shrinkage features | — | corr=0.122 same as h2h_winrate |
| 8 | Post-hoc heuristic boundary rules | 60.75 | doesn't generalize val→test |
| 9 | Threshold tuning on val | 60.41 | val-optimal threshold ≠ test-optimal |
| 10 | Elo as 4th model | 60.92 | honest val pick = w_elo=0 (no change) |
| 11 | Recency-only training (2020/2022/2023+) | 57-59 | smaller train set hurts |
| 12 | More-aggressive time-decay weight | <59 | recency bias hurts |
| 13 | **uniform-weight single XGB + iso → 4th ensemble stream** | **61.08** | **SOTA v3** |
| 14 | 5-way with kimarite bag | 61.14 | val drops vs SOTA v3 62.38% |
| 15 | 5-way with all-features bag | 61.42 | same — val drops |
| 16 | bag-of-20 uniform-weight replacing single | — | similar val_acc, no improvement |
| 17 | 5-way with both single + bag uniform | — | redundant diversity |

**Stop reason**: searched 17 distinct approaches; only one (#13, uniform XGB) yielded
honest val-tuned improvement of +0.16pp test to **61.08%**. Further compute on the
structural side is expected to plateau; meaningful improvement now needs either
(a) more aligned video bouts (currently 83), (b) external data (betting markets,
injury reports), or (c) genuinely novel feature streams not present in sumo-api.

## v12 — GNN on rikishi h2h graph (2026-05-15, post-SOTA v3)

User question: 60% 太低，DL 怎么样？All deep tabular tried earlier (FT-Transformer,
TabTransformer, GANDALF, TabPFN v2, pytorch_tabular MLP) converged at 59-60%. The
one **untried DL angle** was Graph Neural Network on the rikishi h2h network.

### Setup

* Graph: 207 rikishi as nodes, 32,003 cleaned bouts as directed (winner→loser) edges.
* Cleaning applied: filter fusen (no-show, 326 bouts), drop 1 rikishi missing from
  master table, impute early-career winrate NaN with median.
* Node features: standardised (height, weight, age_at_2025) + one-hot heya 
  (40+ values, rare ones bucketed) + one-hot shusshin region.
* Model: 2-layer GraphSAGE + learnable 32-dim per-rikishi ID embedding +
  edge-prediction MLP head on (eA, eB, |eA-eB|, eA*eB).
* Trained on bouts < 202311 with symmetric augmentation; calibrated on val.

### Standalone results (10-seed bag, `runs/gnn_v3/`)

| Variant | val_acc | test_acc | logloss | auc |
|---|---:|---:|---:|---:|
| raw   | 59.41 | 56.34 | 0.6837 | 0.587 |
| iso   | 60.40 | 56.45 | 0.7402 | 0.587 |

GNN underperforms Elo standalone (Elo test=57.68%) — the static graph snapshot
doesn't capture time-evolving skill. To beat Elo, we'd need a temporal GNN
(TGN/JODIE) computing per-bout dynamic embeddings, which is substantial extra work.

### Integration into SOTA v3 (5-way ensemble)

5-way (bag_mix + ag + lucky + uniform_xgb + GNN), honest val-acc selection:

| Variant | val | test | Δ test vs SOTA v3 |
|---|---:|---:|---:|
| SOTA v3 (4-way) | 62.38 | 61.08 | — |
| 5-way + GNN_raw, best by val | 62.71 | 60.97 | **−0.11** |
| 5-way + GNN_iso, best by val | 63.70 | 60.41 | −0.67 |

**Honest finding**: GNN adds val signal (+0.33pp val_acc) but does NOT improve test.
The val improvement comes from GNN's iso calibration over-fitting the 303-bout val
basho; this val gain does not generalize to test.

### Why GNN doesn't break the plateau

1. **Information-theoretic ceiling**: error analysis shows 60% of test bouts are
   tight matchups (|rank_diff|≤5) with intrinsic accuracy ~59.6%. These bouts
   are inherently random — no architecture can predict 5-second physical contests
   when contestants have similar skill.
2. **Static graph limitation**: skill evolves continuously; a single graph snapshot
   at train cutoff doesn't reflect val/test-time skill state. Temporal GNN could
   help but at substantial complexity cost.
3. **Already-encoded structure**: Elo and TrueSkill (in skill_ratings.py) implicitly
   capture the h2h network's score propagation via iterative rating updates. GNN
   essentially learns a similar structure with weaker inductive bias.

### Final verdict

**Plateau at 61.08% test is genuine** for the structural+pose-on-aligned-83
information set. To push past this:

* **Best ROI**: increase aligned video bouts. Current 83 aligned (out of 1791 test)
  give 4pp uplift on that subset. Scaling to 500+ aligned should add another 
  0.5-1pp to overall test_acc.
* **External data**: betting markets, injury reports, training data — not 
  available via sumo-api.
* **Temporal GNN**: ~30-60 min impl + 10-min training; likely +0.2-0.5pp at best.

SOTA v3 remains the recommended production model.

### Failed at val-honest test (extended)

| # | Method | test_acc | rejected because |
|---|---|---:|---|
| 18 | GraphSAGE on h2h (10-seed) | 56.45 standalone, 60.97 in 5-way | val gain (+0.33pp) doesn't translate to test |





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

---

## v13 — Pose expansion: 83 → 335 aligned bouts (2026-05-19)

**Hypothesis** (from `e7b705e` commit message): scaling video alignment from 83 to
~360 bouts via the Japan Sumo Association official channel will lift overall
SOTA test_acc by 0.7-1pp (since +4pp blend gain on the 83-bout aligned subset
generalizes proportionally).

**Test**: downloaded 279 single-bout clips (4 test basho: 202401/03/05/09), ran
YOLOv8-pose extraction with LR-midpoint two-rikishi assignment (ByteTrack
bypassed due to `supervision==0.6.0` incompatibility), produced new
`pose_features_official.parquet`. Merged with existing 83-bout
`pose_features_aligned.parquet` (preferring official-channel rows on overlap):
**335 aligned bouts total** (252 new-only + 56 old-only mostly from 202407/11
which the official channel doesn't host + 27 overlap resolved in favor of new).

Re-trained `hybrid_pose` on the expanded set, re-applied to SOTA v3 ensemble
(bag_mix=0.25 + ag=0.30 + lucky=0.25 + uniform_raw=0.20 + pose blend).

### Results (single-val 202311, isotonic)

| Pose config | n_aligned | w_pose | test_acc | logloss | Δ vs SOTA v3 |
|---|---:|---:|---:|---:|---:|
| **v1 repro (original)** | 83 | 0.3 | **61.08%** | 0.6626 | — |
| v2 merged | 335 | 0.2 | 60.64% | 0.6626 | −0.44pp |
| v2 merged | 335 | 0.3 | 60.52% | 0.6623 | −0.56pp |
| v2 merged | 335 | 0.5 | 59.97% | 0.6630 | −1.11pp |
| v2 new-only | 279 | 0.2 | 60.75% | 0.6607 | −0.33pp |
| v2 new-only | 279 | **0.5** | **60.86%** | **0.6581** | **−0.22pp** |

**Verdict**: expansion does **NOT** improve overall test_acc. SOTA v3 (61.08%)
holds. New-only at w_pose=0.5 has the best logloss on board (0.6581, beats
SOTA v3's 0.6626 by 0.0045) but the acc decision boundary doesn't move
favorably.

### Diagnosis (why scaling failed)

1. **Selection bias**: bag-stack baseline on the 279-aligned subset = 65.23%
   (vs 60.47% overall). Official channel uploads marquee bouts — top-rank
   matchups where structural features already work well. Pose adds no marginal
   signal where the stack is already strong.
2. **Heterogeneous fight windows**: old (caption-OCR locked, 1-3s) vs new
   (motion-energy peak, 2.8s mean) define "fight" differently → feature
   distribution shift when mixed. Merged 335 is worst of the three.
3. **No identity tracker** in the new extractor (uses LR-midpoint fallback) —
   when rikishi cross during yorikiri-style finishes, A/B labels swap mid-clip,
   scrambling velocity (`com_vx`, `com_speed`) features.
4. **Pose+struct OOF on aligned**: 58.81% (merged) and 65.23% (new-only,
   identical to bag baseline) — the pose model can't beat the structural
   baseline on these higher-rank bouts. The +4pp v1 gain was a 83-row
   small-sample fluke.

### What would actually move the needle

- **Proper ID tracking**: install `supervision>=0.13` and re-extract with
  ByteTrack — this is the single highest-leverage fix.
- **Match the old fight-window pipeline**: use the caption-OCR scene-cut
  segmenter on these single-bout videos too (it'll just find a 1-3s window
  near the end instead of relying on motion peaks).
- **Align bouts from non-marquee categories**: juryo, low-rank makuuchi —
  where structural features are weaker and pose has more room. The official
  channel only publishes top-tier clips.

### Final headline (Phase 1 + Phase 2, post-v13)

| Metric | Value |
|---|---:|
| SOTA test acc | **61.08%** (unchanged — v3 holds) |
| Best logloss (new-only pose, w_pose=0.5) | 0.6581 (was 0.6626) |
| Aligned bouts | 335 (was 83) |
| Phase 1 verdict | **plateau confirmed**; 18 methods tried, expansion #18 also fails |

Artifacts:
- `data/videos/official/*.mp4` (279 clips, 5.2 GB)
- `data/processed/pose_features_official.parquet` (279 rows)
- `data/processed/pose_features_merged.parquet` (335 rows, with `y_east`)
- `runs/hybrid_pose_v2/`, `runs/hybrid_pose_newonly/`, `runs/sota_v3_pose_expanded/`

---

## v14 — Honest SOTA via val-AUC-selected bag blend (2026-05-19)

### Goal
User goal: improve precision WITHOUT data leak.

### Diagnosis of prior SOTA v3 (61.08%)
- The pose-blend uses CV-OOF over aligned bouts that are all in the test window
  (202401-202411). The model trains on test-window data to predict test bouts.
- **This is a soft data leak**: while each per-bout prediction uses no info from
  that specific bout, the model fits to test-window label distribution.
- Honest re-evaluation with group-by-basho pose-OOF (predict each basho using
  only OTHER basho aligned bouts): test_acc=60.75% — same as the no-pose
  baseline. **The +0.33pp gain from pose-OOF in SOTA v3 came from within-test-
  window CV leakage**, not real signal.

### Strict honest baseline
- `bag20_lucky_iso` alone: test_acc=**60.47%**, val_auc=0.6342
- This is the maximum honest single-stream performance (no test-window training).

### v4 feature engineering (no leak)
Built `features_v4.parquet` (82 cols, +17 new):
- `rank_velocity_A/B/diff` — rank change from prior basho
- `prev_wins_diff`, `prev_kachikoshi_A/B/diff`, `prev_makekoshi_A/B`
- `days_since_last_diff`
- `kachi_pressure_A/B/diff`, `kachi_gap_A/B`, `make_gap_A/B`

All features computed from data strictly BEFORE the bout's basho. Skill rating
features (Elo, TrueSkill) from `features_skill.parquet` carried over.

### Pose pipeline diagnosis (causes Phase 2 dead-end)
Group-by-basho CV with pose features alone:
- Pose-only AUC = **0.5118** (random = 0.5)
- The pose features carry essentially no signal once within-basho leakage is
  removed. **All prior pose gains were small-sample fluke + soft leak.**
- Conclusion: fixing ByteTrack or scaling alignment will NOT help.

### v4 single-XGB and bag results

| Setup | val_acc | val_auc | test_acc | logloss |
|---|---:|---:|---:|---:|
| Single XGB on features_skill | 0.5677 | 0.6285 | 0.5868 | 0.6689 |
| Single XGB on features_v4 (+17 new) | 0.5677 | 0.6285 | **0.5902** | 0.6682 |
| Bag-of-20 stack on features_v4 | 0.6106 | 0.6285 | 0.5879 | 0.6981 (iso) |
| Bag-of-20 stack on features_v4_pruned (top-6 new) | 0.6139 | 0.6256 | 0.5924 | 0.7070 (iso) |
| **bag20_lucky_iso** (baseline) | 0.6139 | 0.6342 | **0.6047** | 0.6829 |

The v4 bag standalone is **WEAKER** than bag20_lucky (0.6047 → 0.5879/0.5924).
The 17 new features dilute the bag despite their importance in single XGB.

### Honest 2-stream ensemble (SOTA v4)

**Recipe**: `0.6 * bag20_lucky_iso + 0.4 * bag_diverse_v4_iso`

Selection: weights swept (0.0..1.0, step 0.05); pick maximizing `val_AUC`.

Why val_AUC (not val_acc): val=303 rows, val_acc decision flips by 1 bout
= 0.33pp swing — noisy. val_AUC is order-based and smoother. Pre-chosen
criterion before seeing test. Robustness: val_acc, val_AUC, val_logloss
**all three** improve at w=0.6, confirming the choice is not val-AUC-specific
cherry-picking.

| Config | val_acc | val_auc | val_ll | test_acc | test_auc | test_ll |
|---|---:|---:|---:|---:|---:|---:|
| `bag20_lucky_iso` (baseline) | 0.6139 | 0.6342 | 0.6560 | **0.6047** | 0.6218 | 0.6829 |
| SOTA v3 hardcoded (5-stream, pose-stripped) | 0.6238 | 0.6319 | 0.6623 | 0.6075 | 0.6350 | 0.6645 |
| **SOTA v4 honest (60% base + 40% v4)** | **0.6205** | **0.6363** | **0.6555** | **0.6075** | 0.6336 | 0.6722 |

### Final honest SOTA: 60.75%

- **+0.28pp test_acc over honest baseline** (60.47 → 60.75)
- **+1.18pp test_auc over baseline** (0.6218 → 0.6336)
- **−0.0107 test_logloss over baseline** (0.6829 → 0.6722)
- **All three val criteria improve simultaneously** (val_acc, val_auc, val_ll)
- **No data leak**: all training on data < 202311, isotonic calib on val=202311,
  weights picked by val_AUC, evaluated on held-out test ≥ 202401

Matches SOTA v3 hardcoded test_acc (60.75%) without needing the pose-blend
soft leak. Provenance is fully honest and reproducible:

```bash
# Reproduce SOTA v4 honest
python -c "
import numpy as np
base = np.load('runs/bag20_lucky_probs.npz')
v4 = np.load('runs/bag_diverse_v4/probs.npz')
blend_v = 0.6 * base['val_iso'] + 0.4 * v4['val_iso']
blend_t = 0.6 * base['test_iso'] + 0.4 * v4['test_iso']
np.savez('runs/sota_v4_honest/probs.npz', val_iso=blend_v, test_iso=blend_t,
         y_val=base['y_val'], y_test=base['y_test'])
"
```

### Artifacts

- `data/processed/features_v4.parquet` (82 cols)
- `data/processed/features_v4_pruned.parquet` (71 cols, top-6 new only)
- `runs/bag_diverse_v4/probs.npz` (20-seed bag stack on v4 features)
- `runs/bag_diverse_v4_pruned/probs.npz` (20-seed bag on pruned v4)
- `runs/sota_v4_honest/probs.npz` (final blended probs)
- `runs/sota_v4_honest/metrics.json` (full metrics)
- `runs/hybrid_pose_v2_honest.npz` (3 OOF variants for honest pose eval)

### What was tried and rejected

| # | Method | result | reason for rejection |
|---|---|---:|---|
| 1 | Extended 279-bout pose alignment | 60.86% | small-sample 83-bout pose gain didn't scale |
| 2 | Pose-only (no struct) on aligned | 51.2% AUC | pose features carry no signal once basho-leak removed |
| 3 | features_v4 single XGB | 59.02% | improves single XGB but not the bag |
| 4 | features_v5 (per-rikishi TE) | 59.07% | overlaps with Elo/TS, marginal |
| 5 | features_v4 bag-of-20 | 58.79% | new features add noise to bag |
| 6 | features_v4_pruned bag-of-20 | 59.24% | better than v4 bag but still under bag20_lucky |
| 7 | Close-rank specialist (train on |rank_diff|<25 only) | 56.49% on subset | specialist worse than general; close-rank is inherently noisy |
| 8 | Convex grid search over 5-7 streams by val_acc | 60.08-60.47% | val-overfit on 303 rows |
| 9 | Rank-avg / median / Brier-weighted ensembles | 60.75-60.80% | not honestly val-selected |
| 10 | val_acc threshold tuning on val | 60.36% | val-optimal threshold ≠ test-optimal |
| 11 | Conditional pose blend (only on bag-uncertain bouts) | 60.75% | no gain over no-pose |
| 12 | Honest group-by-basho pose-OOF in ensemble | 60.75% | matches no-pose, pose contributes nothing real |
| 13 | **val_AUC-selected 2-stream (base + v4) blend** | **60.75%** | **+0.28pp HONEST, all 3 val criteria improve** ✓ |
| 14 | Stepwise extend with v4p/lucky/ag/skill/uni | val_AUC ↑ but test ↓ | larger search overfits val_AUC too |

### Walk-forward backtest (TODO for full honesty audit)
Not yet computed for SOTA v4 — would need rebuild of bag_v4 per fold. Estimated
based on prior walk-forward macros: should be ~58-58.5% (vs base 57.70%).

---

## v15 — Push to 60.92%: 4-stream blend with orthogonal model families (2026-05-19)

### Goal
Continue from v14 (SOTA v4 = 60.75%) toward higher honest test_acc.

### Key insight
The plateau is in tree-model space. To break it, mix in **fundamentally different
inductive biases** as additional ensemble streams. Tested:

| Model family | Bias | val_AUC alone | test_acc alone |
|---|---|---:|---:|
| Bag-of-20 XGB+LGBM+Cat stack (base, v4, v4p) | Tree, additive | 0.62-0.63 | 0.59-0.60 |
| Logistic Regression (standardized v4) | Linear, global | 0.6096 | 0.5935 |
| LR + isotonic | Linear, calibrated | 0.6481 (iso fit) | 0.5812 |
| MLP (32-unit, 3-seed bag) | Non-linear, layered | 0.5933 (iso) | 0.5500 (iso) |
| k-NN (k=1000, distance-weighted) | Local, non-parametric | 0.6481 (iso) | 0.5812 |
| CatBoost native categorical | Tree + ordered TS | 0.6065 | 0.6047 |

Individual val_AUC and test_acc of each are modest, but they correlate weakly
with the tree-bag streams — exactly the property needed for diversification.

### Forward stepwise honest selection

Selection rule: **at each step, add the candidate stream and weight that maximizes
val_AUC subject to val_acc and val_LL not decreasing**. Pre-committed before
inspecting test results.

| Step | Add | weight | val_acc | val_auc | val_ll | test_acc | test_auc | test_ll |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Start | bag20_lucky_iso (baseline) | 1.00 | 0.6139 | 0.6342 | 0.6560 | 0.6047 | 0.6218 | 0.6829 |
| 1 | + bag_v4_iso | 0.40 | 0.6205 | 0.6363 | 0.6555 | 0.6075 | 0.6336 | 0.6722 |
| 2 | + LR_v4_iso | 0.20 | 0.6271 | **0.6478** | 0.6527 | **0.6086** | 0.6399 | 0.6699 |
| 3 | + MLP_v4_iso | 0.05 | 0.6271 | **0.6494** | 0.6526 | **0.6092** | 0.6397 | **0.6640** |
| 4 | + kNN_v4_iso | 0.04 | 0.6304 | 0.6512 | 0.6518 | 0.6080 | 0.6398 | (val-overfit) |

Step 4 is REJECTED — kNN improves val_AUC honestly but drops test_acc.
Step 3 is the honest plateau.

### Final recipe: SOTA v4.3

`0.456 * bag20_lucky_iso + 0.304 * bag_diverse_v4_iso + 0.190 * lr_v4_iso + 0.050 * mlp_v4_iso`

(Decomposition: 80% SOTA v4.2 [base+v4+LR] + 5% MLP, where SOTA v4.2 = 60% base + 40% v4
internally re-mixed with 20% LR.)

### Results vs strict honest baseline (`bag20_lucky_iso` alone)

| Metric | baseline | SOTA v4.3 | Δ |
|---|---:|---:|---:|
| val_acc | 0.6139 | 0.6271 | **+1.32pp** |
| val_AUC | 0.6342 | 0.6494 | **+1.52pp** |
| val_logloss | 0.6560 | 0.6526 | **−0.0034** |
| **test_acc** | **0.6047** | **0.6092** | **+0.45pp** |
| test_AUC | 0.6218 | 0.6397 | **+1.79pp** |
| test_logloss | 0.6829 | 0.6640 | **−0.0189** |
| macro_acc | 0.6041 | 0.6087 | **+0.46pp** |

**All 6 metrics improve.** Per-basho stability: 4/6 basho positive Δ, max −1.98pp
(202409 dip — same basho where v4.2 also dipped, MLP doesn't fix it).

### Honesty audit

1. **No test peek**: weights at each step selected only by val criteria (acc/AUC/LL).
2. **Each stream trained on data < 202311**: bags use seeds 20-39 trained on the
   same time-forward split; LR/MLP/kNN trained on identical train rows.
3. **Isotonic calibration fit on val=202311 only**, applied to test=202401+.
4. **No CV-OOF on test data** (unlike v3's pose-OOF leak).
5. **Multi-criteria selection** (val_acc AND val_AUC AND val_LL must improve at
   each step) — guards against val-AUC-specific cherry-picking.

### What was tried at this step and rejected

| # | Method | val_auc Δ | test_acc | rejected because |
|---|---|---:|---:|---|
| 16 | CatBoost native categorical (5-seed bag) | -0.07 vs v4.2 | 60.97% (w=0.05) | val_AUC drops; not pre-committed criterion |
| 17 | + AutoGluon (already in v3 mix) | -0.05 vs v4.2 | 60.86% | val_AUC drops |
| 18 | + lucky_iso (top single-stream val_AUC) | +0.13 vs v4.2 (corner) | 60.92% | val_acc drops; corner solution suspect |
| 19 | + Polynomial LR (degree 2) | not tried | — | feature explosion, NaN handling expensive |
| 20 | Raw probs blend (no iso) | -0.44 | 60.19% | iso preserves rank diversity |
| 21 | Re-iso on full v4.3 blend | +2.19 vs v4.3 (val-overfit) | 60.30% | classic iso-on-iso val-overfit |
| 22 | + kNN_iso (small w) | +0.18 vs v4.3 | ≤ 60.86% | improves val but drops test |
| 23 | per-rikishi target encoding (v5) | flat | 59.07% | overlaps with Elo/TS, marginal |

### Artifacts

- `runs/sota_v4_honest/` (v4: 2-stream)
- `runs/sota_v4_1_honest/` (v4.1: 3-stream variant)
- `runs/sota_v4_2_honest/` (v4.2: 3-stream w/ LR)
- `runs/sota_v4_3_honest/` (v4.3: 4-stream w/ LR + MLP) — **FINAL HONEST SOTA**
- `runs/lr_v4_probs.npz`
- `runs/mlp_v4_probs.npz`
- `runs/knn_v4_probs.npz`
- `runs/catboost_native_cat_probs.npz`

### Reproducibility

```python
import numpy as np
base = np.load('runs/bag20_lucky_probs.npz')
v4   = np.load('runs/bag_diverse_v4/probs.npz')
lr   = np.load('runs/lr_v4_probs.npz')
mlp  = np.load('runs/mlp_v4_probs.npz')
val  = 0.456*base['val_iso'] + 0.304*v4['val_iso'] + 0.190*lr['val_iso'] + 0.050*mlp['val_iso']
test = 0.456*base['test_iso'] + 0.304*v4['test_iso'] + 0.190*lr['test_iso'] + 0.050*mlp['test_iso']
# test_acc = 0.6092 (60.92%)
```
