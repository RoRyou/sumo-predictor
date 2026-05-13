"""Train Route A (structural / tabular) ensemble for sumo bout prediction.

Pipeline (incorporates Tricks T1-T8 from the readme):
    1. Load features.parquet
    2. Time-ordered split by basho:
           train = basho < val_basho
           val   = basho == val_basho
           test  = basho > val_basho
    3. KFold target-encoding (T1) on heya / shusshin
    4. Train XGBoost + LightGBM + CatBoost as base models (T4)
       - each uses sample_weight from features (T3)
    5. Stack via Logistic Regression on 5-fold OOF probs (T4)
    6. Calibrate the stacked model (Platt + Isotonic) (T5)
    7. Report metrics on val/test + per-rank-tier breakdown

CLI::

    python -m src.training.train_struct run \\
        --features data/processed/features.parquet \\
        --val-basho 202311 --test-start 202401 \\
        --out-dir runs/struct_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Feature column selection
# ---------------------------------------------------------------------- #
ID_COLS = ["bashoId", "day", "matchNo", "eastId", "westId", "kimarite"]
LABEL_COL = "y"
WEIGHT_COL = "sample_weight"
CATEGORICAL_COLS = ["heya_A", "heya_B", "shusshin_A", "shusshin_B"]


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns (everything except ids/label/weight/categoricals)."""
    drop = set(ID_COLS + [LABEL_COL, WEIGHT_COL] + CATEGORICAL_COLS)
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------- #
# Trick T1: KFold target encoding
# ---------------------------------------------------------------------- #
class KFoldTargetEncoder:
    """KFold target encoder with global-mean fallback.

    No leakage: training rows are encoded with the mean from the *other*
    folds; test rows are encoded with the full-training mean.
    """

    def __init__(self, cols: list[str], n_splits: int = 5, smoothing: float = 10.0,
                 random_state: int = 42) -> None:
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean_: float | None = None
        self.maps_: dict[str, dict[str, float]] = {}

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        X = X.copy()
        self.global_mean_ = float(np.mean(y))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for col in self.cols:
            if col not in X.columns:
                continue
            enc = np.full(len(X), self.global_mean_, dtype=float)
            for tr_idx, va_idx in kf.split(X):
                tr_x, tr_y = X.iloc[tr_idx][col].astype("string"), y[tr_idx]
                stats = pd.DataFrame({"x": tr_x, "y": tr_y}).groupby("x")["y"].agg(["mean", "count"])
                w = stats["count"] / (stats["count"] + self.smoothing)
                smoothed = w * stats["mean"] + (1 - w) * self.global_mean_
                va_vals = X.iloc[va_idx][col].astype("string").map(smoothed)
                enc[va_idx] = va_vals.fillna(self.global_mean_).to_numpy()
            X[f"te__{col}"] = enc
            # full-data map for transform()
            stats = pd.DataFrame({"x": X[col].astype("string"), "y": y}).groupby("x")["y"].agg(["mean", "count"])
            w = stats["count"] / (stats["count"] + self.smoothing)
            smoothed = w * stats["mean"] + (1 - w) * self.global_mean_
            self.maps_[col] = smoothed.to_dict()
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            if col not in X.columns or self.global_mean_ is None:
                continue
            X[f"te__{col}"] = X[col].astype("string").map(self.maps_.get(col, {})).fillna(self.global_mean_).to_numpy()
        return X


# ---------------------------------------------------------------------- #
# Split helpers
# ---------------------------------------------------------------------- #
@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_basho: str
    test_start: str

    def __repr__(self) -> str:
        return (
            f"Split(train={len(self.train):,}, val={len(self.val):,}, "
            f"test={len(self.test):,}, val_basho={self.val_basho}, "
            f"test_start={self.test_start})"
        )


def time_split(df: pd.DataFrame, val_basho: str, test_start: str) -> Split:
    df = df.copy()
    df["bashoId"] = df["bashoId"].astype(str)
    train = df[df["bashoId"] < val_basho].reset_index(drop=True)
    val = df[df["bashoId"] == val_basho].reset_index(drop=True)
    test = df[df["bashoId"] >= test_start].reset_index(drop=True)
    return Split(train=train, val=val, test=test, val_basho=val_basho, test_start=test_start)


# ---------------------------------------------------------------------- #
# Models
# ---------------------------------------------------------------------- #
def make_xgb(params: dict[str, Any] | None = None):
    import xgboost as xgb

    base = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.05,
        max_depth=5,
        n_estimators=500,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    base.update(params or {})
    return xgb.XGBClassifier(**base)


# ---------------------------------------------------------------------- #
# T8 — Focal-loss XGBoost (custom objective)
# ---------------------------------------------------------------------- #
def _focal_obj_factory(gamma: float = 2.0, alpha: float = 0.5):
    """Return (grad, hess) function suitable for XGBoost custom obj.

    Implements focal loss for binary classification with the standard
    derivation:  L = -alpha * (1-p)^gamma * y log p
                    -(1-alpha) * p^gamma * (1-y) log(1-p)
    where p = sigmoid(z).  We compute first/second derivatives wrt z.
    """

    def _obj(preds, dtrain):  # preds is the raw margin (xgb pre-sigmoid)
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        eps = 1e-7
        p = np.clip(p, eps, 1 - eps)
        # alpha-weight per class
        a = np.where(y == 1, alpha, 1 - alpha)
        # focal modulation
        # dL/dp for positive: -alpha * (1-p)^gamma / p + alpha*gamma*(1-p)^(gamma-1) * log p
        # use chain rule via z (z = logit): dp/dz = p(1-p)
        # combined gradient/hessian (standard focal-loss derivation):
        pt = np.where(y == 1, p, 1 - p)
        # gradient wrt z
        # -(1-pt)^gamma * (y - p) approximation hides alpha; do the full form:
        # See: Lin et al. 2017, eqs (3)-(5)
        sign = np.where(y == 1, 1.0, -1.0)
        # dL/dz = a * sign * (1-pt)^gamma * ( gamma * pt * log(pt) + pt - 1 ) * sign
        # Simplify: dL/dz = a * (1-pt)^gamma * (gamma * pt * log(pt) + pt - 1) * (-sign)
        # but easier: numerically compute via standard formula
        modulator = (1 - pt) ** gamma
        # gradient:
        grad = a * modulator * (gamma * pt * np.log(np.clip(pt, eps, 1.0)) + pt - 1) * (-sign)
        # hessian (positive approximation):
        hess = a * modulator * pt * (1 - pt) * (
            gamma * (1 - pt) * (1 - gamma * np.log(np.clip(pt, eps, 1.0)) * pt / np.clip(1 - pt, eps, 1.0))
            + 1
        )
        hess = np.clip(hess, 1e-6, None)
        return grad, hess

    return _obj


def make_xgb_focal(params: dict[str, Any] | None = None):
    """XGB Booster wrapper with custom focal-loss objective.

    Returns a thin estimator that mimics ``XGBClassifier`` interface
    (``fit`` / ``predict_proba``) so it plugs into the stacking loop.
    """
    import xgboost as xgb

    base = dict(
        learning_rate=0.05,
        max_depth=5,
        n_estimators=500,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=45,
        tree_method="hist",
        gamma_focal=2.0,
        alpha_focal=0.5,
    )
    base.update(params or {})
    gamma_focal = base.pop("gamma_focal")
    alpha_focal = base.pop("alpha_focal")
    n_estimators = base.pop("n_estimators")

    class _FocalXGB:
        def __init__(self) -> None:
            self.params = base
            self.gamma = gamma_focal
            self.alpha = alpha_focal
            self.n_estimators = n_estimators
            self.booster_ = None

        def fit(self, X, y, sample_weight=None):
            dtrain = xgb.DMatrix(np.asarray(X), label=np.asarray(y), weight=sample_weight)
            obj = _focal_obj_factory(self.gamma, self.alpha)
            xgb_params = {
                "eta": self.params.get("learning_rate", 0.05),
                "max_depth": self.params.get("max_depth", 5),
                "subsample": self.params.get("subsample", 0.85),
                "colsample_bytree": self.params.get("colsample_bytree", 0.85),
                "lambda": self.params.get("reg_lambda", 1.0),
                "tree_method": self.params.get("tree_method", "hist"),
                "nthread": self.params.get("n_jobs", -1),
                "seed": self.params.get("random_state", 45),
                "disable_default_eval_metric": 1,
                "base_score": 0.5,
            }
            self.booster_ = xgb.train(
                xgb_params, dtrain, num_boost_round=self.n_estimators, obj=obj,
            )
            return self

        def predict_proba(self, X):
            d = xgb.DMatrix(np.asarray(X))
            raw = self.booster_.predict(d, output_margin=True)
            p = 1.0 / (1.0 + np.exp(-raw))
            return np.column_stack([1 - p, p])

    return _FocalXGB()


def make_lgbm(params: dict[str, Any] | None = None):
    import lightgbm as lgb

    base = dict(
        objective="binary",
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        n_estimators=500,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=43,
        verbosity=-1,
    )
    base.update(params or {})
    return lgb.LGBMClassifier(**base)


def make_catboost(params: dict[str, Any] | None = None):
    from catboost import CatBoostClassifier

    base = dict(
        loss_function="Logloss",
        learning_rate=0.05,
        depth=6,
        iterations=500,
        l2_leaf_reg=3.0,
        random_seed=44,
        verbose=False,
        allow_writing_files=False,
    )
    base.update(params or {})
    return CatBoostClassifier(**base)


def _lazy_make_tabpfn(params=None):  # imported lazily — tabpfn is heavy
    from src.training.tabpfn_wrapper import make_tabpfn

    return make_tabpfn(params)


BASE_MODEL_BUILDERS = {
    "xgb": make_xgb,
    "lgbm": make_lgbm,
    "cat": make_catboost,
    "xgb_focal": make_xgb_focal,
    "tabpfn": _lazy_make_tabpfn,
}

# Base models that silently ignore sample_weight (so we don't pass it).
_NO_SAMPLE_WEIGHT_MODELS = {"tabpfn"}


# ---------------------------------------------------------------------- #
# Stacking ensemble (T4)
# ---------------------------------------------------------------------- #
@dataclass
class StackResult:
    base_oof: dict[str, np.ndarray]      # OOF probabilities on train
    base_val: dict[str, np.ndarray]      # val probabilities
    base_test: dict[str, np.ndarray]     # test probabilities
    meta: LogisticRegression
    val_proba: np.ndarray
    test_proba: np.ndarray


def train_stack(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    base_models: tuple[str, ...] = ("xgb", "lgbm", "cat"),
    model_params: dict[str, dict[str, Any]] | None = None,
    meta: str = "lr",  # "lr" or "xgb"
    n_splits: int = 5,
    random_state: int = 42,
) -> StackResult:
    """5-fold OOF stacking; ``meta`` selects the meta-learner."""
    base_oof: dict[str, np.ndarray] = {}
    base_val: dict[str, np.ndarray] = {}
    base_test: dict[str, np.ndarray] = {}
    model_params = model_params or {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for name in base_models:
        builder = BASE_MODEL_BUILDERS[name]
        oof = np.zeros(len(X_tr), dtype=float)
        val_acc = np.zeros(len(X_val), dtype=float)
        test_acc = np.zeros(len(X_test), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
            model = builder(model_params.get(name))
            fit_kwargs: dict[str, Any] = {}
            if name not in _NO_SAMPLE_WEIGHT_MODELS:
                fit_kwargs["sample_weight"] = w_tr[tr_idx]
            model.fit(X_tr.iloc[tr_idx], y_tr[tr_idx], **fit_kwargs)
            oof[va_idx] = model.predict_proba(X_tr.iloc[va_idx])[:, 1]
            if len(X_val):
                val_acc += model.predict_proba(X_val)[:, 1] / n_splits
            if len(X_test):
                test_acc += model.predict_proba(X_test)[:, 1] / n_splits
            logger.info(
                "  [%s] fold %d  oof_logloss=%.4f", name, fold,
                log_loss(y_tr[va_idx], np.clip(oof[va_idx], 1e-6, 1 - 1e-6)),
            )
        base_oof[name] = oof
        base_val[name] = val_acc
        base_test[name] = test_acc
        logger.info(
            "  [%s] overall OOF: logloss=%.4f acc=%.4f auc=%.4f",
            name,
            log_loss(y_tr, np.clip(oof, 1e-6, 1 - 1e-6)),
            accuracy_score(y_tr, oof > 0.5),
            roc_auc_score(y_tr, oof) if len(np.unique(y_tr)) > 1 else float("nan"),
        )

    # meta-learner
    Z_tr = np.column_stack([base_oof[n] for n in base_models])
    Z_val = np.column_stack([base_val[n] for n in base_models])
    Z_test = np.column_stack([base_test[n] for n in base_models])
    if meta == "xgb":
        import xgboost as xgb_

        meta_model = xgb_.XGBClassifier(
            objective="binary:logistic",
            max_depth=2,
            n_estimators=100,
            learning_rate=0.1,
            subsample=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=99,
        )
    else:
        meta_model = LogisticRegression(max_iter=1000, C=1.0)
    meta_model.fit(Z_tr, y_tr, sample_weight=w_tr)
    val_proba = meta_model.predict_proba(Z_val)[:, 1] if len(X_val) else np.array([])
    test_proba = meta_model.predict_proba(Z_test)[:, 1] if len(X_test) else np.array([])
    return StackResult(base_oof, base_val, base_test, meta_model, val_proba, test_proba)


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #
def report_metrics(y: np.ndarray, p: np.ndarray, name: str) -> dict[str, float]:
    if len(y) == 0:
        return {}
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, p > 0.5)),
        "logloss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "p_mean": float(p.mean()),
        "y_mean": float(y.mean()),
    }
    logger.info("[%s] acc=%.4f logloss=%.4f auc=%.4f n=%d", name, out["acc"], out["logloss"], out["auc"], out["n"])
    return out


def tier_breakdown(df: pd.DataFrame, p: np.ndarray) -> dict[str, dict[str, float]]:
    """Accuracy stratified by banzuke tier of the higher-ranked of the two."""
    if "rank_diff" not in df.columns:
        return {}
    # Tier based on absolute min rank value of A and B (we use heuristics:
    # use rank_diff sign and abs to find "high rank pair", but for now just bin
    # on abs rank_diff).
    out: dict[str, dict[str, float]] = {}
    df = df.copy()
    df["abs_rd"] = df["rank_diff"].abs()
    bins = [(-1, 50, "very-close"), (50, 200, "close"), (200, 500, "mid-gap"), (500, 10_000, "large-gap")]
    for lo, hi, name in bins:
        m = (df["abs_rd"] > lo) & (df["abs_rd"] <= hi)
        if m.sum() < 5:
            continue
        y = df.loc[m, "y"].to_numpy()
        ps = p[m.to_numpy()]
        out[name] = {
            "n": int(m.sum()),
            "acc": float(accuracy_score(y, ps > 0.5)),
        }
    return out


# ---------------------------------------------------------------------- #
# Main runner
# ---------------------------------------------------------------------- #
def run(
    features_path: Path,
    val_basho: str,
    test_start: str,
    out_dir: Path,
    augment: bool = False,
    augment_before_te: bool = True,
    add_stage: bool = True,
    add_kimarite_matchup: bool = True,
    base_models: tuple[str, ...] = ("xgb", "lgbm", "cat"),
    xgb_params_path: Path | None = None,
    lgbm_params_path: Path | None = None,
    cat_params_path: Path | None = None,
    meta: str = "lr",
    calib: str = "isotonic",  # "isotonic" | "platt" | "none"
) -> dict[str, Any]:
    from src.features.structural import (
        KimariteMatchupTable,
        add_stage_features,
        symmetric_augment,
    )

    df = pd.read_parquet(features_path)
    df["bashoId"] = df["bashoId"].astype(str)
    logger.info("Loaded %d rows from %s", len(df), features_path)
    logger.info(
        "basho range: %s..%s  east-win rate: %.4f",
        df["bashoId"].min(), df["bashoId"].max(), df[LABEL_COL].mean(),
    )

    split = time_split(df, val_basho, test_start)
    logger.info(repr(split))

    train_df = split.train
    val_df = split.val
    test_df = split.test

    # T15: Basho stage features (deterministic, no leakage)
    if add_stage:
        train_df = add_stage_features(train_df)
        val_df = add_stage_features(val_df)
        test_df = add_stage_features(test_df)
        logger.info("Added T15 stage features.")

    # T7: Kimarite matchup table — fit on TRAIN only
    if add_kimarite_matchup:
        km = KimariteMatchupTable().fit(train_df)
        train_df = km.transform(train_df)
        val_df = km.transform(val_df)
        test_df = km.transform(test_df)
        logger.info("Added T7 kimarite_matchup_wr. table size=%d", len(km.lookup_))

    # T6 (fix): symmetric augmentation BEFORE target encoding so the TE
    # sees both directions and the mirrored rows get their *own* encoded
    # categorical values (no leakage from the original direction).
    if augment and augment_before_te:
        train_df = symmetric_augment(train_df)
        logger.info("Symmetric augment BEFORE TE: train rows = %d", len(train_df))

    # T1: Target encoding (fit on train, transform val/test)
    te = KFoldTargetEncoder(CATEGORICAL_COLS)
    y_tr = train_df[LABEL_COL].to_numpy()
    X_tr_full = te.fit_transform(train_df, y_tr)
    X_val_full = te.transform(val_df)
    X_test_full = te.transform(test_df)

    # Legacy path: augment AFTER TE — kept for ablation comparison
    if augment and not augment_before_te:
        X_tr_full = symmetric_augment(X_tr_full)
        y_tr = X_tr_full[LABEL_COL].to_numpy()
        logger.info("Symmetric augment AFTER TE: train rows = %d", len(X_tr_full))

    cols = feature_cols(X_tr_full)
    logger.info("Using %d numeric features", len(cols))

    X_tr = X_tr_full[cols].fillna(-9999.0)
    X_val = X_val_full[cols].fillna(-9999.0) if len(X_val_full) else X_val_full[cols]
    X_test = X_test_full[cols].fillna(-9999.0) if len(X_test_full) else X_test_full[cols]
    w_tr = X_tr_full[WEIGHT_COL].to_numpy() if WEIGHT_COL in X_tr_full.columns else np.ones(len(X_tr))
    y_val = val_df[LABEL_COL].to_numpy() if len(val_df) else np.array([])
    y_test = test_df[LABEL_COL].to_numpy() if len(test_df) else np.array([])

    # Optuna-tuned per-base params (if available)
    model_params: dict[str, dict[str, Any]] = {}
    for name, path in (("xgb", xgb_params_path), ("lgbm", lgbm_params_path),
                       ("cat", cat_params_path)):
        if path is not None and Path(path).exists():
            with open(path) as f:
                best = json.load(f)
            model_params[name] = best
            logger.info("Loaded %s best params from %s: %s", name, path, best)

    # T4: Stack
    stack = train_stack(
        X_tr, y_tr, w_tr, X_val, X_test,
        base_models=base_models,
        model_params=model_params,
        meta=meta,
    )

    # T5: Calibration on val.  Isotonic is flexible but can over-fit a small
    # val basho (303 bouts); Platt (sigmoid) is the more robust default and
    # is what we now use unless `calib='isotonic'` is passed.
    cal_val_proba = stack.val_proba
    cal_test_proba = stack.test_proba
    if len(y_val) and len(np.unique(y_val)) > 1:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression as _LR

        if calib == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(stack.val_proba, y_val)
            cal_val_proba = iso.transform(stack.val_proba)
            if len(cal_test_proba):
                cal_test_proba = iso.transform(stack.test_proba)
        elif calib == "platt":
            # Sigmoid (Platt) — 2-parameter calibration is well-suited to
            # the tiny val (303 rows here).
            lr = _LR(C=1.0, max_iter=200)
            lr.fit(stack.val_proba.reshape(-1, 1), y_val)
            cal_val_proba = lr.predict_proba(stack.val_proba.reshape(-1, 1))[:, 1]
            if len(cal_test_proba):
                cal_test_proba = lr.predict_proba(stack.test_proba.reshape(-1, 1))[:, 1]
        # else: calib == "none", leave raw

    # ----- Report -----
    metrics: dict[str, Any] = {}
    metrics["train_oof"] = {
        name: report_metrics(y_tr, stack.base_oof[name], f"train-oof-{name}")
        for name in stack.base_oof
    }
    metrics["val_raw"] = report_metrics(y_val, stack.val_proba, "val-stacked")
    metrics["val_cal"] = report_metrics(y_val, cal_val_proba, "val-calibrated")
    metrics["test_raw"] = report_metrics(y_test, stack.test_proba, "test-stacked")
    metrics["test_cal"] = report_metrics(y_test, cal_test_proba, "test-calibrated")
    metrics["test_by_tier"] = tier_breakdown(test_df, cal_test_proba)
    # meta_coefs only meaningful for LR meta; XGB meta has feature_importances_
    if hasattr(stack.meta, "coef_"):
        metrics["meta_coefs"] = {n: float(c) for n, c in zip(base_models, stack.meta.coef_[0])}
    elif hasattr(stack.meta, "feature_importances_"):
        metrics["meta_importance"] = {
            n: float(c) for n, c in zip(base_models, stack.meta.feature_importances_)
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Wrote %s", out_dir / "metrics.json")
    return metrics


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 1:
        level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_run(args: argparse.Namespace) -> int:
    metrics = run(
        features_path=Path(args.features),
        val_basho=args.val_basho,
        test_start=args.test_start,
        out_dir=Path(args.out_dir),
        augment=args.augment,
        augment_before_te=not args.augment_after_te,
        add_stage=not args.no_stage,
        add_kimarite_matchup=not args.no_kimarite,
        base_models=tuple(args.base_models),
        xgb_params_path=Path(args.xgb_params) if args.xgb_params else None,
        lgbm_params_path=Path(args.lgbm_params) if args.lgbm_params else None,
        cat_params_path=Path(args.cat_params) if args.cat_params else None,
        meta=args.meta,
        calib=args.calib,
    )
    print(json.dumps(metrics, indent=2, default=str))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Route A stacked ensemble")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run the full training pipeline")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--val-basho", required=True, help="bashoId used for validation (e.g. 202311)")
    r.add_argument("--test-start", required=True, help="first basho of test set (e.g. 202401)")
    r.add_argument("--out-dir", default="runs/struct_v1")
    r.add_argument("--augment", action="store_true", help="apply symmetric augmentation")
    r.add_argument("--augment-after-te", action="store_true",
                   help="(ablation) apply augment AFTER target encoding [old behaviour]")
    r.add_argument("--no-stage", action="store_true", help="skip T15 stage features")
    r.add_argument("--no-kimarite", action="store_true", help="skip T7 kimarite matchup table")
    r.add_argument("--base-models", nargs="+",
                   default=["xgb", "lgbm", "cat"],
                   help="base models to stack (xgb lgbm cat xgb_focal)")
    r.add_argument("--xgb-params", default=None,
                   help="path to JSON of XGB best params (from Optuna)")
    r.add_argument("--lgbm-params", default=None,
                   help="path to JSON of LGBM best params (from Optuna)")
    r.add_argument("--cat-params", default=None,
                   help="path to JSON of CatBoost best params (from Optuna)")
    r.add_argument("--meta", choices=["lr", "xgb"], default="lr",
                   help="meta-learner type (lr = LogisticRegression, xgb = small XGB)")
    r.add_argument("--calib", choices=["isotonic", "platt", "none"], default="isotonic",
                   help="probability calibration on val set (default isotonic)")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
