"""Smoke tests for train_and_evaluate (synthetic data, no SDWPF CSV)."""

import numpy as np
import pandas as pd
import pytest

from sdwpf.constants import (
    DEFAULT_METEO_MAX_LAG,
    ERA5_EXTRA_COLS,
    POOL_TURB_ID_COL,
    SDWPF_V1_ANCHOR,
    STEPS_PER_DAY,
)
from sdwpf.features import temporal_split_train_val_test
from sdwpf.pipeline import (
    _patv_feature_description,
    build_modeling_frame,
    evaluate_on_indices,
    resolve_xgb_device_spec,
    train_and_evaluate,
    train_and_evaluate_pooled,
)

pytest.importorskip("xgboost")


def test_resolve_xgb_device_spec_accepts_cuda_ordinal():
    assert resolve_xgb_device_spec("cuda:0") == "cuda:0"
    assert resolve_xgb_device_spec("CUDA:3") == "cuda:3"
    with pytest.raises(ValueError):
        resolve_xgb_device_spec("gpu")


def _synthetic_scada_merged(n: int, rng: np.random.Generator) -> pd.DataFrame:
    base = pd.Timestamp(SDWPF_V1_ANCHOR)
    t = pd.date_range(base, periods=n, freq="10min")
    wspd = np.clip(rng.normal(8.0, 1.5, n), 0.0, None)
    patv = np.clip(wspd * 50.0 + rng.normal(0, 30.0, n), 0.0, None)
    row: dict[str, np.ndarray | pd.DatetimeIndex] = {
        "datetime": t,
        "Patv": patv,
        "Wspd": wspd,
        "Wdir": rng.uniform(0, 360, size=n),
        "Etmp": rng.normal(15.0, 3.0, n),
        "Itmp": rng.normal(20.0, 2.0, n),
    }
    for c in ERA5_EXTRA_COLS:
        row[c] = rng.normal(1.0, 0.2, n).astype(np.float64)
    return pd.DataFrame(row)


def test_train_and_evaluate_meteo_mode_produces_metrics():
    rng = np.random.default_rng(0)
    df = _synthetic_scada_merged(900, rng)
    res = train_and_evaluate(
        df,
        horizon_steps=STEPS_PER_DAY,
        train_frac=0.65,
        val_frac=0.15,
        max_lag=6,
        no_patv_now=True,
        no_patv_lags=True,
        time_meteo_only=True,
        early_stopping_rounds=10,
        xgb_device="cpu",
    )
    assert res.horizon_steps == STEPS_PER_DAY
    assert res.n_features == (
        4 + len(ERA5_EXTRA_COLS) * (1 + DEFAULT_METEO_MAX_LAG) + 3
    )
    assert res.n_rows >= 400
    assert res.xgboost_test_mae is not None and res.xgboost_test_rmse is not None
    assert res.xgboost_val_mae is not None
    assert np.isfinite(res.naive_mean_train_test_mae)
    assert len(res.top_importances) >= 1
    assert res.persistence_test_mae is None
    assert res.train_patv_pmax > 0 and res.train_patv_mean > 0
    assert np.isfinite(res.naive_test_nmae_vs_pmax_pct)
    assert res.xgboost_test_nmae_vs_pmax_pct is not None
    assert np.isfinite(res.xgboost_test_nmae_vs_pmax_pct)
    assert res.xgboost_skill_vs_naive is not None
    exp_skill = 1.0 - float(res.xgboost_test_mae) / float(res.naive_mean_train_test_mae)
    assert res.xgboost_skill_vs_naive == pytest.approx(exp_skill, rel=1e-5)
    assert res.xgboost_test_bias is not None and np.isfinite(res.xgboost_test_bias)


def test_train_and_evaluate_pooled_meteo():
    rng = np.random.default_rng(3)
    df1 = _synthetic_scada_merged(600, rng)
    df2 = df1.copy()
    df2["Patv"] = np.clip(df2["Patv"] * 0.92 + rng.normal(0, 8.0, len(df2)), 0.0, None)
    res = train_and_evaluate_pooled(
        [df1, df2],
        [1, 2],
        horizon_steps=STEPS_PER_DAY,
        train_frac=0.65,
        val_frac=0.15,
        max_lag=6,
        no_patv_now=True,
        no_patv_lags=True,
        time_meteo_only=True,
        early_stopping_rounds=10,
        xgb_device="cpu",
    )
    assert POOL_TURB_ID_COL in res.feature_cols
    assert res.feature_cols[0] == POOL_TURB_ID_COL
    exp_n, _ = build_modeling_frame(
        df1,
        horizon_steps=STEPS_PER_DAY,
        max_lag=6,
        no_patv_now=True,
        no_patv_lags=True,
        time_meteo_only=True,
    )
    assert res.n_rows == 2 * len(exp_n)
    assert res.xgboost_test_mae is not None
    assert "pooled" in res.patv_feature_mode


def test_train_and_evaluate_full_scada_path():
    rng = np.random.default_rng(1)
    df = _synthetic_scada_merged(1600, rng)
    res = train_and_evaluate(
        df,
        horizon_steps=STEPS_PER_DAY,
        train_frac=0.7,
        val_frac=None,
        max_lag=4,
        early_stopping_rounds=0,
        xgb_device="cpu",
    )
    assert res.xgboost_test_mae is not None
    assert "patv_now" in res.feature_cols
    assert res.persistence_test_mae is not None
    assert np.isfinite(res.persistence_test_mae)


def test_train_and_evaluate_matches_evaluate_on_indices():
    rng = np.random.default_rng(42)
    df = _synthetic_scada_merged(1200, rng)
    r1 = train_and_evaluate(
        df,
        horizon_steps=STEPS_PER_DAY,
        train_frac=0.65,
        val_frac=0.15,
        max_lag=6,
        no_patv_now=True,
        no_patv_lags=True,
        time_meteo_only=True,
        early_stopping_rounds=10,
        xgb_device="cpu",
    )
    feat, cols = build_modeling_frame(
        df,
        horizon_steps=STEPS_PER_DAY,
        max_lag=6,
        no_patv_now=True,
        no_patv_lags=True,
        time_meteo_only=True,
    )
    tr, va, te = temporal_split_train_val_test(len(feat), 0.65, 0.15)
    patv = _patv_feature_description(
        no_patv_now=True,
        no_patv_lags=True,
        time_meteo_only=True,
        meteo_max_lag=DEFAULT_METEO_MAX_LAG,
    )
    r2 = evaluate_on_indices(
        feat,
        cols,
        horizon_steps=STEPS_PER_DAY,
        patv_feature_mode=patv,
        train_idx=tr,
        val_idx=va,
        test_idx=te,
        early_stopping_rounds=10,
        xgb_device="cpu",
    )
    assert r1.xgboost_test_mae == pytest.approx(r2.xgboost_test_mae, rel=1e-5)
    assert r1.naive_mean_train_test_mae == pytest.approx(r2.naive_mean_train_test_mae, rel=1e-5)
    assert r1.xgboost_skill_vs_naive is not None and r2.xgboost_skill_vs_naive is not None
    assert r1.xgboost_skill_vs_naive == pytest.approx(r2.xgboost_skill_vs_naive, rel=1e-5)
    assert r1.xgboost_test_bias is not None and r2.xgboost_test_bias is not None
    assert r1.xgboost_test_bias == pytest.approx(r2.xgboost_test_bias, rel=1e-4)


def test_train_and_evaluate_return_test_predictions():
    rng = np.random.default_rng(2)
    df = _synthetic_scada_merged(900, rng)
    res = train_and_evaluate(
        df,
        horizon_steps=STEPS_PER_DAY,
        train_frac=0.65,
        val_frac=0.15,
        max_lag=6,
        no_patv_now=True,
        no_patv_lags=True,
        time_meteo_only=True,
        early_stopping_rounds=5,
        return_test_predictions=True,
        xgb_device="cpu",
    )
    assert res.test_datetime is not None
    assert res.y_test is not None
    assert res.xgboost_pred_test is not None
    assert len(res.test_datetime) == len(res.y_test) == len(res.xgboost_pred_test)
