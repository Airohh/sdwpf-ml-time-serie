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
    build_modeling_frame,
    describe_feature_mode,
    evaluate_on_indices,
    train_and_evaluate,
    train_and_evaluate_pooled,
)
from sdwpf.xgb_device import XGB_DEVICE_ENV, resolve_xgb_device_spec

pytest.importorskip("xgboost")


def test_resolve_xgb_device_spec_accepts_cuda_ordinal():
    assert resolve_xgb_device_spec("cuda:0") == "cuda:0"
    assert resolve_xgb_device_spec("CUDA:3") == "cuda:3"
    with pytest.raises(ValueError):
        resolve_xgb_device_spec("gpu")


def test_resolve_xgb_device_spec_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv(XGB_DEVICE_ENV, raising=False)
    assert resolve_xgb_device_spec(None) == "cpu"
    monkeypatch.setenv(XGB_DEVICE_ENV, "auto")
    assert resolve_xgb_device_spec(None) == "auto"


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
    df = _synthetic_scada_merged(2000, rng)
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
        2 + len(ERA5_EXTRA_COLS) * (1 + DEFAULT_METEO_MAX_LAG) + 3
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
    df1 = _synthetic_scada_merged(2000, rng)
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


def test_build_modeling_frame_target_is_exactly_h_ahead_despite_gaps():
    rng = np.random.default_rng(4)
    n = 1200
    df = _synthetic_scada_merged(n, rng)
    df["Patv"] = np.arange(n, dtype=float)  # Patv encodes the step index
    keep = rng.random(n) > 0.02
    df = df.loc[keep].reset_index(drop=True)
    for meteo in (True, False):
        feat, _ = build_modeling_frame(
            df,
            horizon_steps=STEPS_PER_DAY,
            no_patv_now=meteo,
            no_patv_lags=meteo,
            time_meteo_only=meteo,
        )
        assert len(feat) > 0
        assert (feat["y_target"] - feat["Patv"] == STEPS_PER_DAY).all()
        lead = pd.to_datetime(feat["datetime"]) + pd.Timedelta(days=1)
        assert lead.isin(pd.to_datetime(df["datetime"])).all()


def test_era5_at_target_is_an_upper_bound_when_power_follows_target_weather():
    rng = np.random.default_rng(5)
    n = 3000
    df = _synthetic_scada_merged(n, rng)
    wind = np.clip(8 + np.cumsum(rng.normal(0, 0.3, n)), 0, 25)
    df["Wspd_w"] = wind
    df["Patv"] = np.clip(1500 * ((wind - 3) / 9) ** 3, 0, 1500) * (wind >= 3)
    kw = dict(
        horizon_steps=STEPS_PER_DAY,
        no_patv_now=True,
        no_patv_lags=True,
        time_meteo_only=True,
        xgb_device="cpu",
    )
    base = train_and_evaluate(df, **kw)
    oracle = train_and_evaluate(df, era5_at_target=True, **kw)
    assert "Wspd_w_at_target" in oracle.feature_cols
    assert "Wspd_w_at_target" not in base.feature_cols
    assert "doy_sin" not in oracle.feature_cols
    assert oracle.xgboost_test_mae < 0.3 * base.naive_mean_train_test_mae
    assert base.power_curve_test_mae is None
    assert oracle.power_curve_test_mae < 0.3 * oracle.naive_mean_train_test_mae


def test_purge_removes_train_rows_whose_target_reaches_the_test_block():
    rng = np.random.default_rng(7)
    df = _synthetic_scada_merged(1600, rng)
    kw = dict(horizon_steps=STEPS_PER_DAY, max_lag=4, xgb_device="cpu")
    res = train_and_evaluate(df, **kw)
    assert res.n_purged == STEPS_PER_DAY
    feat, cols = build_modeling_frame(df, horizon_steps=STEPS_PER_DAY, max_lag=4)
    n_tr = int(len(feat) * 0.7)
    tr, te = np.arange(n_tr), np.arange(n_tr, len(feat))
    no_purge = evaluate_on_indices(
        feat, cols, horizon_steps=STEPS_PER_DAY, patv_feature_mode="", train_idx=tr,
        val_idx=np.array([], dtype=np.int64), test_idx=te, xgb_device="cpu", purge=False,
    )
    assert no_purge.n_purged == 0


def test_climatology_baseline_captures_diurnal_cycle():
    rng = np.random.default_rng(8)
    df = _synthetic_scada_merged(3000, rng)
    hour = pd.to_datetime(df["datetime"]).dt.hour.to_numpy()
    df["Patv"] = 500 + 400 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 20, len(df))
    res = train_and_evaluate(
        df, horizon_steps=STEPS_PER_DAY, no_patv_now=True, no_patv_lags=True,
        time_meteo_only=True, xgb_device="cpu",
    )
    assert res.climatology_test_mae is not None
    assert res.climatology_test_mae < 0.2 * res.naive_mean_train_test_mae


def test_build_modeling_frame_day_of_year_is_opt_in():
    rng = np.random.default_rng(6)
    df = _synthetic_scada_merged(600, rng)
    kw = dict(horizon_steps=STEPS_PER_DAY, no_patv_now=True, no_patv_lags=True, time_meteo_only=True)
    _, cols = build_modeling_frame(df, **kw)
    _, cols_doy = build_modeling_frame(df, include_day_of_year=True, **kw)
    assert "doy_sin" not in cols and "hour_sin" in cols
    assert {"doy_sin", "doy_cos"} <= set(cols_doy)


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
    df = _synthetic_scada_merged(2000, rng)
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
    patv = describe_feature_mode(
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
    df = _synthetic_scada_merged(2000, rng)
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
