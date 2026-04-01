"""Unit tests for sdwpf feature and horizon helpers (no large CSV needed)."""

import numpy as np
import pandas as pd
import pytest

from sdwpf.constants import SDWPF_V1_ANCHOR, STEPS_PER_DAY
from sdwpf.constants import ERA5_EXTRA_COLS
from sdwpf.features import (
    add_features,
    build_feature_columns,
    resolve_horizon_steps,
    temporal_split,
    temporal_split_train_val_test,
    walk_forward_indices,
)


def test_resolve_horizon_steps_hours_days():
    assert resolve_horizon_steps(1, 1.0, None) == 6 * 1  # 1 h
    assert resolve_horizon_steps(1, None, 1.0) == STEPS_PER_DAY
    assert resolve_horizon_steps(1, 2.0, 1.0) == 12 + STEPS_PER_DAY


def test_resolve_horizon_steps_fallback():
    assert resolve_horizon_steps(144, None, None) == 144
    assert resolve_horizon_steps(None, None, None) == STEPS_PER_DAY


def test_resolve_horizon_steps_errors():
    with pytest.raises(ValueError):
        resolve_horizon_steps(0, None, None)
    with pytest.raises(ValueError):
        resolve_horizon_steps(1, 0.0, 0.0)


def test_temporal_split():
    tr, te = temporal_split(100, 0.7)
    assert len(tr) == 70 and len(te) == 30


def test_temporal_split_train_val_test():
    tr, va, te = temporal_split_train_val_test(1000, 0.5, 0.2)
    assert len(tr) == 500 and len(va) == 200 and len(te) == 300
    assert tr[-1] == 499 and va[0] == 500 and te[0] == 700


def test_temporal_split_train_val_test_errors():
    with pytest.raises(ValueError):
        temporal_split_train_val_test(100, 0.5, 0.6)


def test_walk_forward_indices():
    folds = walk_forward_indices(100, n_splits=3, test_size=10)
    assert len(folds) == 3
    assert np.array_equal(folds[0][1], np.arange(70, 80))
    assert np.array_equal(folds[1][1], np.arange(80, 90))
    assert np.array_equal(folds[2][1], np.arange(90, 100))
    assert folds[0][0][0] == 0 and folds[0][0][-1] == 69
    assert folds[2][0][-1] == 89


def test_walk_forward_indices_errors():
    with pytest.raises(ValueError):
        walk_forward_indices(10, n_splits=3, test_size=5)


def test_add_features_y_target():
    base = pd.Timestamp(SDWPF_V1_ANCHOR)
    t = pd.date_range(base, periods=20, freq="10min")
    df = pd.DataFrame(
        {
            "datetime": t,
            "Patv": np.arange(20, dtype=float) * 10,
            "Wspd": np.linspace(5, 8, 20),
            "Wdir": 0.0,
            "Etmp": 20.0,
            "Itmp": 25.0,
        }
    )
    out = add_features(df, max_lag=3, horizon=2)
    # y_target[t] = Patv shifted -2; check aligned row after warmup
    row = out.iloc[5]
    assert row["patv_lag1"] == df.loc[4, "Patv"]
    assert row["y_target"] == df.loc[7, "Patv"]


def test_add_features_calendar_encoding():
    base = pd.Timestamp(SDWPF_V1_ANCHOR)
    t = pd.date_range(base, periods=10, freq="10min")
    df = pd.DataFrame(
        {
            "datetime": t,
            "Patv": np.arange(10, dtype=float),
            "Wspd": 5.0,
            "Wdir": 0.0,
            "Etmp": 20.0,
            "Itmp": 25.0,
        }
    )
    out = add_features(df, max_lag=0, horizon=1, calendar_encoding=True)
    assert "hour_sin" in out.columns and "doy_cos" in out.columns
    assert out["hour_sin"].notna().all()


def test_build_feature_columns_time_meteo_only():
    base = pd.Timestamp(SDWPF_V1_ANCHOR)
    t = pd.date_range(base, periods=5, freq="10min")
    row = {
        "datetime": t,
        "Patv": [1.0, 2, 3, 4, 5],
        "Wspd": [5.0] * 5,
        "Wdir": [0.0] * 5,
        "Etmp": [20.0] * 5,
        "Itmp": [25.0] * 5,
        "hour_sin": [0.0] * 5,
        "hour_cos": [1.0] * 5,
        "doy_sin": [0.1] * 5,
        "doy_cos": [0.9] * 5,
        "y_target": [2.0, 3, 4, 5, np.nan],
    }
    for c in ERA5_EXTRA_COLS:
        row[c] = [0.5] * 5
    feat = pd.DataFrame(row)
    cols = build_feature_columns(
        feat, max_lag=0, no_patv_now=True, no_patv_lags=True, time_meteo_only=True
    )
    assert cols[0] == "hour_sin"
    assert all(c in cols for c in ERA5_EXTRA_COLS)
    assert "wspd_lag1" not in cols
