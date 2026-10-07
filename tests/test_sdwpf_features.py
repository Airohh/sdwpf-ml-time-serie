"""Unit tests for sdwpf feature and horizon helpers (no large CSV needed)."""

import numpy as np
import pandas as pd
import pytest

from sdwpf.constants import ERA5_EXTRA_COLS, SDWPF_V1_ANCHOR, STEPS_PER_DAY
from sdwpf.features import (
    add_features,
    build_feature_columns,
    purge_target_overlap,
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


def test_purge_target_overlap_keeps_targets_strictly_before_next_block():
    dt = pd.Series(pd.date_range(pd.Timestamp(SDWPF_V1_ANCHOR), periods=20, freq="10min"))
    kept = purge_target_overlap(dt, np.arange(10), np.arange(10, 20), horizon_steps=3)
    assert np.array_equal(kept, np.arange(7))
    assert np.array_equal(purge_target_overlap(dt, np.arange(10), np.array([], dtype=int), 3), np.arange(10))


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


def _frame_with_missing_steps(n: int, missing: list[int]) -> pd.DataFrame:
    """10-min series where ``Patv = 10 × step`` and some steps are absent (cleaning, join)."""
    t = pd.date_range(pd.Timestamp(SDWPF_V1_ANCHOR), periods=n, freq="10min")
    df = pd.DataFrame(
        {
            "datetime": t,
            "Patv": np.arange(n, dtype=float) * 10,
            "Wspd": np.arange(n, dtype=float),
            "Wdir": 0.0,
            "Etmp": 20.0,
            "Itmp": 25.0,
        }
    )
    for c in ERA5_EXTRA_COLS:
        df[c] = np.arange(n, dtype=float)
    return df.drop(index=missing).reset_index(drop=True)


def test_add_features_aligns_on_time_not_rows():
    df = _frame_with_missing_steps(30, missing=[10])
    out = add_features(df, max_lag=2, horizon=3).set_index("Wspd")
    # step 7 -> target step 10, absent: NaN (a row shift would return step 11)
    assert np.isnan(out.loc[7.0, "y_target"])
    assert out.loc[8.0, "y_target"] == 110.0
    assert np.isnan(out.loc[11.0, "patv_lag1"])
    assert out.loc[11.0, "patv_lag2"] == 90.0
    assert out.loc[12.0, "patv_lag1"] == 110.0
    assert np.isnan(out.loc[12.0, "wspd_lag2"])


def test_add_features_meteo_lags_align_on_time():
    df = _frame_with_missing_steps(30, missing=[10])
    out = add_features(
        df, max_lag=0, horizon=3, calendar_encoding=True, meteo_max_lag=2
    ).set_index("Wspd")
    assert np.isnan(out.loc[11.0, "Wspd_w_lag1"])
    assert out.loc[11.0, "Wspd_w_lag2"] == 9.0


def test_add_features_rejects_duplicate_datetimes():
    df = _frame_with_missing_steps(10, missing=[])
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique datetimes"):
        add_features(df, max_lag=1, horizon=1)


def test_calendar_describes_target_time():
    df = _frame_with_missing_steps(30, missing=[])
    out = add_features(df, max_lag=0, horizon=6, calendar_encoding=True)
    target_time = pd.to_datetime(df["datetime"]) + pd.Timedelta(hours=1)
    minutes = target_time.dt.hour * 60 + target_time.dt.minute
    assert np.allclose(out["hour_sin"], np.sin(2 * np.pi * minutes / 1440))


def test_era5_at_target_reads_weather_at_t_plus_h():
    df = _frame_with_missing_steps(30, missing=[10])
    out = add_features(
        df, max_lag=0, horizon=3, calendar_encoding=True, era5_at_target=True
    ).set_index("Wspd")
    assert out.loc[5.0, "Wspd_w_at_target"] == 8.0
    assert np.isnan(out.loc[7.0, "Wspd_w_at_target"])
    assert out.loc[5.0, "Wspd_w_cube_at_target"] == 512.0
    cols = build_feature_columns(
        out.reset_index(),
        max_lag=0,
        no_patv_now=True,
        no_patv_lags=True,
        time_meteo_only=True,
        era5_at_target=True,
    )
    assert "Wspd_w_at_target" in cols and "Wdir_w_sin_at_target" in cols


def test_era5_at_target_requires_meteo_mode():
    df = _frame_with_missing_steps(10, missing=[])
    with pytest.raises(ValueError, match="calendar_encoding"):
        add_features(df, max_lag=1, horizon=1, era5_at_target=True)


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
